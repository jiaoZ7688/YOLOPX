#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.common import Conv, DWConv, GhostConv, RepConv

def meshgrid(*tensors):
        return torch.meshgrid(*tensors)

class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes, # 1
        width=0.75,
        strides=[8, 16, 32],
        in_channels=[128, 256, 512],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes # 1
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        BaseConv = GhostConv if depthwise else Conv

        for i in range(len(in_channels)):
            self.stems.append(
                Conv(
                    c1=int(in_channels[i]),
                    c2=int(256 * width),
                    k=1,
                    s=1,
                    act=True,
                )
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            c1=int(256 * width),
                            c2=int(256 * width),
                            k=3,
                            s=1,
                            act=True,
                        ),
                        Conv(
                            c1=int(256 * width),
                            c2=int(256 * width),
                            k=3,
                            s=1,
                            act=True,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            c1=int(256 * width),
                            c2=int(256 * width),
                            k=3,
                            s=1,
                            act=True,
                        ),
                        Conv(
                            c1=int(256 * width),
                            c2=int(256 * width),
                            k=3,
                            s=1,
                            act=True,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        self.strides = strides
        # self.scales = [nn. Parameter(torch.FloatTensor([1.0]), requires_grad = True) for _ in range(4)]

    def initialize_biases(self, prior_prob = 1e-2):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # labels: bboxs[ img1_bboxs [ gt1, gt2, gt3 ], img2_bboxs [ gt1, gt2, gt3 ], img3_bboxs [ gt1, gt2, gt3 ], ...... ]
    def forward(self, xin):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        xin = xin[4:]

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # print(output.shape)

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
            # outputs: [ layer_num, [b, c, h,w]]
            outputs.append(output)

        if self.training:
            return outputs
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            inf_outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1) # b, 4*h*w, c
            if self.decode_in_inference:
                return self.decode_outputs(inf_outputs, dtype=xin[0].type()), outputs
            else:
                return outputs

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides # xy投影到输入图像
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides # wh投影到输入图像
        # print(outputs.size())
        # exit(0)
        # b, 4*h*w, 6
        # 特征图上每一点输出bbox的中心点坐标与wh
        return outputs
