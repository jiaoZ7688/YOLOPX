import torch
from torch import tensor
import torch.nn as nn

from torch.nn import Conv2d

import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
# from lib.models.common import Conv, SPPF, Focus, Concat, Detect, MergeBlock, C3
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.common import Conv, seg_head, PSA_p, MergeBlock
from lib.models.common import Concat, FPN_C2, FPN_C3, FPN_C4, ELANNet, ELANBlock_Head, PaFPNELAN, IDetect, RepConv
# from lib.models.YOLOX_Head_scales import YOLOXHead
from lib.models.YOLOX_Head_scales_noshare import YOLOXHead

# 修改
# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
###### prediction head index
# [2, 16, 28],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx no_use c2
[2, 16, 31],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx use_c2

###### Backbone
[ -1, ELANNet, [True]],   #0

###### PaFPNELAN
[ -1, PaFPNELAN, []],   #1

###### Detect Head
[ -1, YOLOXHead,  [1]], #2 #Detection head

# ###### 渐进式上采样
[ 1, FPN_C3, []],   #3
[ 1, FPN_C4, []],   #4

# segmentation head
[ 4, Conv, [512, 256, 3, 1]],  
[ -1, Upsample, [None, 2, 'bilinear']],  
[ -1, ELANBlock_Head, [256, 128]],
[ -1, Conv, [128, 64, 3, 1]], 
[ -1, Upsample, [None, 2, 'bilinear']], 
[ -1, Conv, [64, 32, 3, 1]], 
[ -1, Upsample, [None, 2, 'bilinear']], 
[ -1, Conv, [32, 16, 3, 1]], 
[ -1, ELANBlock_Head, [16, 8]], 
[ -1, Upsample, [None, 2, 'bilinear']], 
[ -1, Conv, [8, 2, 3, 1]], 
[ -1, seg_head, ['sigmoid']],  #16 segmentation head

# # no use C2
###########
# [ 3, Conv, [256, 128, 3, 1]],    #17
# [ -1, Upsample, [None, 2, 'bilinear']],  
# [ -1, ELANBlock_Head, [128, 64]], 
# [ -1, PSA_p, [64, 64]], 
# [ -1, Conv, [64, 32, 3, 1]], 
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [32, 16, 3, 1]], 
# [ -1, ELANBlock_Head, [16, 8]], 
# [ -1, PSA_p, [8, 8]], 
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [8, 2, 3, 1]], 
# [ -1, seg_head, ['sigmoid']],  #28 segmentation head
###########

# use C2
###########
[ 1, FPN_C2, []],  #17
[ -1, Conv, [256, 128, 3, 1]],    #18
# sum c2 and p3
[ 3, Conv, [256, 128, 3, 1]],  
[ -1, Upsample, [None, 2, 'bilinear']],  
[ [-1, 18], MergeBlock, ["add"]],    #C2 and P3
[ -1, ELANBlock_Head, [128, 64]], 
[ -1, PSA_p, [64, 64]], 
[ -1, Conv, [64, 32, 3, 1]], 
[ -1, Upsample, [None, 2, 'bilinear']], 
[ -1, Conv, [32, 16, 3, 1]], 
[ -1, ELANBlock_Head, [16, 8]], 
[ -1, PSA_p, [8, 8]], 
[ -1, Upsample, [None, 2, 'bilinear']], 
[ -1, Conv, [8, 2, 3, 1]], #
[ -1, seg_head, ['sigmoid']],  #31 segmentation head

# # use C2
# ###########
# [ 1, FPN_C2, []],  #17
# [ -1, Conv, [256, 128, 3, 1]],    #18
# [ -1, Conv, [128, 64, 3, 1]],    #19
# [ 3, Conv, [256, 128, 3, 1]],  
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [128, 64, 3, 1]],  
# [ [-1, 19], MergeBlock, ["cat"]],    #concat C2 and P3

# [ -1, ELANBlock_Head, [128, 64]], 
# [ -1, PSA_p, [64, 64]], 
# [ -1, Conv, [64, 32, 3, 1]], 
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [32, 16, 3, 1]], 
# [ -1, ELANBlock_Head, [16, 8]], 
# [ -1, PSA_p, [8, 8]], 
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [8, 2, 3, 1]], #
# [ -1, seg_head, ['sigmoid']],  #33 segmentation head
# ###########

]

# 修改
class MCnet(nn.Module):
    # block_cfg = YOLOP-list
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        # 27 
        self.det_out_idx = block_cfg[0][0]

        # 63 67
        self.seg_out_idx = block_cfg[0][1:]
        
        # Build model
        # e.g. [ -1, Focus, [3, 32, 3]],   #0
        # i从0开始编号，from_ = -1，block = Focus，args = [3, 32, 3]
        # 注意，block是类，不是str
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is YOLOXHead:
                # detector_index  # 27
                self.detector_index = i

            # *args,参数解码 [3, 32, 3] -> 3, 32, 3
            # 构建一系列模块，实例化block
            block_ = block(*args)

            # 模块索引，模块输入来源索引          
            block_.index, block_.from_ = i, from_

            # 向layers_list中添加block_
            layers.append(block_)

            # [ 6, 4, 14, 10, 23, 17, 20, 23, 25, 26, 26, 25, 23, 20, 17, 2, 37, 45, 51, 55, 57, 58, 59, 59 ]
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, YOLOXHead):
            s = 512  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                
            self.stride = Detector.strides           
            Detector.initialize_biases(1e-2)

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        # block_.index = i, 模块索引, from 0,1,2,3....67
        # block_.from_ = from_，模块输入来源索引, -1 or list[-1, 16] or int(16)
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result
                # x=x.float()
                # m=nn.Softmax(dim=1)
                # m=nn.Sigmoid()
                out.append(x)
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
            # cache[index] = x if block.index in self.save else None

        out.insert(0,det_out)
        # out include (det_out, DD_out, LL_out)
        return out            

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            # elif isinstance(m, RepConv_OREPA):
            #     #print(f" switch_to_deploy")
            #     m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, IDetect):
                m.fuse()
                m.forward = m.fuseforward
        # self.info()
        return self      

def get_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out,SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
 
