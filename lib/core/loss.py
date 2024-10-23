import torch.nn as nn
import torch.nn.functional as F
import torch
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric
from torch.nn.modules.loss import _Loss
from lib.models.YOLOX_Loss import YOLOX_Loss
#import torch.nn.functional as f
#from typing import optional, list

class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        # loss_list = [BCEcls = FocalLoss, BCEobj = FocalLoss, BCEseg = nn.BCEWithLogitsLoss]
        self.loss_list = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model, imgs):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """

        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model, imgs)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model, imgs):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3, det_head4, det_head5], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device

        Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss = self.loss_list

        # ComputeLossOTA
        det_all_loss = Det_loss(predictions[0], targets[0], imgs)

        # driving area BCE loss 
        # predictions[1] = shape( B 2 H W ) ， 两个channel代表前景（1）与背景（0）
        drive_area_seg_predicts = predictions[1].view(-1)
        # target[1] = shape( B 2 H W ) , dim0=bg, dim1 = road
        drive_area_seg_targets = targets[1].view(-1)
        da_seg_loss = Da_Seg_Loss(drive_area_seg_predicts, drive_area_seg_targets)

        # lane line focal loss
        # predictions[2] = shape( B 2 H W ) ， 两个channel代表前景（1）与背景（0）
        lane_line_seg_predicts = predictions[2].view(-1)
        # target[2] = shape( B 2 H W ) 
        lane_line_seg_targets = targets[2].view(-1)
        ll_seg_loss = Ll_Seg_Loss(lane_line_seg_predicts, lane_line_seg_targets)

        # predictions[1] = shape( B 3 H W ) ， dim0=bg, dim1 = road, dim2 = lane
        tversky_predicts = predictions[2]
        # target[1] = shape( B 3 H W ) ,   dim0=bg, dim1 = road, dim2 = lane
        tversky_targets = targets[2]
        ll_tversky_loss = Tversky_Loss(tversky_predicts, tversky_targets)


        det_all_loss *= 0.02 * self.lambdas[1]
        da_seg_loss *= 0.2 * self.lambdas[2]
        ll_seg_loss *= 0.2 * self.lambdas[3]
        ll_tversky_loss *= 0.2 * self.lambdas[4]
        
        loss = det_all_loss + da_seg_loss + ll_seg_loss + ll_tversky_loss

        return loss, (det_all_loss.item(), da_seg_loss.item(), ll_seg_loss.item(), ll_tversky_loss.item(), loss.item())


def get_loss(cfg, device, model):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # ComputeLossOTA
    Det_loss = YOLOX_Loss(device, 1)

    # segmentation loss criteria
    Da_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)

    Ll_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)

    # Tversky_Loss criteria
    Tversky_Loss = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0 / 3).to(device)

    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0.0:
        Ll_Seg_Loss = FocalLossSeg(Ll_Seg_Loss, gamma)

    loss_list = [Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLossSeg(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLossSeg, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        p_t = torch.exp(-loss)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class DiceLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        ch = y_true.size(1)
        dims = (0 , 2)
        y_pred = y_pred.view(bs, ch, -1)
        y_true = y_true.view(bs, ch, -1)

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        pass

def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)

    return tversky_score

class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Positives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """
    def __init__(
        self,
        smooth: float = 0.0,
        eps: float = 1e-7,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.0,
    ):
        super().__init__(smooth, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)
