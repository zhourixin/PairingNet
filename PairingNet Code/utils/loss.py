import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.55, gamma=8, size_average=True):
        """
        focal_loss, -α(1-yi)**γ *ce_loss(xi,yi)

        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, gt_mask, pad_mask_with_gt):
        """
        :param preds: predicted similarity matrix [bs, N, N]
        :param gt_mask: positive labels [N, N]
        :param pad_mask_with_gt: positive and padded labels [N, N]
        :return: positive and negative loss, positive loss
        """

        loss_p = - self.alpha * (1 - preds[gt_mask]) ** self.gamma * \
            torch.log(preds[gt_mask] + torch.tensor([1e-9]).cuda())


        loss_n = - (1 - self.alpha) * preds[~pad_mask_with_gt] ** self.gamma * torch.log(
            1 - preds[~pad_mask_with_gt] + torch.tensor([1e-9]).cuda())

        if self.size_average:
            loss_np = torch.cat((loss_p, loss_n), 0).mean()
            loss_p = loss_p.mean()
        else:
            loss_np = torch.cat((loss_p, loss_n), 0).sum()
            loss_p = loss_p.sum()

        return 400 * loss_np, loss_p

def ori_focal_loss(conv_m_pre, mask, mask_):
    loss_m0 = -(1 - conv_m_pre[mask]) ** 2 * torch.log(conv_m_pre[mask] + torch.tensor([1e-9]).cuda())
    loss_m1 = -conv_m_pre[mask_] ** 2 * torch.log(1 - conv_m_pre[mask_] + torch.tensor([1e-9]).cuda())
    loss_m = 1000 * torch.cat((loss_m0, loss_m1), 0).mean()
    return loss_m, loss_m0.mean(), loss_m1.mean()