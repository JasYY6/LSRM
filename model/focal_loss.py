import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
        Ref: https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py
        FL(pt) = -alpha_t(1-pt)^gamma log(pt)
        alpha: 类别权重,常数时，类别权重为:[alpha,1-alpha,1-alpha,...]；列表时，表示对应类别权重
        gamma: 难易分类的样本权重，使得模型更关注难分类的样本
        优点：帮助区分难分类的不均衡样本数据
    """

    def __init__(self, num_classes, alpha=0.25, gamma=2, reduce=True):

        super(FocalLoss, self).__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.reduce = reduce

        if isinstance(alpha,float):
            if alpha is None:
                self.alpha = torch.ones(self.num_classes, 1)
            else:
                self.alpha = torch.zeros(num_classes)
                self.alpha[0] = alpha
                self.alpha[1:] += (1 - alpha)
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, preds, targets):
        "preds:[B,C],targets:[B]"
        preds = preds.view(-1, preds.size(-1))  # [B,C]
        alpha = self.alpha.to(preds.device)
        logpt = F.log_softmax(preds, dim=1)
        pt = F.softmax(preds).clamp(min=0.0001, max=1.0)

        logpt = logpt.gather(1, targets.view(-1, 1))  # 对应类别值
        pt = pt.gather(1, targets.view(-1, 1))
        alpha = alpha.gather(0, targets.view(-1))

        loss = -(1 - pt) ** self.gamma * logpt
        loss = alpha * loss.t()

        if self.reduce:
            return loss.mean()
        else:
            return loss.sum()