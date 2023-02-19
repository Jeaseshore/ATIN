import torch.nn as nn

def reduce_loss(loss, reduction):
    reduction_enum = nn.functional._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2, alpha=0.75, reduction='mean', avg_factor=None):
    pred_sigmoid = pred[:, 0]
    target = target[:, 0]
    target = target.type_as(pred)
    pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)

    focal_weight = at * (1 - pt).pow(gamma)
    # binary_cross_entropy_with_logits函数将sigmoid层和binaray_cross_entropy合在一起计算比分开依次计算,有更好的数值稳定性
    loss = nn.functional.binary_cross_entropy(
        pred_sigmoid, target, reduction='none') * focal_weight
    # print(loss)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss