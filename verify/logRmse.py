import torch


def log_rmse(flag, net, x, y, lossfunction):
    if flag == 1: # valid 数据集
        net.eval()
    output = net(x)
    result = torch.max(output, 1)[1].numpy()
    label_y = torch.max(y, 1)[1].data.numpy()
    accuracy = 100 * (result == label_y).sum() / len(label_y)
    loss = lossfunction(output, y)
    net.train()

    return (loss.data.item(), accuracy)


def log_rmse_batch(flag, net, dataLoader, lossfunction):
    if flag == 1:   # valid 数据集
        net.eval()
    y = None
    pred = None
    for step, (seq, target) in enumerate(dataLoader):
        if y is None:
            y = target
        else:
            y = torch.cat([y, target], dim=0)
        with torch.no_grad():
            y_pred = net(seq)
            if pred is None:
                pred = y_pred
            else:
                pred = torch.cat([pred, y_pred], dim=0)

    result = torch.max(pred, 1)[1].numpy()
    label_y = torch.max(y, 1)[1].data.numpy()
    accuracy = 100 * (result == label_y).sum() / len(label_y)
    loss = lossfunction(pred, y)
    net.train()
    return (loss.data.item(), accuracy)