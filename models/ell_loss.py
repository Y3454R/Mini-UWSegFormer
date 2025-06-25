# models/ell_loss.py
import torch
import torch.nn.functional as F


def scharr_operator():
    kx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
    ky = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
    return kx.unsqueeze(0).unsqueeze(0), ky.unsqueeze(0).unsqueeze(0)


def edge_loss(pred, target):
    if pred.shape[1] > 1:
        pred = pred.argmax(dim=1)
    pred = pred.unsqueeze(1).float()
    target = target.unsqueeze(1).float()

    kx, ky = scharr_operator().to(pred.device)
    grad_pred = torch.sqrt(
        F.conv2d(pred, kx, padding=1) ** 2 + F.conv2d(pred, ky, padding=1) ** 2
    )
    grad_target = torch.sqrt(
        F.conv2d(target, kx, padding=1) ** 2 + F.conv2d(target, ky, padding=1) ** 2
    )

    return F.l1_loss(grad_pred, grad_target)
