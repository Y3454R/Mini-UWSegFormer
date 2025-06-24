# losses.py
import torch
import torch.nn.functional as F


def scharr_edge_map(x):
    # x shape: (B, 1, H, W) or (B, C, H, W)
    device = x.device
    kx = (
        torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    ky = (
        torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    edge_x = F.conv2d(x, kx, padding=1)
    edge_y = F.conv2d(x, ky, padding=1)
    return torch.sqrt(edge_x**2 + edge_y**2)


def edge_loss(pred_mask_probs, gt_mask):
    """
    pred_mask_probs: (B, C, H, W), softmax probabilities over classes
    gt_mask: (B, H, W) integer class labels

    Convert gt_mask to one-hot, compute edges on both, then L1 loss.
    """
    B, C, H, W = pred_mask_probs.shape

    gt_one_hot = F.one_hot(gt_mask, num_classes=C).permute(0, 3, 1, 2).float()

    pred_edges = scharr_edge_map(pred_mask_probs)
    gt_edges = scharr_edge_map(gt_one_hot)

    return F.l1_loss(pred_edges, gt_edges)


# scharr_edge_map(x): Computes the edge map of an input tensor using the Scharr operator (an edge detection filter similar to Sobel but more accurate).

# edge_loss(pred_mask_probs, gt_mask): Computes an edge-aware loss by comparing the edges of the predicted segmentation mask (probabilities) and the ground truth mask.
