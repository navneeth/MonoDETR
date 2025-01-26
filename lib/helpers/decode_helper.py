import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    Decode raw detection outputs into meaningful 2D and 3D bounding box predictions.

    Parameters:
        dets (numpy.ndarray): Array of shape [batch x max_dets x dim], containing raw detection outputs.
        info (dict): Dictionary containing metadata for input images, with keys:
            - 'img_size' (list of tuples): Image dimensions for each batch item [(width, height), ...].
            - 'img_id' (list): List of image IDs for the batch.
        calibs (list): List of calibration objects for each image in the batch. Each calibration object should
            implement methods like `img_to_rect()` and `alpha2ry()`.
        cls_mean_size (numpy.ndarray): Array containing mean sizes (height, width, length) for each object class.
        threshold (float): Confidence threshold. Detections with scores below this value are ignored.

    Returns:
        dict: A dictionary where keys are image IDs and values are lists of decoded detections. 
              Each detection is represented as:
              [cls_id, alpha, x_min, y_min, x_max, y_max, h, w, l, x, y, z, ry, score]

    Detection Format:
        Each decoded detection contains the following attributes:
        - cls_id (int): The class ID of the detected object (e.g., car, pedestrian, etc.).
        - alpha (float): Observation angle of the object in radians, relative to the camera's centerline.
        - x_min (float): Left coordinate of the 2D bounding box in image space.
        - y_min (float): Top coordinate of the 2D bounding box in image space.
        - x_max (float): Right coordinate of the 2D bounding box in image space.
        - y_max (float): Bottom coordinate of the 2D bounding box in image space.
        - h (float): Height of the object in meters (3D dimensions).
        - w (float): Width of the object in meters (3D dimensions).
        - l (float): Length of the object in meters (3D dimensions).
        - x (float): X-coordinate of the object's center in 3D space (left-right position).
        - y (float): Y-coordinate of the object's center in 3D space (vertical position).
        - z (float): Z-coordinate of the object's center in 3D space (depth from the camera).
        - ry (float): Rotation angle of the object around the vertical axis (Y-axis) in radians.
        - score (float): Confidence score of the detection.

    Note:
        - The 2D bounding box is decoded using normalized coordinates and scaled to the input image size.
        - The 3D bounding box is decoded using depth, dimensions, and object center location in real-world space.
        - `alpha` is the observation angle, while `ry` is the object's rotation angle in 3D space.
    '''
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2D bounding box decoding
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            # 3D bounding box decoding
            # depth decoding
            depth = dets[i, j, 6]
            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            # positions decoding
            x3d = dets[i, j, 34] * info['img_size'][i][0]
            y3d = dets[i, j, 35] * info['img_size'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2  # Adjust for the object's height

            # Heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)

            # Adjust score
            score = score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


def extract_dets_from_outputs(outputs, K=50, topk=50):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]
    
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)


    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

    xs3d = boxes[:, :, 0: 1] 
    ys3d = boxes[:, :, 1: 2] 

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes) 
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]
    
    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

    return detections


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)
