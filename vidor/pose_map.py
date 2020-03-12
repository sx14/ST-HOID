import pickle
import numpy as np
from matplotlib import pyplot as plt
import cv2

body_parts = ["head",
              "left_hand",
              "right_hand",
              "hip",
              "left_leg",
              "right_leg"]


key_points = ["nose",
              "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

all_part_kps = {
    'left_leg': ['left_ankle', 'left_knee'],
    'right_leg': ['right_ankle', 'right_knee'],
    'left_hand': ['left_hand', 'left_wrist', 'left_elbow'],
    'right_hand': ['right_hand', 'right_wrist', 'right_elbow'],
    'hip': ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'],
    'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
}


def _iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    xmin_i = max(xmin1, xmin2)
    ymin_i = max(ymin1, ymin2)
    xmax_i = min(xmax1, xmax2)
    ymax_i = min(ymax1, ymax2)
    w_i = xmax_i - xmin_i + 1
    h_i = ymax_i - ymin_i + 1
    if w_i > 0 and h_i > 0:
        area_i = w_i * h_i
        return area_i / (area1 + area2 - area_i)
    else:
        return 0


def _est_hand(wrist, elbow):
    hand = np.zeros(3)
    hand[:2] = wrist[:2] - 0.5 * (wrist[:2] - elbow[:2])
    hand[2] = (wrist[2] + elbow[2]) / 2.0
    return hand


def _get_body_part_kps(part, all_kps):
    kp2ind = dict(zip(key_points, range(len(key_points))))
    part_kps = np.zeros((len(all_part_kps[part]), 3))
    for i, kp_name in enumerate(all_part_kps[part]):
        if kp_name == 'left_hand':
            left_wrist = all_kps[kp2ind['left_wrist']]
            left_elbow = all_kps[kp2ind['left_elbow']]
            kp = _est_hand(left_wrist, left_elbow)
        elif kp_name == 'right_hand':
            right_wrist = all_kps[kp2ind['right_wrist']]
            right_elbow = all_kps[kp2ind['right_elbow']]
            kp = _est_hand(right_wrist, right_elbow)
        else:
            kp = all_kps[kp2ind[kp_name]]
        part_kps[i] = kp
    return part_kps


def _get_body_part_alpha(part):
    all_body_part_alpha = {
        'head': 0.2,
        'left_hand': 0.2,
        'right_hand': 0.2,
        'hip': 0.25,
        'left_leg': 0.25,
        'right_leg': 0.25,
    }
    return all_body_part_alpha[part]


def _gen_body_part_box(all_kps, human_wh, part):
    human_w, human_h = human_wh
    human_s = (human_w + human_h) / 2.0

    part_kps = _get_body_part_kps(part, all_kps)
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    conf_sum = 0.0
    for i in range(len(part_kps)):
        xmin = min(xmin, part_kps[i, 0])
        ymin = min(ymin, part_kps[i, 1])
        xmax = max(xmax, part_kps[i, 0])
        ymax = max(ymax, part_kps[i, 1])
        conf_sum += part_kps[i, 2]

    return [xmin - _get_body_part_alpha(part) * human_s,
            ymin - _get_body_part_alpha(part) * human_s,
            xmax + _get_body_part_alpha(part) * human_s,
            ymax + _get_body_part_alpha(part) * human_s,
            conf_sum / len(part_kps)]


def gen_part_boxes(hbox, skeleton, im_hw):
    im_h, im_w = im_hw
    h_xmin, h_ymin, h_xmax, h_ymax = hbox
    h_wh = [h_xmax - h_xmin + 1, h_ymax - h_ymin + 1]

    if skeleton is None:
        part_boxes = []
        for _ in range(len(body_parts)):
            part_boxes.append([h_xmin, h_ymin, h_xmax, h_ymax, 0.01])
        return part_boxes

    part_boxes = []
    for _, body_part in enumerate(body_parts):
        box = _gen_body_part_box(skeleton, h_wh, body_part)

        # check part box
        xmin, ymin, xmax, ymax, conf = box
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, im_w-1)
        ymax = min(ymax, im_h-1)

        if (xmax > xmin) and (ymax > ymin):
            part_boxes.append([xmin, ymin, xmax, ymax, conf])
        else:
            part_boxes.append([h_xmin, h_ymin, h_xmax, h_ymax, 0.01])

    return part_boxes


def _show_boxes(im_path, dets, cls=None, colors=None):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(dets))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):

        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i], linewidth=1.5)
        )
        if cls is not None and len(cls) == len(dets):
            ax.text(bbox[0], bbox[1] - 2,
                    '{}'.format(cls[i]),
                    bbox=dict(facecolor=colors[i], alpha=0.5),
                    fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
    plt.show()
