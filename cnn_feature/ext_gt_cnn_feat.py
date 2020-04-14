# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import math
import sys
import pickle
import json
import numpy as np
import argparse
import pprint
import time
import cv2
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from pose_map import gen_part_boxes

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vidor_hoid_mini', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--data_root', dest='data_root',
                        help='directory to load images for demo',
                        default="../data/vidor_hoid_mini")
    parser.add_argument('--split', dest='split',
                        default="train")
    parser.add_argument('--cuda', dest='cuda',
                        default=True,
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=6, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=18131, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--scene', dest='scene',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


def is_human(cate):
    return cate in {'adult', 'child', 'baby'}


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'ho-rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray([str(i) for i in range(30)])

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        exit(-1)

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()
    fasterRCNN.eval()

    image_root = os.path.join(args.data_root, 'Data', 'VID', args.split)
    feat_root = os.path.join(args.data_root, 'feat_gt', args.split)
    if args.split == 'val':
        split = 'validation'
    elif args.split == 'train':
        split = 'training'
    anno_root = os.path.join(args.data_root, 'anno_with_pose', split)

    pool_feat_size = 4
    pool_feat_chnl = 2048
    body_part_num = 6
    seg_len = 10
    scene_tid = -1

    print('feature extracting ...')
    for pkg_id in os.listdir(image_root):
        pkg_root = os.path.join(image_root, pkg_id)

        for vid_id in os.listdir(pkg_root):
            print('[%s/%s]' % (pkg_id, vid_id))

            output_dir = os.path.join(feat_root, pkg_id, vid_id)
            if args.resume and os.path.exists(output_dir):
                continue

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            vid_frm_dir = os.path.join(pkg_root, vid_id)
            vid_frm_anno_file_path = os.path.join(anno_root, pkg_id, vid_id+'.json')
            with open(vid_frm_anno_file_path) as f:
                vid_anno = json.load(f)

            width = vid_anno['width']
            height = vid_anno['height']

            frm_list = sorted(os.listdir(vid_frm_dir))
            num_frames = vid_anno['frame_count']
            num_segs = int(math.ceil(num_frames * 1.0 / seg_len))

            tid2feat = {}
            tid2cate = {}

            traj_info_list = vid_anno['subject/objects']
            for traj_info in traj_info_list:
                tid2cate[traj_info['tid']] = traj_info['category']
                if is_human(traj_info['category']):
                    tid2feat[traj_info['tid']] = np.zeros((num_segs,
                                                           1 + body_part_num,
                                                           3,
                                                           pool_feat_chnl,
                                                           pool_feat_size,
                                                           pool_feat_size))
                else:
                    tid2feat[traj_info['tid']] = np.zeros((num_segs, 1,
                                                           3,
                                                           pool_feat_chnl,
                                                           pool_feat_size,
                                                           pool_feat_size))
            # scene
            tid2cate[scene_tid] = '__scene__'
            tid2feat[scene_tid] = np.zeros((num_segs, 1,
                                            3,
                                            pool_feat_chnl,
                                            pool_feat_size,
                                            pool_feat_size))

            trajs = vid_anno['trajectories']
            for frm_idx in range(num_frames):
                seg_idx = int(frm_idx / seg_len)
                seg_frm_idx = frm_idx - seg_idx * seg_len
                boxes = [[frm_det['bbox']['xmin'],
                          frm_det['bbox']['ymin'],
                          frm_det['bbox']['xmax'],
                          frm_det['bbox']['ymax'],
                          1.0]
                         for frm_det in trajs[frm_idx]]

                boxes.append([0, 0, width, height, 1.0])    # scene
                tids = [frm_det['tid'] for frm_det in trajs[frm_idx]]
                tids.append(scene_tid)  # scene
                cates = [tid2cate[tid] for tid in tids]
                has_kps = [False] * len(tids)

                for box_ind in range(len(tids)):
                    if is_human(cates[box_ind]):
                        kps = trajs[frm_idx][box_ind]['kps']
                        if kps is not None:
                            kps_np = np.array(kps).reshape((17, 3))
                            pboxes = gen_part_boxes(boxes[box_ind][:4], kps_np, [height, width])
                            boxes += pboxes
                            has_kps[box_ind] = True

                if len(boxes) == 0:
                    continue

                # Load the demo image
                im_file = os.path.join(vid_frm_dir, frm_list[frm_idx])
                im_in = np.array(imread(im_file))
                if len(im_in.shape) == 2:
                    im_in = im_in[:, :, np.newaxis]
                    im_in = np.concatenate((im_in, im_in, im_in), axis=2)
                # rgb -> bgr
                im_in = im_in[:, :, ::-1]
                im = im_in

                blobs, im_scales = _get_image_blob(im)
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

                # resize boxes
                boxes = np.array(boxes)
                boxes[:, :4] = boxes[:, :4] * im_scales[0]
                boxes = boxes[np.newaxis, :, :]

                im_data_pt = torch.from_numpy(im_blob)
                im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                im_info_pt = torch.from_numpy(im_info_np)
                boxes = torch.from_numpy(boxes)

                im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.data.resize_(boxes.size()).copy_(boxes)
                num_boxes.data.resize_(1).zero_()

                pool5 = fasterRCNN.get_pool5(im_data, im_info, gt_boxes, num_boxes)
                pool5 = pool5.cpu().data.numpy()

                num_entity = len(has_kps)
                human_with_kps_cnt = 0
                for ii in range(len(tids)):
                    entity_feat = pool5[ii: ii+1]

                    if is_human(cates[ii]):
                        if has_kps[ii]:
                            stt_idx = num_entity + body_part_num * human_with_kps_cnt
                            end_idx = stt_idx + body_part_num
                            body_part_feats = pool5[stt_idx: end_idx]
                            entity_feat = np.concatenate((entity_feat, body_part_feats))
                            human_with_kps_cnt += 1
                        else:
                            body_part_feats = np.zeros((body_part_num,
                                                        pool_feat_chnl,
                                                        pool_feat_size,
                                                        pool_feat_size))
                            entity_feat = np.concatenate((entity_feat, body_part_feats))

                    entity_feat0 = tid2feat[tids[ii]][seg_idx]
                    if seg_frm_idx == 0:
                        # start
                        entity_feat0[0] = entity_feat
                    else:
                        # end
                        entity_feat0[2] = entity_feat
                    # max-pooling
                    entity_feat0[1] = np.maximum(entity_feat, entity_feat0[1])
                    tid2feat[tids[ii]][seg_idx] = entity_feat0

            for tid in tid2feat:
                output_path = os.path.join(output_dir, str(tid)+'.bin')
                with open(output_path, 'wb') as f:
                    feat = tid2feat[tid].mean(4).mean(3).astype('float32')
                    # print(feat.shape)
                    pickle.dump(feat, f)





