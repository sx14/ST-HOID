import os
import copy
import json
from math import log, e

import pickle
import yaml
import numpy as np
import torch
from torch.autograd import Variable

from datasets.vidor import VidOR
from models.fcnet import FCNet


class Tester:

    def __init__(self, dataset, model, all_trajs, seg_len,
                 max_per_video, feat_root, output_root, use_gpu=True):
        self.dataset = dataset
        self.model = model
        self.all_trajs = all_trajs
        self.seg_len = seg_len
        self.max_per_video = max_per_video
        self.feat_root = feat_root
        self.output_root = output_root
        self.use_gpu = use_gpu

        if not os.path.exists(output_root):
            os.makedirs(output_root)

        if use_gpu:
            model.cuda()

    def generate_relation_segments(self, sbj, obj):
        video_h = sbj['height']
        video_w = sbj['width']

        sbj_traj = sbj['trajectory']
        sbj_fids = sorted([int(fid) for fid in sbj_traj.keys()])
        sbj_stt_fid = sbj_fids[0]
        sbj_end_fid = sbj_fids[-1]
        sbj_cls = sbj['category']
        sbj_scr = sbj['score']
        sbj_tid = sbj['tid']

        obj_traj = obj['trajectory']
        obj_fids = sorted([int(fid) for fid in obj_traj.keys()])
        obj_stt_fid = obj_fids[0]
        obj_end_fid = obj_fids[-1]
        obj_cls = obj['category']
        obj_scr = obj['score']
        obj_tid = obj['tid']

        rela_segments = []
        if sbj_end_fid < obj_stt_fid or sbj_stt_fid > obj_end_fid:
            # no temporal intersection
            return rela_segments

        # intersection
        i_stt_fid = max(sbj_stt_fid, obj_stt_fid)
        i_end_fid = min(sbj_end_fid, obj_end_fid)

        added_seg_ids = set()
        for seg_fid in range(i_stt_fid, i_end_fid + 1):
            seg_id = int(seg_fid / self.seg_len)
            if seg_id in added_seg_ids:
                continue
            seg_stt_fid = max(seg_id * self.seg_len, i_stt_fid)
            seg_end_fid = min(seg_id * self.seg_len + self.seg_len - 1, i_end_fid)
            seg_dur = seg_end_fid - seg_stt_fid + 1

            if seg_dur >= 2:

                seg_sbj_traj = {}
                seg_obj_traj = {}

                for fid in range(seg_stt_fid, seg_end_fid + 1):
                    seg_sbj_traj['%06d' % fid] = sbj_traj['%06d' % fid]
                    seg_obj_traj['%06d' % fid] = obj_traj['%06d' % fid]

                seg = {'seg_id': seg_id,
                       'sbj_traj': seg_sbj_traj,
                       'obj_traj': seg_obj_traj,
                       'sbj_cls': sbj_cls,
                       'obj_cls': obj_cls,
                       'sbj_scr': sbj_scr,
                       'obj_scr': obj_scr,
                       'sbj_tid': sbj_tid,
                       'obj_tid': obj_tid,
                       'vid_h': video_h,
                       'vid_w': video_w,
                       'connected': False}
                rela_segments.append(seg)
                added_seg_ids.add(seg_id)
        return rela_segments

    def ext_language_feat(self, rela_segs):
        lan_feat = np.zeros((len(rela_segs), self.dataset.obj_vecs.shape[1] * 2))
        for i, rela_seg in enumerate(rela_segs):
            sbj_cate_idx = self.dataset.obj_cate2idx[rela_seg['sbj_cls']]
            obj_cate_idx = self.dataset.obj_cate2idx[rela_seg['obj_cls']]
            sbj_cate_vec = self.dataset.obj_vecs[sbj_cate_idx]
            obj_cate_vec = self.dataset.obj_vecs[obj_cate_idx]
            lan_feat[i] = np.concatenate((sbj_cate_vec, obj_cate_vec))
        return lan_feat

    @staticmethod
    def ext_spatial_feat(rela_segs):

        def cal_spatial_feat(sbj_box, obj_box, img_h, img_w):
            sbj_h = sbj_box[3] - sbj_box[1] + 1
            sbj_w = sbj_box[2] - sbj_box[0] + 1
            obj_h = obj_box[3] - obj_box[1] + 1
            obj_w = obj_box[2] - obj_box[0] + 1
            spatial_feat = [
                sbj_box[0] * 1.0 / img_w,
                sbj_box[1] * 1.0 / img_h,
                sbj_box[2] * 1.0 / img_w,
                sbj_box[3] * 1.0 / img_h,
                (sbj_h * sbj_w * 1.0) / (img_h * img_w),
                obj_box[0] * 1.0 / img_w,
                obj_box[1] * 1.0 / img_h,
                obj_box[2] * 1.0 / img_w,
                obj_box[3] * 1.0 / img_h,
                (obj_h * obj_w * 1.0) / (img_h * img_w),
                (sbj_box[0] - obj_box[0] + 1) / (obj_w * 1.0),
                (sbj_box[1] - obj_box[1] + 1) / (obj_h * 1.0),
                log(sbj_w * 1.0 / obj_w, e),
                log(sbj_h * 1.0 / obj_h, e)]
            spatial_feat = np.array(spatial_feat)
            return spatial_feat

        spa_feat = np.zeros((len(rela_segs), 14 * 3))

        for i, rela_seg in enumerate(rela_segs):
            sbj_traj = rela_seg['sbj_traj']
            obj_traj = rela_seg['obj_traj']

            sbj_stt_idx = sorted(sbj_traj.keys())[0]
            sbj_end_idx = sorted(sbj_traj.keys())[-1]
            sbj_stt_box = sbj_traj[sbj_stt_idx]
            sbj_end_box = sbj_traj[sbj_end_idx]

            obj_stt_idx = sorted(obj_traj.keys())[0]
            obj_end_idx = sorted(obj_traj.keys())[-1]
            obj_stt_box = obj_traj[obj_stt_idx]
            obj_end_box = obj_traj[obj_end_idx]

            width = rela_seg['vid_w']
            height = rela_seg['vid_h']

            spa_feat_stt = cal_spatial_feat(sbj_stt_box, obj_stt_box, width, height)
            spa_feat_end = cal_spatial_feat(sbj_end_box, obj_end_box, width, height)
            spa_feat_dif = spa_feat_end - spa_feat_stt
            spa_feat[i] = np.concatenate((spa_feat_stt, spa_feat_end, spa_feat_dif))

        return spa_feat

    @staticmethod
    def ext_toi_feat(rela_segs, tid2feat):
        feat_len = tid2feat[rela_segs[0]['sbj_tid']].shape[-1]
        body_part_num = tid2feat[rela_segs[0]['sbj_tid']].shape[1] - 1

        sbj_feat = np.zeros((len(rela_segs), feat_len))
        obj_feat = np.zeros((len(rela_segs), feat_len))
        body_feat = np.zeros((len(rela_segs), feat_len * body_part_num))

        for i, rela_seg in enumerate(rela_segs):
            sbj_feat[i] = tid2feat[rela_seg['sbj_tid']][rela_seg['seg_id'], 0]
            obj_feat[i] = tid2feat[rela_seg['obj_tid']][rela_seg['seg_id'], 0]
            body_feat[i] = tid2feat[rela_seg['sbj_tid']][rela_seg['seg_id'], 1:].reshape(-1)

        return sbj_feat, obj_feat, body_feat

    def predict_predicate(self, rela_segments, tid2feat):
        if len(rela_segments) == 0:
            return rela_segments

        lan_feat = self.ext_language_feat(rela_segments)
        spa_feat = self.ext_spatial_feat(rela_segments)
        sbj_feat, obj_feat, body_feat = self.ext_toi_feat(rela_segments, tid2feat)

        sbj_feat_v = Variable(torch.from_numpy(sbj_feat)).float()
        obj_feat_v = Variable(torch.from_numpy(obj_feat)).float()
        lan_feat_v = Variable(torch.from_numpy(lan_feat)).float()
        spa_feat_v = Variable(torch.from_numpy(spa_feat)).float()
        body_feat_v = Variable(torch.from_numpy(body_feat)).float()

        if self.use_gpu:
            sbj_feat_v = sbj_feat_v.cuda()
            obj_feat_v = obj_feat_v.cuda()
            lan_feat_v = lan_feat_v.cuda()
            spa_feat_v = spa_feat_v.cuda()
            body_feat_v = body_feat_v.cuda()

        probs, _ = self.model(sbj_feat_v, obj_feat_v, body_feat_v, lan_feat_v, spa_feat_v)
        if self.use_gpu:
            probs = probs.cpu()
        probs = probs.data.numpy()
        all_rela_segments = [[] for _ in range(len(rela_segments))]

        # get top 10 predictions
        for i in range(probs.shape[0]):
            rela_probs = probs[i]
            rela_cls_top10 = np.argsort(rela_probs)[::-1][:10]
            rela_seg = rela_segments[i]
            for t in range(10):
                rela_seg_copy = copy.deepcopy(rela_seg)
                pred_pre_idx = rela_cls_top10[t]
                pred_pre_scr = rela_probs[pred_pre_idx]
                pred_pre = self.dataset.pre_cates[pred_pre_idx]
                rela_seg_copy['pre_cls'] = pred_pre
                rela_seg_copy['pre_scr'] = pred_pre_scr
                all_rela_segments[i].append(rela_seg_copy)

        return all_rela_segments

    @staticmethod
    def greedy_association(rela_cand_segments):
        if len(rela_cand_segments) == 0:
            return []

        rela_instances = []
        for i in range(len(rela_cand_segments)):
            curr_segments = rela_cand_segments[i]

            for j in range(len(curr_segments)):
                # current
                curr_segment = curr_segments[j]
                curr_scores = [curr_segment['pre_scr']]
                if curr_segment['connected']:
                    continue
                else:
                    curr_segment['connected'] = True

                for p in range(i + 1, len(rela_cand_segments)):
                    # try to connect next segment
                    next_segments = rela_cand_segments[p]
                    success = False
                    for q in range(len(next_segments)):
                        next_segment = next_segments[q]

                        if next_segment['connected']:
                            continue

                        if curr_segment['pre_cls'] == next_segment['pre_cls']:
                            # merge trajectories
                            curr_sbj = curr_segment['sbj_traj']
                            curr_seg_sbj = next_segment['sbj_traj']
                            curr_sbj.update(curr_seg_sbj)
                            curr_obj = curr_segment['obj_traj']
                            curr_seg_obj = next_segment['obj_traj']
                            curr_obj.update(curr_seg_obj)

                            # record segment predicate scores
                            curr_scores.append(next_segment['pre_scr'])
                            next_segment['connected'] = True
                            success = True
                            break

                    if not success:
                        break

                curr_segment['pre_scr'] = sum(curr_scores) / len(curr_scores)
                rela_instances.append(curr_segment)
        return rela_instances

    @staticmethod
    def filter(rela_cands, max_per_video):
        rela_cands = [rela_cand for rela_cand in rela_cands if rela_cand['pre_cls'] != '__no_interaction__']
        for rela_cand in rela_cands:
            rela_cand['score'] = rela_cand['sbj_scr'] * rela_cand['obj_scr'] * rela_cand['pre_scr']
        sorted_cands = sorted(rela_cands, key=lambda rela: rela['score'], reverse=True)
        return sorted_cands[:max_per_video]

    @staticmethod
    def format(relas):
        format_relas = []
        for rela in relas:
            format_rela = dict()
            format_rela['triplet'] = [rela['sbj_cls'], rela['pre_cls'], rela['obj_cls']]
            format_rela['score'] = rela['score']

            sbj_traj = rela['sbj_traj']
            obj_traj = rela['obj_traj']
            sbj_fid_boxes = sorted(sbj_traj.items(), key=lambda fid_box: int(fid_box[0]))
            obj_fid_boxes = sorted(obj_traj.items(), key=lambda fid_box: int(fid_box[0]))
            stt_fid = int(sbj_fid_boxes[0][0])          # inclusive
            end_fid = int(sbj_fid_boxes[-1][0]) + 1     # exclusive
            format_rela['duration'] = [stt_fid, end_fid]

            format_sbj_traj = [fid_box[1] for fid_box in sbj_fid_boxes]
            format_obj_traj = [fid_box[1] for fid_box in obj_fid_boxes]
            format_rela['sub_traj'] = format_sbj_traj
            format_rela['obj_traj'] = format_obj_traj
            format_relas.append(format_rela)
        return format_relas

    def run_video(self, vid_trajs, tid2feat):

        def get_sbjs_and_objs(ds, trajs):
            sbjs = [traj for traj in trajs if ds.is_subject(traj['category'])]
            objs = trajs
            return sbjs, objs

        # add tid
        for tid, traj in enumerate(vid_trajs):
            vid_trajs[tid]['tid'] = tid

        vid_relas = []
        sbjs, objs = get_sbjs_and_objs(self.dataset, vid_trajs)
        for sbj in sbjs:
            for obj in objs:
                if sbj['tid'] == obj['tid']: continue
                rela_cand_segs = self.generate_relation_segments(sbj, obj)
                rela_cand_segs = self.predict_predicate(rela_cand_segs, tid2feat)
                rela_instances = self.greedy_association(rela_cand_segs)
                vid_relas += rela_instances

        return vid_relas

    @staticmethod
    def load_toi_feat(video_feat_root):
        tid2feat = {}
        feat_files = os.listdir(video_feat_root)
        for feat_file in sorted(feat_files):
            tid = feat_file.split('.')[0]
            feat_path = os.path.join(video_feat_root, feat_file)
            with open(feat_path) as f:
                feat = pickle.load(f)
            tid2feat[int(tid)] = feat
        return tid2feat

    def run(self):
        vid_num = len(self.all_trajs)
        for i, pid_vid in enumerate(sorted(self.all_trajs.keys())):
            pid, vid = pid_vid.split('/')
            print('[%d/%d] %s' % (i+1, vid_num, vid))

            vid_save_path = os.path.join(self.output_root, vid + '.json')
            if os.path.exists(vid_save_path):
                with open(vid_save_path) as f:
                    json.load(f)
                continue

            vid_feat_root = os.path.join(self.feat_root, pid, vid)
            tid2feat = self.load_toi_feat(vid_feat_root)
            vid_relas = self.run_video(self.all_trajs[pid_vid], tid2feat)
            vid_relas = self.filter(vid_relas, self.max_per_video)
            vid_relas = self.format(vid_relas)

            with open(vid_save_path, 'w') as f:
                json.dump({vid: vid_relas}, f)


if __name__ == '__main__':
    exp = 'fc'
    dataset_name = 'vidor_hoid_mini'
    dataset_root = '../data/%s' % dataset_name
    cfg_path = 'cfgs/%s_%s.yaml' % (dataset_name, exp)
    with open(cfg_path) as f:
        cfg = yaml.load(f)

    # load detected trajectories
    print('Loading trajectory detections ...')
    test_traj_det_path = '../data/%s/%s' % (dataset_name, cfg['test_traj_det'])
    with open(test_traj_det_path) as f:
        test_trajs = json.load(f)['results']

    # load model
    print('Loading model ...')
    dataset = VidOR(dataset_name, dataset_root, cfg['test_split'], '../cache')
    model = FCNet(dataset.category_num('predicate'))
    model_weight_path = os.path.join(cfg['weight_root'], cfg['dataset'], cfg['exp'],
                                     '%s_%d.pkl' % (model.name, cfg['test_epoch']))
    model.load_weight(model_weight_path)
    model.eval()

    # init tester
    print('---- Testing start ----')
    feat_root = '../data/%s/feat_pr/%s' % (dataset_name, cfg['test_split'])
    output_root = os.path.join(cfg['test_output_root'], dataset_name)
    tester = Tester(dataset, model, test_trajs, cfg['seg_len'],
                    cfg['test_max_per_video'], feat_root, output_root, cfg['use_gpu'])
    tester.run()
