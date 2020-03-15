import os
import json
import pickle
from copy import deepcopy
from math import log, e

import numpy as np
from torch.utils.data import Dataset


class VidOR(Dataset):
    def __init__(self, dataset_root, split):

        self.val_ratio = 0.1
        self.dataset_root = dataset_root
        if split == 'train':
            self.data_root = os.path.join(dataset_root, 'Data', 'VID', 'train')
            self.feat_root = os.path.join(dataset_root, 'feat', 'VID', 'train')
            self.anno_root = os.path.join(dataset_root, 'anno_with_pose', 'training')
        elif split == 'val':
            self.data_root = os.path.join(dataset_root, 'Data', 'VID', 'val')
            self.feat_root = os.path.join(dataset_root, 'feat', 'VID', 'val')
            self.anno_root = os.path.join(dataset_root, 'anno_with_pose', 'validation')
        else:
            print('split = "train" or "val"')
            exit(-1)

        self.obj_cates = None
        self.pre_cates = None
        self.obj_cate2idx = None
        self.pre_cate2idx = None
        self.sbj_cates = {'human', 'adult', 'child'}
        self._load_category_sets()

        self.SEG_LEN = 10
        self.all_trajs = None
        self.all_insts = None
        self.all_vid_info = None
        self.all_traj_cates = None
        self.all_inst_count = None
        self._load_annotations()

        self.objvecs = None
        self._load_object_vectors()

        self.curr_pkg_id = -1
        self.curr_vid_id = -1
        self.traj_feats = None

    def __getitem__(self, item):
        inst = self.all_insts[item]
        lan_feat = self._ext_language_feature(inst)
        spa_feat = self._ext_spatial_feature(inst)
        sbj_feat, obj_feat, body_feat, = self._ext_cnn_feature(inst)
        pre_cate = inst['pre_cate']
        return sbj_feat, obj_feat, body_feat, lan_feat, spa_feat, pre_cate

    def __len__(self):
        return len(self.all_insts)

    def is_subject(self, cate):
        if isinstance(cate, str):
            return cate in self.sbj_cates
        else:
            return self.obj_cates[cate] in self.sbj_cates


    def split_self(self):
        self2 = deepcopy(self)
        vid_num = len(self.all_inst_count)
        val_num = int(vid_num * self.val_ratio)

        self2.all_inst_count = self2.all_inst_count[:val_num]
        val_inst_num = sum(self2.all_inst_count)
        self2.all_insts = self2.all_insts[:val_inst_num]

        self.all_inst_count = self.all_inst_count[val_num:]
        self.all_insts = self.all_insts[val_inst_num:]
        return self2

    def category_num(self, target):
        if target == 'predicate':
            return len(self.pre_cates)
        elif target == 'object':
            return len(self.obj_cates)
        else:
            return -1

    def _ext_language_feature(self, inst):
        vid_id = inst['vid_id']
        obj_tid = inst['obj_tid']
        sbj_tid = inst['sbj_tid']
        obj_cate_idx = self.all_traj_cates[vid_id][obj_tid]
        sbj_cate_idx = self.all_traj_cates[vid_id][sbj_tid]
        objvec = self.objvecs[obj_cate_idx]
        sbjvec = self.objvecs[sbj_cate_idx]
        return np.concatenate((objvec, sbjvec))

    def _ext_spatial_feature(self, inst):

        def cal_spatial_feat(sbj_box, obj_box, img_h, img_w):
            sbj_h = sbj_box[3] - sbj_box[1] + 1
            sbj_w = sbj_box[2] - sbj_box[0] + 1
            obj_h = obj_box[3] - obj_box[1] + 1
            obj_w = obj_box[2] - obj_box[0] + 1
            spatial_feat = [
                sbj_box[0] * 1.0 / img_w, sbj_box[1] * 1.0 / img_h,
                sbj_box[2] * 1.0 / img_w, sbj_box[3] * 1.0 / img_h,
                (sbj_h * sbj_w * 1.0) / (img_h * img_w),
                obj_box[0] * 1.0 / img_w, obj_box[1] * 1.0 / img_h,
                obj_box[2] * 1.0 / img_w, obj_box[3] * 1.0 / img_h,
                (obj_h * obj_w * 1.0) / (img_h * img_w),
                (sbj_box[0] - obj_box[0] + 1) / (obj_w * 1.0),
                (sbj_box[1] - obj_box[1] + 1) / (obj_h * 1.0),
                log(sbj_w * 1.0 / obj_w, e), log(sbj_h * 1.0 / obj_h, e)]
            spatial_feat = np.array(spatial_feat)
            return spatial_feat

        vid_id = inst['vid_id']
        stt_fid = inst['stt_fid']
        end_fid = inst['end_fid']
        obj_tid = inst['obj_tid']
        sbj_tid = inst['sbj_tid']

        sbj_stt_box = self.all_trajs[vid_id][sbj_tid][stt_fid]
        sbj_end_box = self.all_trajs[vid_id][sbj_tid][end_fid]

        obj_stt_box = self.all_trajs[vid_id][obj_tid][stt_fid]
        obj_end_box = self.all_trajs[vid_id][obj_tid][end_fid]

        width = self.all_vid_info[vid_id]['width']
        height = self.all_vid_info[vid_id]['height']

        spatial_feat_stt = cal_spatial_feat(sbj_stt_box, obj_stt_box, width, height)
        spatial_feat_end = cal_spatial_feat(sbj_end_box, obj_end_box, width, height)
        spatial_feat_dif = spatial_feat_end - spatial_feat_stt
        return np.concatenate((spatial_feat_stt, spatial_feat_end, spatial_feat_dif))

    def _ext_cnn_feature(self, inst):
        pkg_id = inst['pkg_id']
        vid_id = inst['vid_id']

        tid2traj_feat = {}
        if pkg_id != self.curr_pkg_id or vid_id != self.curr_vid_id:
            feat_dir = os.path.join(self.feat_root, pkg_id, vid_id)
            for traj_feat_file in os.listdir(feat_dir):
                tid = traj_feat_file.split('.')[0]
                traj_feat_file_path = os.path.join(feat_dir, traj_feat_file)
                with open(traj_feat_file_path) as f:
                    traj_feat = pickle.load(f)
                tid2traj_feat[tid] = traj_feat
            self.traj_feats = tid2traj_feat
            self.curr_pkg_id = pkg_id
            self.curr_vid_id = vid_id

        sbj_tid = inst['sbj_tid']
        obj_tid = inst['obj_tid']
        stt_fid = inst['stt_fid']
        seg_idx = int(stt_fid / self.SEG_LEN)
        obj_feat = self.traj_feats[obj_tid][seg_idx]
        sbj_feat = self.traj_feats[sbj_tid][seg_idx][0]
        body_feat = self.traj_feats[sbj_tid][seg_idx][1:].reshape(-1)
        return sbj_feat, obj_feat, body_feat

    def _load_object_vectors(self):
        objvec_path = os.path.join(self.dataset_root, 'object_vectors.mat')
        with open(objvec_path) as f:
            self.objvecs = pickle.load(f)

    def _load_category_sets(self):
        obj_cate_path = os.path.join(self.dataset_root, 'object_labels.txt')
        pre_cate_path = os.path.join(self.dataset_root, 'predicate_labels.txt')

        with open(obj_cate_path) as f:
            # 0 base
            self.obj_cates = [line.strip() for line in f.readlines()]
            self.obj_cate2idx = {cate: idx for idx, cate in enumerate(self.obj_cates)}
        with open(pre_cate_path) as f:
            # 1 base (with "no_interaction")
            self.pre_cates = ['__no_interaction__'] + [line.strip() for line in f.readlines()]
            self.pre_cate2idx = {cate: idx for idx, cate in enumerate(self.pre_cates)}

    @staticmethod
    def _load_trajectories(org_trajs, vid_info):
        vid_len = vid_info['frame_count']
        tid2traj = {}
        tid2dur = {}    # [stt_fid, end_fid)

        for frm_idx, frm_dets in enumerate(org_trajs):
            for det in frm_dets:
                tid = det['tid']
                box = [det['bbox']['xmin'],
                       det['bbox']['ymin'],
                       det['bbox']['xmax'],
                       det['bbox']['ymax']]
                if tid not in tid2traj:
                    tid2traj[tid] = [[-1] * 4] * vid_len
                if tid not in tid2dur:
                    tid2dur[tid] = [frm_idx, vid_len]

                tid2traj[tid][frm_idx] = box
                tid2dur[tid][1] = min(tid2dur[tid][1], frm_idx + 1)

        for tid in tid2traj:
            tid2traj[tid] = np.array(tid2traj[tid]).astype(np.int)
        return tid2traj, tid2dur

    def _load_instances(self, org_insts, pkg_id, vid_id):
        insts = []
        seg_len = self.SEG_LEN
        for org_inst in org_insts:
            sbj_tid = org_inst['subject_tid']
            obj_tid = org_inst['object_tid']
            stt_frm_idx = org_inst['start_fid']
            end_frm_idx = org_inst['end_fid']
            pre_cate = org_inst['predicate']

            for frm_idx in range(stt_frm_idx % seg_len, end_frm_idx % seg_len, seg_len):
                seg_stt_frm_idx = frm_idx
                seg_end_frm_idx = frm_idx + seg_len
                insts.append({
                    'pkg_id': pkg_id,
                    'vid_id': vid_id,
                    'stt_fid': seg_stt_frm_idx,
                    'end_fid': seg_end_frm_idx,
                    'sbj_tid': sbj_tid,
                    'obj_tid': obj_tid,
                    'pre_cate': self.pre_cate2idx[pre_cate]})
        return insts

    def _gen_negative_instances(self, pos_insts, tid2dur, tid2cate, pkg_id, vid_id):

        # collect candidates
        cands = {}
        for sbj_tid in tid2dur:
            if not self.is_subject(tid2cate[sbj_tid]):
                continue
            sbj_stt_frm_idx, sbj_end_frm_idx = tid2dur[sbj_tid]
            for obj_tid in tid2dur:
                if obj_tid == sbj_tid:
                    continue
                obj_stt_frm_idx, obj_end_frm_idx = tid2dur[obj_tid]

                cand_stt_frm_idx = max(sbj_stt_frm_idx, obj_stt_frm_idx)
                cand_end_frm_idx = min(sbj_end_frm_idx, obj_end_frm_idx)
                if cand_end_frm_idx > cand_stt_frm_idx:
                    cands['%d-%d' % (sbj_tid, obj_tid)] = np.zeros(cand_end_frm_idx)
                    cands['%d-%d' % (sbj_tid, obj_tid)][cand_stt_frm_idx: cand_end_frm_idx] = 1

        # mark positive instances
        for pos_inst in pos_insts:
            sbj_tid = pos_inst['subject_tid']
            obj_tid = pos_inst['object_tid']
            stt_frm_idx = pos_inst['start_fid']
            end_frm_idx = pos_inst['end_fid']
            cands['%d-%d' % (sbj_tid, obj_tid)][stt_frm_idx: end_frm_idx] = 0

        # generate negative instances
        neg_insts = []
        for sid_oid, cand_dur in cands.items():
            sid, oid = sid_oid.split('-')
            sbj_tid = int(sid)
            obj_tid = int(oid)
            for frm_idx in range(0, cand_dur.shape[0] % self.SEG_LEN, self.SEG_LEN):
                if cand_dur[frm_idx: frm_idx+self.SEG_LEN] == self.SEG_LEN:
                    neg_insts.append({
                        'pkg_id': pos_inst,
                        'vid_id': vid_id,
                        'stt_fid': frm_idx,
                        'end_fid': frm_idx + self.SEG_LEN,
                        'sbj_tid': sbj_tid,
                        'obj_tid': obj_tid,
                        'pre_cate': self.pre_cate2idx['__no_interaction__']})
        return neg_insts

    def _load_annotations(self):

        cache_path = 'data_cache.bin'
        if os.path.exists(cache_path):
            print('%s found! loading ...')
            with open(cache_path) as f:
                data_cache = pickle.load(f)

            self.all_trajs = data_cache['all_trajs']
            self.all_insts = data_cache['all_insts']
            self.all_vid_info = data_cache['all_vid_info']
            self.all_traj_cates = data_cache['all_traj_cates']
            self.all_inst_count = data_cache['all_inst_count']
            return

        self.all_vid_info = {}
        self.all_trajs = {}
        self.all_traj_cates = {}
        self.all_insts = []
        self.all_inst_count = []

        for pkg_id in sorted(os.listdir(self.anno_root)):
            pkg_root = os.path.join(self.anno_root, pkg_id)
            for vid_anno_file in sorted(os.listdir(pkg_root)):
                vid_id = vid_anno_file.split('.')[0]
                vid_anno_path = os.path.join(pkg_root, vid_anno_file)
                with open(vid_anno_path) as f:
                    vid_anno = json.load(f)
                vid_info = {'frame_count': vid_anno['frame_count'],
                            'width': vid_anno['width'],
                            'height': vid_anno['height'],
                            'vid_id': vid_id,
                            'pkg_id': pkg_id}
                tid2traj, tid2dur = self._load_trajectories(vid_anno['trajectories'], vid_info)
                pos_insts = self._load_instances(vid_anno['relation_instances'], pkg_id, vid_id)
                tid2cate_idx = {traj_info['tid']: self.obj_cate2idx[traj_info['category']]
                                for traj_info in vid_anno['subject/objects']}
                neg_insts = self._gen_negative_instances(vid_anno['relation_instances'],
                                                         tid2dur, tid2cate_idx, pkg_id, vid_id)
                vid_insts = pos_insts + neg_insts[:3 * len(pos_insts)]
                self.all_vid_info[vid_id] = vid_info
                self.all_trajs[vid_id] = tid2traj
                self.all_traj_cates[vid_id] = tid2cate_idx
                self.all_insts += vid_insts
                self.all_inst_count.append(len(vid_insts))

        with open(cache_path) as f:
            pickle.dump({'all_trajs': self.all_trajs,
                         'all_insts': self.all_insts,
                         'all_vid_info': self.all_vid_info,
                         'all_traj_cates': self.all_traj_cates,
                         'all_inst_count': self.all_inst_count}, f)
        print('%s created' % cache_path)