import os
import time
import json
import pickle
from random import shuffle
from copy import deepcopy
from math import log, e

import numpy as np
from torch.utils.data import Dataset


class VidOR(Dataset):
    def __init__(self, ds_name, ds_root, split, cache_root):

        self.val_ratio = 0.01
        self.dataset_name = ds_name
        self.dataset_root = ds_root
        self.cache_root = cache_root
        self.split = split

        if split == 'train':
            self.feat_root = os.path.join(ds_root, 'feat_gt', 'train')
            self.data_root = os.path.join(ds_root, 'Data', 'VID', 'train')
            self.anno_root = os.path.join(ds_root, 'anno_with_pose', 'training')
        elif split == 'val':
            self.feat_root = os.path.join(ds_root, 'feat_gt', 'val')
            self.data_root = os.path.join(ds_root, 'Data', 'VID', 'val')
            self.anno_root = os.path.join(ds_root, 'anno_with_pose', 'validation')
        else:
            print('split = "train" or "val"')
            exit(-1)

        self.obj_cates = None
        self.pre_cates = None
        self.obj_cate2idx = None
        self.pre_cate2idx = None
        self.sbj_cates = {'baby', 'adult', 'child'}
        self._load_category_sets()

        self.SEG_LEN = 10
        self.all_trajs = None
        self.all_insts = None
        self.all_vid_info = None
        self.all_traj_cates = None
        self.all_inst_count = None
        self.obj2pre_mask = None
        self.sbj2pre_mask = None
        self._load_annotations()
        self.obj_vecs = self._load_object_vectors(ds_root)

        self.curr_pkg_id = -1
        self.curr_vid_id = -1
        self.traj_feats = None

    def __getitem__(self, item):
        inst = self.all_insts[item]
        lan_feat = self._ext_language_feature(inst)
        spa_feat = self._ext_spatial_feature(inst)
        sbj_feat, obj_feat, sce_feat, body_feat = self._ext_cnn_feature(inst)

        adj_mat = np.ones((9, 9))
        rowsum = adj_mat.sum(1)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        adj_mat = r_mat_inv.dot(adj_mat)

        vid_id = inst['vid_id']
        sbj_tid = inst['sbj_tid']
        obj_tid = inst['obj_tid']
        sbj_cate = self.all_traj_cates[vid_id][sbj_tid]
        obj_cate = self.all_traj_cates[vid_id][obj_tid]
        sbj_pre_mask = self.sbj2pre_mask[sbj_cate]
        obj_pre_mask = self.obj2pre_mask[obj_cate]
        pre_mask = sbj_pre_mask * obj_pre_mask
        pre_cate = np.zeros(len(self.pre_cates))
        pre_cate[inst['pre_cate']] = 1
        return sbj_feat, obj_feat, body_feat, lan_feat, spa_feat, sce_feat, adj_mat, pre_mask, pre_cate

    def __len__(self):
        return len(self.all_insts)

    def is_subject(self, cate):
        if isinstance(cate, int):
            return self.obj_cates[cate] in self.sbj_cates
        else:
            return cate in self.sbj_cates

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
        obj_vec = self.obj_vecs[obj_cate_idx]
        sbj_vec = self.obj_vecs[sbj_cate_idx]
        return np.concatenate((obj_vec, sbj_vec))

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
        sbj_end_box = self.all_trajs[vid_id][sbj_tid][end_fid-1]

        obj_stt_box = self.all_trajs[vid_id][obj_tid][stt_fid]
        obj_end_box = self.all_trajs[vid_id][obj_tid][end_fid-1]

        width = self.all_vid_info[vid_id]['width']
        height = self.all_vid_info[vid_id]['height']

        spatial_feat_stt = cal_spatial_feat(sbj_stt_box, obj_stt_box, width, height)
        spatial_feat_end = cal_spatial_feat(sbj_end_box, obj_end_box, width, height)
        spatial_feat_dif = spatial_feat_end - spatial_feat_stt
        return np.concatenate((spatial_feat_stt, spatial_feat_end, spatial_feat_dif))

    def _ext_cnn_feature(self, inst):
        pkg_id = inst['pkg_id']
        vid_id = inst['vid_id']

        if pkg_id != self.curr_pkg_id or vid_id != self.curr_vid_id:
            feat_dir = os.path.join(self.feat_root, pkg_id, vid_id)
            tid2traj_feat = {}
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
        sce_tid = inst['sce_tid']
        stt_fid = inst['stt_fid']
        seg_idx = int(stt_fid / self.SEG_LEN)
        obj_feat = self.traj_feats[str(obj_tid)][seg_idx][0]
        sbj_feat = self.traj_feats[str(sbj_tid)][seg_idx][0]
        sce_feat = self.traj_feats[str(sce_tid)][seg_idx][0]
        body_feat = self.traj_feats[str(sbj_tid)][seg_idx][1:].reshape(-1)
        body_part_num = self.traj_feats[str(sbj_tid)][seg_idx].shape[0] - 1
        if body_feat.sum() == 0:
            body_feat = np.tile(sbj_feat, body_part_num)
        return sbj_feat, obj_feat, sce_feat, body_feat

    def _load_object_vectors(self, ds_root):
        # load object word2vec
        o2v_path = os.path.join(ds_root, 'object_vectors.mat')
        w2v_path = os.path.join(ds_root, 'GoogleNews-vectors-negative300.bin')
        if not os.path.exists(o2v_path):
            self._prepare_w2v(w2v_path, o2v_path)
        with open(o2v_path, 'r') as f:
            obj_vecs = pickle.load(f)
        return obj_vecs

    def _prepare_w2v(self, w2v_path, o2v_path):
        import gensim
        # load pre-trained word2vec model
        print('Loading pretrained word vectors ...')
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        obj_vecs = self._extract_object_vectors(w2v_model)

        # save object label vectors
        with open(o2v_path, 'w') as f:
            pickle.dump(obj_vecs, f)
        print('VidOR object word vectors saved at: %s' % o2v_path)

    def _extract_object_vectors(self, w2v_model, vec_len=300, debug=False):
        # object labels to vectors
        print('Extracting word vectors for VidOR object labels ...')
        obj_vecs = np.zeros((len(self.obj_cates), vec_len))
        for i in range(len(self.obj_cates)):
            obj_label = self.obj_cates[i]
            obj_label = obj_label.split('/')[0]

            if obj_label == 'traffic_light':
                obj_label = 'signal'
            if obj_label == 'stop_sign':
                obj_label = 'sign'
            if obj_label == 'baby_seat':
                obj_label = 'stroller'
            if obj_label == 'electric_fan':
                obj_label = 'fan'
            if obj_label == 'baby_walker':
                obj_label = 'walker'

            vec = w2v_model[obj_label]
            if debug and vec is None or len(vec) == 0 or np.sum(vec) == 0:
                print('[WARNING] %s' % obj_label)
            obj_vecs[i] = vec
        return obj_vecs

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
                    tid2dur[tid] = [frm_idx, frm_idx+1]

                tid2traj[tid][frm_idx] = box
                tid2dur[tid][1] = frm_idx + 1

        for tid in tid2traj:
            tid2traj[tid] = np.array(tid2traj[tid]).astype(np.int)
        return tid2traj, tid2dur

    def _gen_pre_mask(self):
        sbj2pre_mask = np.zeros((self.category_num('object'), self.category_num('predicate')))
        obj2pre_mask = np.zeros((self.category_num('object'), self.category_num('predicate')))

        for inst in self.all_insts:
            vid = inst['vid_id']
            sbj_tid = inst['sbj_tid']
            obj_tid = inst['obj_tid']
            sbj_cate = self.all_traj_cates[vid][sbj_tid]
            obj_cate = self.all_traj_cates[vid][obj_tid]
            pre_cate = inst['pre_cate']
            obj2pre_mask[obj_cate, pre_cate] = 1
            sbj2pre_mask[sbj_cate, pre_cate] = 1

        self.obj2pre_mask = obj2pre_mask
        self.sbj2pre_mask = sbj2pre_mask

    def _gen_positive_instances(self, org_insts, pkg_id, vid_id):
        insts = []
        seg_len = self.SEG_LEN
        for org_inst in org_insts:
            sbj_tid = org_inst['subject_tid']
            obj_tid = org_inst['object_tid']
            stt_frm_idx = org_inst['begin_fid']
            end_frm_idx = org_inst['end_fid']
            pre_cate = org_inst['predicate']

            for frm_idx in range(int(stt_frm_idx / seg_len) * seg_len,
                                 int(end_frm_idx / seg_len) * seg_len, seg_len):
                seg_stt_frm_idx = frm_idx
                seg_end_frm_idx = frm_idx + seg_len
                insts.append({
                    'pkg_id': pkg_id,
                    'vid_id': vid_id,
                    'stt_fid': seg_stt_frm_idx,
                    'end_fid': seg_end_frm_idx,
                    'sbj_tid': sbj_tid,
                    'obj_tid': obj_tid,
                    'sce_tid': -1,
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
            stt_frm_idx = pos_inst['begin_fid']
            end_frm_idx = pos_inst['end_fid']
            cands['%d-%d' % (sbj_tid, obj_tid)][stt_frm_idx: end_frm_idx] = 0

        # generate negative instances
        neg_insts = []
        for sid_oid, cand_dur in cands.items():
            sid, oid = sid_oid.split('-')
            sbj_tid = int(sid)
            obj_tid = int(oid)
            for frm_idx in range(0, int(cand_dur.shape[0] / self.SEG_LEN) * self.SEG_LEN, self.SEG_LEN):
                if cand_dur[frm_idx: frm_idx+self.SEG_LEN].sum() == self.SEG_LEN:
                    neg_insts.append({
                        'pkg_id': pkg_id,
                        'vid_id': vid_id,
                        'stt_fid': frm_idx,
                        'end_fid': frm_idx + self.SEG_LEN,
                        'sbj_tid': sbj_tid,
                        'obj_tid': obj_tid,
                        'sce_tid': -1,
                        'pre_cate': self.pre_cate2idx['__no_interaction__']})
        return neg_insts

    def _load_annotations(self):

        cache_path = os.path.join(self.cache_root, '%s_%s_anno_cache.bin' % (self.dataset_name, self.split))
        if os.path.exists(cache_path):
            # print('%s found! loading ...' % cache_path)
            with open(cache_path) as f:
                data_cache = pickle.load(f)

            self.all_trajs = data_cache['all_trajs']
            self.all_insts = data_cache['all_insts']
            self.all_vid_info = data_cache['all_vid_info']
            self.all_traj_cates = data_cache['all_traj_cates']
            self.all_inst_count = data_cache['all_inst_count']
            self.sbj2pre_mask = data_cache['sbj2pre_mask']
            self.obj2pre_mask = data_cache['obj2pre_mask']
            return

        print('Processing annotations ...')
        time.sleep(2)
        self.all_vid_info = {}
        self.all_trajs = {}
        self.all_traj_cates = {}
        self.all_insts = []
        self.all_inst_count = []

        from tqdm import tqdm
        for pkg_id in tqdm(sorted(os.listdir(self.anno_root))):
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
                tid2cate_idx = {traj_info['tid']: self.obj_cate2idx[traj_info['category']]
                                for traj_info in vid_anno['subject/objects']}
                pos_insts = self._gen_positive_instances(vid_anno['relation_instances'], pkg_id, vid_id)
                neg_insts = self._gen_negative_instances(vid_anno['relation_instances'], tid2dur, tid2cate_idx, pkg_id, vid_id)
                # vid_insts = pos_insts + neg_insts[: len(pos_insts)]
                vid_insts = pos_insts
                shuffle(vid_insts)
                self.all_vid_info[vid_id] = vid_info
                self.all_trajs[vid_id] = tid2traj
                self.all_traj_cates[vid_id] = tid2cate_idx
                self.all_insts += vid_insts
                self.all_inst_count.append(len(vid_insts))

        self._gen_pre_mask()

        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)

        with open(cache_path, 'w') as f:
            pickle.dump({'all_trajs': self.all_trajs,
                         'all_insts': self.all_insts,
                         'all_vid_info': self.all_vid_info,
                         'all_traj_cates': self.all_traj_cates,
                         'all_inst_count': self.all_inst_count,
                         'sbj2pre_mask': self.sbj2pre_mask,
                         'obj2pre_mask': self.obj2pre_mask}, f)
        print('%s created' % cache_path)
