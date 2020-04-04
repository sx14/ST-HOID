import torch
from torch import nn
from torch.nn.functional import softmax, sigmoid
from torch.nn.functional import cross_entropy, binary_cross_entropy
from torch.autograd import Variable

from gcn import GCN


class FCNet(nn.Module):

    def load_weight(self, weight_path):
        # load weights
        weight_dict = torch.load(weight_path)
        resume_values = weight_dict.values()
        resume_dict = self.state_dict()
        for name, param in zip(list(resume_dict.keys()), list(resume_values)):
            resume_dict[name] = param
        self.load_state_dict(resume_dict)

    def train(self, mode=True):
        super(FCNet, self).train(mode)
        self.training = mode

    def eval(self):
        super(FCNet, self).eval()
        self.training = False

    def __init__(self, cate_num,
                 sbj_feat_len=2048,
                 obj_feat_len=2048,
                 sce_feat_lan=2048,
                 body_feat_lan=2048 * 6,
                 lan_feat_len=600,
                 spa_feat_lan=14 * 3):

        super(FCNet, self).__init__()

        self.name = 'fc_gcn_net'
        self.training = False

        self.gcn = GCN(2048, 2048, 2048, 0.5)

        self.lan_branch = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(lan_feat_len, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, cate_num))

        self.spa_branch = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(spa_feat_lan, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, cate_num))

        self.sbj_branch = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(sbj_feat_len, sbj_feat_len),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(sbj_feat_len, cate_num))

        self.sce_branch = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(sce_feat_lan, sce_feat_lan),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(sce_feat_lan, cate_num))

        self.obj_branch = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(obj_feat_len, obj_feat_len),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(obj_feat_len, cate_num))

        self.body_branch = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(body_feat_lan, body_feat_lan),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(body_feat_lan, cate_num))

    def forward(self, adj_mat, sbj_feat, obj_feat, body_feat, lan_feat, spa_feat, sce_feat, pre_mask, pre_label=None):
        sbj_feat = sbj_feat.unsqueeze(1)
        obj_feat = obj_feat.unsqueeze(1)
        sce_feat = sce_feat.unsqueeze(1)
        body_feat = body_feat.view(body_feat.shape[0], 6, -1)

        all_vis_feat = torch.cat((sbj_feat, obj_feat, sce_feat, body_feat), 1)
        all_vis_feat = self.gcn(all_vis_feat, adj_mat)

        sbj_feat = sbj_feat.squeeze(1) + all_vis_feat[:, 0].squeeze(1)
        obj_feat = obj_feat.squeeze(1) + all_vis_feat[:, 1].squeeze(1)
        sce_feat = sce_feat.squeeze(1) + all_vis_feat[:, 2].squeeze(1)
        body_feat = body_feat.view(body_feat.shape[0], -1) + all_vis_feat[:, 3:].view(all_vis_feat.shape[0], -1)

        sbj_score = self.sbj_branch(sbj_feat)
        obj_score = self.obj_branch(obj_feat)
        spa_score = self.spa_branch(spa_feat)
        lan_score = self.lan_branch(lan_feat)
        sce_score = self.sce_branch(sce_feat)

        sbj_prob = sigmoid(sbj_score)
        obj_prob = sigmoid(obj_score)
        spa_prob = sigmoid(spa_score)
        lan_prob = sigmoid(lan_score)
        sce_prob = sigmoid(sce_score)

        sbj_prob = sbj_prob * pre_mask
        obj_prob = obj_prob * pre_mask
        spa_prob = spa_prob * pre_mask
        lan_prob = lan_prob * pre_mask
        sce_prob = sce_prob * pre_mask

        branch_cnt = 5.0
        prob = sbj_prob + obj_prob + lan_prob + spa_prob + sce_prob

        if body_feat.sum() != 0:
            body_score = self.body_branch(body_feat)
            body_prob = sigmoid(body_score)
            body_prob = body_prob * pre_mask
            branch_cnt += 1
            prob += body_prob

        if self.training and pre_label is not None:
            sbj_loss = binary_cross_entropy(sbj_prob, pre_label, size_average=False)
            obj_loss = binary_cross_entropy(obj_prob, pre_label, size_average=False)
            spa_loss = binary_cross_entropy(spa_prob, pre_label, size_average=False)
            lan_loss = binary_cross_entropy(lan_prob, pre_label, size_average=False)
            sce_loss = binary_cross_entropy(sce_prob, pre_label, size_average=False)

            loss = sbj_loss + obj_loss + spa_loss + lan_loss + sce_loss
            if body_feat.sum() != 0:
                body_loss = binary_cross_entropy(body_prob, pre_label, size_average=False)
                loss += body_loss
        else:
            loss = Variable(torch.FloatTensor(-1))

        prob = prob / branch_cnt
        return prob, loss

    def name(self):
        return self.name