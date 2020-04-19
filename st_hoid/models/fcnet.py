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

        self.name = 'base+prior+pb'
        self.training = False

        self.gcn = GCN(2048, 2048, 2048, 0.5)

        self.obj_attention = nn.Sequential(
            nn.Linear(300, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 6))

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

        # fc
        spa_score = self.spa_branch(spa_feat)
        lan_score = self.lan_branch(lan_feat)
        body_score = self.body_branch(body_feat)

        spa_prob = sigmoid(spa_score)
        lan_prob = sigmoid(lan_score)
        body_prob = sigmoid(body_score)

        # CTX
        spa_prob = spa_prob * pre_mask
        lan_prob = lan_prob * pre_mask
        body_prob = body_prob * pre_mask

        branch_cnt = 3.0
        prob = (lan_prob + spa_prob + body_prob) / branch_cnt

        if self.training and pre_label is not None:
            # lf
            spa_loss = binary_cross_entropy(spa_prob, pre_label, size_average=False)
            lan_loss = binary_cross_entropy(lan_prob, pre_label, size_average=False)
            body_loss = binary_cross_entropy(body_prob, pre_label, size_average=False)
            loss = spa_loss + lan_loss + body_loss
        else:
            loss = Variable(torch.FloatTensor(-1))

        return prob, loss

    def name(self):
        return self.name