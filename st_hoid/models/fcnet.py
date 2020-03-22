import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torch.autograd import Variable


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
                 body_feat_lan=2048 * 6,
                 lan_feat_len=600,
                 spa_feat_lan=14 * 3):

        super(FCNet, self).__init__()

        self.name = 'fcnet'
        self.training = False

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

    def forward(self, sbj_feat, obj_feat, body_feat, lan_feat, spa_feat, pre_label=None):
        sbj_score = self.sbj_branch(sbj_feat)
        obj_socre = self.obj_branch(obj_feat)
        spa_score = self.spa_branch(spa_feat)
        lan_score = self.lan_branch(lan_feat)

        if body_feat.sum() != 0:
            body_score = self.body_branch.forward(body_feat)
            score = sbj_score + obj_socre + lan_score + spa_score + body_score
        else:
            score = sbj_score + obj_socre + lan_score + spa_score

        if self.training and pre_label is not None:
            loss = cross_entropy(score, pre_label, size_average=False)
        else:
            loss = Variable(torch.FloatTensor(-1))
        return softmax(score, dim=1), score, loss

    def name(self):
        return self.name