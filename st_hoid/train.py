import os

import yaml
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.autograd import Variable

from datasets.vidor import VidOR
from models.fcnet import FCNet


class Container:

    def __init__(self, cfg, model, dataset):
        # hyper-params
        self.num_epoch = cfg['train_epoch']
        self.batch_size = cfg['train_batch_size']
        self.lr_init = cfg['train_lr']
        self.lr_adjust_rate = cfg['train_lr_adjust_rate']
        self.lr_adjust_freq = cfg['train_lr_adjust_freq']
        self.train_momentum = cfg['train_momentum']
        self.train_weight_decay = cfg['train_weight_decay']
        self.save_freq = cfg['train_save_freq']
        self.print_freq = cfg['train_print_freq']
        self.eval = cfg['eval']
        self.weight_root = os.path.join(cfg['weight_root'], cfg['dataset'], cfg['exp'])
        self.weight_path = os.path.join(self.weight_root, model.name+'_%d.pkl')

        # init dataset
        self.dataset = dataset
        if self.eval:
            self.dataset_val = dataset.split_self()
        else:
            self.dataset_val = None

        # init model
        self.model = model
        self.loss_func = cross_entropy
        self.optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                         lr=self.lr_init,
                                         momentum=self.train_momentum,
                                         weight_decay=self.train_weight_decay,
                                         nesterov=True)
        # resume params
        self.resume = cfg['train_resume']
        self.resume_epoch = cfg['train_resume_epoch']

        # gpu switch
        self.use_gpu = cfg['use_gpu']

    @staticmethod
    def cal_acc(probs, labels):
        pre_labels = np.argmax(probs, axis=1)
        diff = labels - pre_labels
        diff[diff != 0] = 1
        fcnt = diff.sum()
        tcnt = diff.shape[0] - fcnt
        return tcnt * 1.0 / diff.shape[0]

    def save_weights(self, curr_finished_epoch):
        if not os.path.exists(self.weight_root):
            os.makedirs(self.weight_root)
        if (curr_finished_epoch + 1) % self.save_freq == 0:
            torch.save(self.model.state_dict(), self.weight_path % curr_finished_epoch)

    def adjust_lr(self, curr_finished_epoch):
        # adjust learning rate AFTER each epoch
        lr_curr = self.lr_init * (self.lr_adjust_rate ** int((curr_finished_epoch + 1) / self.lr_adjust_freq))
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}],
                                         lr=lr_curr,
                                         momentum=self.train_momentum,
                                         weight_decay=self.train_weight_decay,
                                         nesterov=True)

    def resume(self):
        # load weights
        weight_dict = torch.load(self.weight_path % self.resume_epoch)
        resume_values = weight_dict.values()
        resume_dict = self.model.state_dict()
        for name, param in zip(list(resume_dict.keys()), list(resume_values)):
            resume_dict[name] = param
        self.model.load_state_dict(resume_dict)
        curr_epoch = self.resume_epoch

        # adjust lr
        self.adjust_lr(self.resume_epoch)
        return curr_epoch

    def evaluation(self):

        sbj_feat_v = Variable(torch.FloatTensor(1))
        obj_feat_v = Variable(torch.FloatTensor(1))
        lan_feat_v = Variable(torch.FloatTensor(1))
        spa_feat_v = Variable(torch.FloatTensor(1))
        body_feat_v = Variable(torch.FloatTensor(1))
        pre_label_v = Variable(torch.LongTensor(1))

        if self.use_gpu:
            self.model.cuda()
            sbj_feat_v = sbj_feat_v.cuda()
            obj_feat_v = obj_feat_v.cuda()
            lan_feat_v = lan_feat_v.cuda()
            spa_feat_v = spa_feat_v.cuda()
            body_feat_v = body_feat_v.cuda()
            pre_label_v = pre_label_v.cuda()

        acc_sum = 0.0
        self.model.eval()
        print('evaluating ...')
        data_loader_val = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.batch_size)
        for itr, data in enumerate(data_loader_val):

            sbj_feat, obj_feat, body_feat, \
            lan_feat, spa_feat, pre_label = data
            sbj_feat_v.data.resize_(sbj_feat.size()).copy_(sbj_feat)
            obj_feat_v.data.resize_(obj_feat.size()).copy_(obj_feat)
            lan_feat_v.data.resize_(lan_feat.size()).copy_(lan_feat)
            spa_feat_v.data.resize_(spa_feat.size()).copy_(spa_feat)
            body_feat_v.data.resize_(body_feat.size()).copy_(body_feat)
            pre_label_v.data.resize_(pre_label.size()).copy_(pre_label)

            probs, scores = self.model.forward(sbj_feat_v, obj_feat_v, body_feat_v, lan_feat_v, spa_feat_v)

            if self.use_gpu:
                probs = probs.cpu()
            acc = self.cal_acc(probs.data.numpy(), pre_label.numpy())
            acc_sum += acc

            acc = acc_sum / len(data_loader_val)
            print('eval acc: %.4f' % acc)

    def train(self):
        curr_epoch = 0

        if self.resume:
            curr_epoch = self.resume()

        sbj_feat_v = Variable(torch.FloatTensor(1))
        obj_feat_v = Variable(torch.FloatTensor(1))
        lan_feat_v = Variable(torch.FloatTensor(1))
        spa_feat_v = Variable(torch.FloatTensor(1))
        body_feat_v = Variable(torch.FloatTensor(1))
        pre_label_v = Variable(torch.LongTensor(1))

        if self.use_gpu:
            self.model.cuda()
            sbj_feat_v = sbj_feat_v.cuda()
            obj_feat_v = obj_feat_v.cuda()
            lan_feat_v = lan_feat_v.cuda()
            spa_feat_v = spa_feat_v.cuda()
            body_feat_v = body_feat_v.cuda()
            pre_label_v = pre_label_v.cuda()

        data_loader_train = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

        while curr_epoch < self.num_epoch:
            self.model.train()
            itr_num = len(data_loader_train)
            for itr, data in enumerate(data_loader_train):
                self.optimizer.zero_grad()

                sbj_feat, obj_feat, body_feat, \
                lan_feat, spa_feat, pre_label = data
                sbj_feat_v.data.resize_(sbj_feat.size()).copy_(sbj_feat)
                obj_feat_v.data.resize_(obj_feat.size()).copy_(obj_feat)
                lan_feat_v.data.resize_(lan_feat.size()).copy_(lan_feat)
                spa_feat_v.data.resize_(spa_feat.size()).copy_(spa_feat)
                body_feat_v.data.resize_(body_feat.size()).copy_(body_feat)
                pre_label_v.data.resize_(pre_label.size()).copy_(pre_label)

                probs, scores = self.model.forward(sbj_feat_v, obj_feat_v, body_feat_v, lan_feat_v, spa_feat_v)
                loss = self.loss_func(scores, pre_label_v)
                loss.backward()
                self.optimizer.step()

                if self.use_gpu:
                    probs = probs.cpu()
                    loss = loss.cpu()
                loss = loss.data.item()
                acc = self.cal_acc(probs.data.numpy(), pre_label.numpy())
                if itr % self.print_freq == 0:
                    print('[epoch %d][%d/%d] loss: %.4f acc: %.4f' % (curr_epoch, itr, itr_num, loss, acc))

            if self.eval:
                self.evaluation()
            self.save_weights(curr_epoch)
            self.adjust_lr(curr_epoch)
            curr_epoch += 1


if __name__ == '__main__':
    exp = 'fc'
    dataset_name = 'vidor_hoid_mini'
    dataset_root = '../data/%s' % dataset_name
    cfg_path = 'cfgs/%s_%s.yaml' % (dataset_name, exp)
    with open(cfg_path) as f:
        cfg = yaml.load(f)
    dataset = VidOR(dataset_name, dataset_root, 'train', '../cache')
    model = FCNet(dataset.category_num('predicate'))
    container = Container(cfg, model, dataset)
    container.train()



