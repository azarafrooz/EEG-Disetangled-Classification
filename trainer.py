import os
import random

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import *
import h5py

from model import *
import config

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class Miir(data.Dataset):
    def __init__(self, data_path=config.DATA_PATH, train=True):
        h5 = h5py.File(data_path, 'r')
        self.train = train

        features = h5['features']
        targets = h5['targets']
        subjects = h5['subjects']

        self.test_subject_id = random.randint(0,8)
        train_indxs = [i for i, e in enumerate(subjects) if e != self.test_subject_id]

        self.train_features = [e for i, e in enumerate(features) if i in train_indxs]
        self.test_features = [e for i, e in enumerate(features) if i not in train_indxs]

        self.train_targets = [e for i, e in enumerate(targets) if i in train_indxs]
        self.test_targets = [e for i, e in enumerate(targets) if i not in train_indxs]

        self.train_subjects = [e for i, e in enumerate(subjects) if i in train_indxs]
        self.test_subjects = [e for i, e in enumerate(subjects) if i not in train_indxs]

        self.train_size = len(self.train_features)
        self.test_size = len(self.test_features)

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size

    def __getitem__(self, idx):
        if self.train:
            return self.train_features[idx], self.train_targets[idx], self.train_subjects[idx]
        else:
            return self.test_features[idx], self.test_targets[idx], self.test_subjects[idx]


def loss_fn(target, pred_target, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    """
    :param target:
    :param pred_target
    :param f_mean:
    :param f_logvar:
    :param z_post_mean:
    :param z_post_logvar:
    :param z_prior_mean:
    :param z_prior_logvar:
    :return:
    Loss function consists of 3 parts, Cross Entropy of the predicted targes and the target, the KL divergence of f,
    and the sum over the KL divergence of each z_t, with the sum divided by batch_size.
    Loss = {CrossEntropy(pred_target, target) + KL of f + sum(KL of z_t)}/batch_size
    Prior of f is a spherical zero_mean unit variance Gaussian and the prior for each z_t is a Gaussian whose
    mean and variance are given by LSTM.
    """
    batch_size = target.size(0)
    cross_entropy = F.cross_entropy(pred_target, target)
    kld_f = - 0.5 * torch.sum(1+f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.mean(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2))
                                                               / z_prior_var) - 1)

    return (cross_entropy + (kld_f + kld_z)) / batch_size, kld_f / batch_size, kld_z / batch_size,\
           cross_entropy/batch_size


def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'opimizer': optim.state_dict()}, path)


def check_accuracy(model, test):
    model.eval()
    total = 0
    correct_target = 0
    with torch.no_grad():
        for item in test:
            features, target, subject = item
            target = torch.argmax(target, dim=1) # one-hot back to int
            *_, pred_target = model(features)
            _, pred_target = torch.max(pred_target.data, 1)
            total += target.size(0)
            correct_target+=(pred_target==target).sum().item()
    model.train()
    return correct_target/total


def train_classifier(model, optim, dataset, epochs, path, test, start = 0):
    model.train()
    for epoch in range(start, epochs):
        losses = []
        kld_fs = []
        kld_zs = []
        cross_entropies = []
        print("Running Epoch: {}".format(epoch+1))
        for i, item in tqdm(enumerate(dataset,1)):
            features, target, subject = item
            target = torch.argmax(target, dim=1)  # one hot back to int
            optim.zero_grad()
            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean,\
            z_prior_logvar, pred_target = model(features)
            loss, kld_f, kld_z, cross_entropy = loss_fn(target, pred_target, f_mean, f_logvar,
                                                       z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            kld_fs.append(kld_f.item())
            kld_zs.append(kld_z.item())
            cross_entropies.append(cross_entropy.item())

        # training_accuracy = check_accuracy(model, dataset)
        test_accuracy = check_accuracy(model, test)
        meanloss = np.mean(losses)
        meanf = np.mean(kld_fs)
        meanz = np.mean(kld_zs)
        mean_cross_entropies = np.mean(cross_entropies)
        print("Epoch {} : Average Loss: {} KL of f : {} KL of z : {} "
              "Cross Entropy: {} Test Accuracy: {}".format(epoch + 1, meanloss, meanf, meanz, mean_cross_entropies,
                                                           test_accuracy))
        save_model(model, optim, epoch, path)


if __name__=='__main__':
    model = DisentangledEEG(factorized=True, nonlinearity=True)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_data = Miir(config.DATA_PATH, True)
    test_data = Miir(config.DATA_PATH, False)
    loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    loader_test = data.DataLoader(test_data, batch_size=60, shuffle=True, num_workers=4)
    train_classifier(model=model, optim=optim, dataset=loader, epochs=200,
                     path='./checkpoint_disentangled_classifier.pth', test=loader_test)