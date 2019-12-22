import random

import h5py

from tqdm import *

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim

import config

torch.manual_seed(0)
random.seed(0)


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


class EEGClassifier(nn.Module):
    """
    A subject-independent classifier:
    There are 64 EEG channels (input_size), 9 subjects ad 12 stimuli.
    Another observation's that EEG signals are long (seq_len = 3518), more than 250-500 time steps used in practice for LSTM.
    One way to address this is to employ LSTM AE.
    Noisy artifacts of EEG -> disentangled representation's used within an LSTM.
    """
    def __init__(self, input_channels=64, target_size=12, n_subjects=9,
                 code_dim=64, hidden_dim=256, seq_len=3518):
        super(EEGClassifier, self).__init__()
        self.seq_len  = seq_len
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        encoding_fc = []
        # 1x1 convolution with default groups=1, is essentially a full matrix applied across all channels f.
        encoding_fc.append(nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=code_dim,
                                                   kernel_size=160, stride=80),
                                         nn.LeakyReLU(0.2)))
        self.encoding_fc = nn.Sequential(*encoding_fc)
        # The last hidden state of LSTM over the eeg signal is used for classification.
        self.classifier_lstm = nn.LSTM(code_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.target = nn.Sequential(nn.Linear(hidden_dim, target_size), nn.Tanh(), nn.Dropout(0.75))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 1)
        #     elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.encoding_fc(x)
        x = x.view(-1, 42 , self.code_dim) # kernel 80 stride 50 ->86
        _, (hidden, _) = self.classifier_lstm(x)
        hidden = hidden.view(-1, self.hidden_dim)
        return self.target(hidden)


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
            pred_target = model(features)
            _, pred_target = torch.max(pred_target.data, 1)
            total += target.size(0)
            correct_target+=(pred_target==target).sum().item()
    model.train()
    return correct_target/total


def train_classifier(model, optim, dataset, epochs, path, test, start = 0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start, epochs):
        running_loss = 0.0
        for i, item  in tqdm(enumerate(dataset,1)):
            features, target, subject = item
            target = torch.argmax(target, dim=1) # one hot back to int
            pred_target = model(features)
            loss = criterion(pred_target, target)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        training_accuracy = check_accuracy(model, dataset)
        test_accuracy = check_accuracy(model, test)
        print("Epoch {} Training Accuracy: {}, Test Accuracy:{}, Avg Loss {}".
              format(epoch+1, training_accuracy, test_accuracy, running_loss/i))
            # save_model(model, optim, epoch, path)


if __name__=='__main__':
    model = EEGClassifier()
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_data = Miir(config.DATA_PATH, True)
    test_data = Miir(config.DATA_PATH, False)
    loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    loader_test = data.DataLoader(test_data, batch_size=60, shuffle=True, num_workers=4)
    train_classifier(model=model, optim=optim, dataset=loader, epochs=200, path= './checkpoint_classifier.pth', test= loader_test)

