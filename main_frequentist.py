from __future__ import print_function
import os
import data
import torch
import pickle
import metrics
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.optim import Adam, lr_scheduler
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.AlexNet import AlexNet
from stopping_crit import earlyStopping, energyBound, accuracyBound
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC

with (open("configuration.pkl", "rb")) as file:
    while True:
        try:
            cfg = pickle.load(file)
        except EOFError:
            break

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs, wide=cfg["model"]["size"]):
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs, wide)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / \
                3Conv3FC')


def train_model(net, optimizer, criterion, train_loader):
    train_loss = 0.0
    net.train()
    accs = []
    for datas, target in train_loader:
        data, target = datas.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return train_loss, np.mean(accs)


def validate_model(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    accs = []
    for datas, target in valid_loader:
        data, target = datas.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return valid_loss, np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    n_epochs = cfg["model"]["n_epochs"]
    lr = cfg["model"]["lr"]
    num_workers = cfg["model"]["num_workers"]
    valid_size = cfg["model"]["valid_size"]
    batch_size = cfg["model"]["batch_size"]

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model\
            _{net_type}_{cfg["model"]["size"]}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    stp = cfg["stopping_crit"]
    sav = cfg["save"]

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6,
                                              verbose=True)
    # valid_loss_min = np.Inf
    # if stp == 2:
    early_stop = []
    train_data = []
    for epoch in range(1, n_epochs+1):

        train_loss, train_acc = train_model(net, optimizer, criterion,
                                            train_loader)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        lr_sched.step(valid_loss)

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        train_data.append([epoch, train_loss, train_acc, valid_loss,
                           valid_acc])
        print('Epoch: {} \tTraining Loss: {: .4f} \tTraining Accuracy: {: .4f}\
              \tValidation Loss: {: .4f} \tValidation Accuracy: {: .4f}\
              '.format(epoch, train_loss, train_acc, valid_loss, valid_acc))

        if stp == 2:
            # print('Using early stopping')
            if earlyStopping(early_stop, valid_acc, epoch,
                             cfg["model"]["sens"]) == 1:
                break
        elif stp == 3:
            # print('Using energy bound')
            if energyBound(cfg["model"]["energy_thrs"]) == 1:
                break
        elif stp == 4:
            # print('Using accuracy bound')
            if accuracyBound(train_acc,
                             cfg["model"]["acc_thrs"]) == 1:
                break
        else:
            print('Training for {} epochs'.format(cfg["model"]["n_epochs"]))

        if sav == 1:
            # save model when finished
            if epoch == n_epochs:
                torch.save(net.state_dict(), ckpt_name)

    with open("freq_exp_data_"+str(cfg["model"]["size"])+".pkl", 'wb') as f:
        pickle.dump(train_data, f)


if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Initial Time =", current_time)
    print("Using frequentist model of size: {}".format(cfg["model"]["size"]))
    run(cfg["data"], cfg["model"]["net_type"])
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Final Time =", current_time)
