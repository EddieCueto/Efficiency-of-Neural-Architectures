from __future__ import print_function
import os
import data
import torch
#import utils
import pickle
import metrics
import argparse
import numpy as np
import torch.nn as nn
import amd_sample_draw
from datetime import datetime
import config_frequentist as cfg
from torch.optim import Adam, lr_scheduler
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.AlexNet import AlexNet
from stopping_crit import earlyStopping, energyBound, accuracyBound
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC

# CUDA settings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs,wide=cfg.wide):
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs,wide)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, train_loader):
    train_loss = 0.0
    net.train()
    accs = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
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
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return valid_loss, np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}_{cfg.wide}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    with open("stp", "r") as file:
        stp = int(file.read())
    with open("sav", "r") as file:
        sav = int(file.read())

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    #valid_loss_min = np.Inf
    if stp == 2:
        early_stop = []
    train_data = []
    for epoch in range(1, n_epochs+1):

        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        lr_sched.step(valid_loss)

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        train_data.append([epoch,train_loss,train_acc,valid_loss,valid_acc])
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
        
        if stp == 2:
            print('Using early stopping')
            earlyStopping(early_stop,train_acc,cfg.sens)
        elif stp == 3: 
            print('Using energy bound')
            energyBound(cfg.energy_thrs)
        elif stp == 4:
            print('Using accuracy bound')
            accuracyBound(cfg.acc_thrs)
        else:
            print('Training for {} epochs'.format(cfg.n_epochs))

        if sav == 1:
            # save model when finished
            if epoch == n_epochs:
                torch.save(net.state_dict(), ckpt_name)
    
    with open("freq_exp_data_"+str(cfg.wide)+".pkl", 'wb') as f:
      pickle.dump(train_data, f)
    

if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Initial Time =", current_time)
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()
    run(args.dataset, args.net_type)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Final Time =", current_time)

