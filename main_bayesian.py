from __future__ import print_function

import os
import data
import utils
import torch
import pickle
import metrics
import numpy as np
from datetime import datetime
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from stopping_crit import earlyStopping, energyBound, accuracyBound

with (open("configuration.pkl", "rb")) as file:
    while True:
        try:
            cfg = pickle.load(file)
        except EOFError:
            break


# CUDA settings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type,
                        wide=cfg["model"]["size"])
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type,
                           activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet\
                / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1,
                beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes,
                              num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type,
                                epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1,
                   epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes,
                              num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(validloader), beta_type,
                                epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg["model"]["layer_type"]
    activation_type = cfg["model"]["activation_type"]
    priors = cfg["model"]["priors"]

    train_ens = cfg["model"]["train_ens"]
    valid_ens = cfg["model"]["valid_ens"]
    n_epochs = cfg["model"]["n_epochs"]
    lr_start = cfg["model"]["lr"]
    num_workers = cfg["model"]["num_workers"]
    valid_size = cfg["model"]["valid_size"]
    batch_size = cfg["model"]["batch_size"]
    beta_type = cfg["model"]["beta_type"]

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type,
                   activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}\
            _{activation_type}_{cfg["model"]["size"]}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    stp = cfg["stopping_crit"]
    sav = cfg["save"]

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6,
                                              verbose=True)
    # valid_loss_max = np.Inf
    # if stp == 2:
    early_stop = []
    train_data = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer,
                                                      criterion,
                                                      train_loader,
                                                      num_ens=train_ens,
                                                      beta_type=beta_type,
                                                      epoch=epoch,
                                                      num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader,
                                               num_ens=valid_ens,
                                               beta_type=beta_type,
                                               epoch=epoch,
                                               num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        train_data.append([epoch, train_loss, train_acc, valid_loss,
                           valid_acc])
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy:\
                {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy:\
                {:.4f} \ttrain_kl_div: {:.4f}'.format(epoch, train_loss,
                                                      train_acc, valid_loss,
                                                      valid_acc, train_kl))

        if stp == 2:
            print('Using early stopping')
            if earlyStopping(early_stop, valid_acc, epoch,
                             cfg["model"]["sens"]) == 1:
                break
        elif stp == 3:
            print('Using energy bound')
            if energyBound(cfg["model"]["energy_thrs"]) == 1:
                break
        elif stp == 4:
            print('Using accuracy bound')
            if accuracyBound(train_acc, cfg.acc_thrs) == 1:
                break
        else:
            print('Training for {} epochs'.format(cfg["model"]["n_epochs"]))

        if sav == 1:
            # save model when finished
            if epoch == cfg.n_epochs-1:
                torch.save(net.state_dict(), ckpt_name)

    with open("bayes_exp_data_"+str(cfg["model"]["size"])+".pkl", 'wb') as f:
        pickle.dump(train_data, f)


if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Initial Time =", current_time)
    run(cfg["data"], cfg["model"]["net_type"])
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Final Time =", current_time)
