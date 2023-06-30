import argparse
from argparse import ArgumentParser

# Construct an argument parser
all_args = argparse.ArgumentParser()


def makeArguments(arguments: ArgumentParser) -> dict:
    all_args.add_argument("-b", "--Bayesian", action="store", dest="b",
                          type=int, choices=range(1, 8),
                          help="Bayesian model of size x")
    all_args.add_argument("-f", "--Frequentist", action="store", dest="f",
                          type=int, choices=range(1, 8),
                          help="Frequentist model of size x")
    all_args.add_argument("-E", "--EarlyStopping", action="store_true",
                          help="Early Stopping criteria")
    all_args.add_argument("-e", "--EnergyBound", action="store_true",
                          help="Energy Bound criteria")
    all_args.add_argument("-a", "--AccuracyBound", action="store_true",
                          help="Accuracy Bound criteria")
    all_args.add_argument("-s", "--Save", action="store_true",
                          help="Save model")
    all_args.add_argument('--net_type', default='lenet', type=str,
                          help='model = [lenet/AlexNet/3Conv3FC]')
    all_args.add_argument('--dataset', default='CIFAR10', type=str,
                          help='dataset = [MNIST/CIFAR10/CIFAR100]')
    return vars(all_args.parse_args())
