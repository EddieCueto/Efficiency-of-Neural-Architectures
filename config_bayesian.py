############### Configuration file for Bayesian ###############

import os
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 100
sens = 1e-9
energy_thrs = 100000
acc_thrs = 0.99
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value


with open("bay", "r") as file:
    bay = int(file.read())

if bay == 1:
    with open("tmp", "r") as file:
        wide = int(file.read())

    #if os.path.exists("tmp"):
    #    os.remove("tmp")
    #else:
    #    raise Exception("Tmp file not found")

    print("Bayesian configured to run with width: {}".format(wide))


#if os.path.exists("bay"): 
#    os.remove("bay")
#else:
#    raise Exception("Bay file not found")
    
