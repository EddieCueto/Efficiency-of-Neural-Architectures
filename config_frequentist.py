############### Configuration file for Frequentist ###############

import os
n_epochs = 100
lr = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256

with open("frq", "r") as file:
    frq = int(file.read())

if frq == 1:
    with open("tmp", "r") as file:
        wide = int(file.read())

    if os.path.exists("tmp"):
        os.remove("tmp")
    else:
        raise Exception("Tmp file not found")

    print("Frequentist configured to run with width: {}".format(wide))



if os.path.exists("frq"):
    os.remove("frq")
else:
    raise Exception("Frq file not found")

