import psutil
import pickle
import arguments
from time import sleep
from pathlib import Path
import subprocess as sub
from arguments import makeArguments


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


cfg = {
        "model": {"net_type": None, "type": None, "size": None, "layer_type":
                  "lrt", "activation_type": "softplus", "priors": {
                      'prior_mu': 0,
                      'prior_sigma': 0.1,
                      'posterior_mu_initial': (0, 0.1),  # (mean,std) normal_
                      'posterior_rho_initial': (-5, 0.1),  # (mean,std) normal_
                      },
                  "n_epochs": 100,
                  "sens": 1e-9,
                  "energy_thrs": 10000,
                  "acc_thrs": 0.99,
                  "lr": 0.001,
                  "num_workers": 4,
                  "valid_size": 0.2,
                  "batch_size": 256,
                  "train_ens": 1,
                  "valid_ens": 1,
                  "beta_type": 0.1,  # 'Blundell','Standard',etc.
                  # Use float for const value
                  },
        "data": None,
        "stopping_crit": None,
        "save": None,
        "pickle_path": None,
}

args = makeArguments(arguments.all_args)

check = list(args.values())
if all(v is None for v in check):
    raise Exception("One argument required")
elif None in check:
    if args['f'] is not None:
        cmd = ["python", "main_frequentist.py"]
        cfg["model"]["type"] = "freq"
    elif args['b'] is not None:
        cmd = ["python", "main_bayesian.py"]
        cfg["model"]["type"] = "bayes"
else:
    raise Exception("Only one argument allowed")


wide = args["f"] or args["b"]

cfg["model"]["size"] = wide
cfg["data"] = args["dataset"]
cfg["model"]["net_type"] = args["net_type"]


if args['EarlyStopping']:
    cfg["stopping_crit"] = 2
elif args['EnergyBound']:
    cfg["stopping_crit"] = 3
elif args['AccuracyBound']:
    cfg["stopping_crit"] = 4
else:
    cfg["stopping_crit"] = 1

if args['Save']:
    cfg["save"] = 1
else:
    cfg["save"] = 0


cfg["pickle_path"] = "{}_wattdata_{}.pkl".format(cfg["model"]["type"],
                                                 cfg["model"]["size"])

with open("configuration.pkl", "wb") as f:
    pickle.dump(cfg, f)

# print(args)
# print(cfg)

sleep(3)

cpu_watt = "cpu_watt.sh"
ram = "mem_free.sh"
gpu = "radeontop.sh"

path_cpu_watt = Path(cpu_watt)
path_ram = Path(ram)
path_gpu = Path(gpu)

path_cpu_watt = str(Path(cpu_watt).absolute()) + '/' + cpu_watt
path_ram = str(Path(ram).absolute()) + '/' + ram
path_gpu = str(Path(gpu).absolute()) + '/' + gpu

if cmd[1] == "main_frequentist.py":
    cmd2 = [path_cpu_watt, "freq_{}_cpu_watts".format(wide)]
    cmd3 = [path_ram, "freq_{}_ram_use".format(wide)]
    cmd4 = [path_gpu, "freq_{}_flop_app".format(wide)]
elif cmd[1] == "main_bayesian.py":
    cmd2 = [path_cpu_watt, "bayes_{}_cpu_watts".format(wide)]
    cmd3 = [path_ram, "bayes_{}_ram_use".format(wide)]
    cmd4 = [path_gpu, "bayes_{}_flop_app".format(wide)]


path = sub.check_output(['pwd'])
path = path.decode()
path = path.replace('\n', '')

startWattCounter = 'python ' + path + '/amd_sample_draw.py'


p1 = sub.Popen(cmd)
p2 = sub.Popen(startWattCounter.split(), stdin=sub.PIPE, stdout=sub.PIPE,
               stderr=sub.PIPE)
p3 = sub.Popen(cmd2, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
p4 = sub.Popen(cmd3, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
p5 = sub.Popen(cmd4, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)

retcode = p1.wait()
print("Return code: {}".format(retcode))

p1.kill()
kill(p2.pid)
kill(p3.pid)
kill(p4.pid)
kill(p5.pid)
