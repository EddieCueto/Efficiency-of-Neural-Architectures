import os
import re
import pickle
import numpy as np
from warnings import warn

c = 1.0
pickle_name = "freq_wattdata_"+str(c)+".pkl"

def get_sample_of_gpu():
  from re import sub, findall
  import subprocess
  from subprocess import run

  no_graph = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
  no_version = "Failed to initialize NVML: Driver/library version mismatch"
  smi_string = run(['nvidia-smi'], stdout=subprocess.PIPE)
  smi_string = smi_string.stdout.decode('utf-8')
  smi_string = smi_string.split("\n")
  if smi_string[0] ==  no_graph:
    raise Exception("It seems that no NVIDIA GPU is installed")
  elif smi_string[0] ==  no_version:
    raise Exception("nvidia-smi version mismatch")
  else:
    return findall("[0-9]*MiB | [0-9]*W",smi_string[9])
    #for l in smi_string:
        #temp = findall("[0-9]*MiB | [0-9]*W",l)
        #if temp:
           #return temp

def total_watt_consumed():
    with open(pickle_name, 'rb') as f:
        x = pickle.load(f)
    x = np.array(x)
    x = x[:,0]
    y = [int(re.findall("\d+",xi)[0]) for xi in x]
    return sum(y)

if __name__ == '__main__':
  dataDump = []
  #var = True
  #pickling_on = open("wattdata.pickle","wb")
  while True:
    #from run_service import retcode
    try:
      dataDump.append(get_sample_of_gpu())
      with open(pickle_name, 'wb') as f:
        pickle.dump(dataDump, f)
    except EOFError:
      warn('Pickle ran out of space')
      c += 0.01
    finally:
        f.close()

    #if retcode == 0:
      #break

  #pickle.dump(dataDump, pickling_on)
  #pickling_on.close()





