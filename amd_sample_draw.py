import os
import re
import pickle
import numpy as np
from warnings import warn

with open("frq", "r") as file:
    frq = int(file.read())

with open("bay", "r") as file:
    bay = int(file.read())

if frq == 1:
  model_t = "freq"
  with open("tmp", "r") as file:
    size = float(file.read())

if bay == 1:
  model_t = "bayes"
  with open("tmp", "r") as file:
    size = int(file.read())

pickle_name = "{}_wattdata_{}.pkl".format(model_t,size)
print("GPU energy file config: {}".format(pickle_name))

def get_sample_of_gpu():
  from re import sub, findall
  import subprocess
  from subprocess import run

  no_graph = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
  no_version = "Failed to initialize NVML: Driver/library version mismatch"
  smi_string = run(['rocm-smi', '-P', '--showvoltage', '--showmemuse'], stdout=subprocess.PIPE)
  smi_string = smi_string.stdout.decode('utf-8')
  smi_string = smi_string.split("\n")
  smi_string = list(filter(lambda x: x, smi_string))
  if smi_string[0] ==  no_graph:
    raise Exception("It seems that no AMD GPU is installed")
  elif smi_string[0] ==  no_version:
    raise Exception("rocm-smi version mismatch")
  else:
    results= []
    gpuW0 = findall("[0-9]*\.[0-9]*",smi_string[2]) 
    gpuW1 = findall("[0-9]*\.[0-9]*",smi_string[4])
    gpuM0 = findall("[0-9]+",smi_string[7]) 
    gpuM1 = findall("[0-9]+",smi_string[9])
    gpuV0 = findall("[0-9]+",smi_string[13]) 
    gpuV1 = findall("[0-9]+",smi_string[14])
    results.append(float(gpuW0[0]) + float(gpuW1[0]))
    if len(gpuM0) == 2 and len(gpuM1) == 2:
      results.append(int(gpuM0[1]) + int(gpuM1[1]))
    elif len(gpuM0) == 2:
      results.append(gpuM0[1])
    elif len(gpuM1) == 2:
      results.append(gpuM1[1])
    results.append(int(gpuV0[1]) + int(gpuV1[1]))
    return results
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
      size += 0.01
    finally:
      f.close()

    #if retcode == 0:
      #break

  #pickle.dump(dataDump, pickling_on)
  #pickling_on.close()





