import os
import re
import pickle
import numpy as np


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

def total_watt_consumed(pickle_name):
  with (open(pickle_name, "rb")) as file:
      while True:
          try:
              x = pickle.load(file)
          except EOFError:
              break
  x = np.array(x)
  x = x[:,0]
  y = [float(re.findall("\d+.\d+",xi)[0]) for xi in x]
  return sum(y)