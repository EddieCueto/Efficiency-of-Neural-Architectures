import pickle
from warnings import warn
from gpu_power_func import get_sample_of_gpu

with (open("configuration.pkl", "rb")) as file:
    while True:
        try:
            cfg = pickle.load(file)
        except EOFError:
            break

#with open("frq", "r") as file:
#    frq = int(file.read())

#with open("bay", "r") as file:
#    bay = int(file.read())

#if frq == 1:
#  model_t = "freq"
#  with open("tmp", "r") as file:
#    size = float(file.read())

#if bay == 1:
#  model_t = "bayes"
#  with open("tmp", "r") as file:
#    size = int(file.read())

#pickle_name = "{}_wattdata_{}.pkl".format(model_t,size)
#print("GPU energy file config: {}".format(pickle_name))

#print(cfg)


if __name__ == '__main__':
  dataDump = []
  #var = True
  #pickling_on = open("wattdata.pickle","wb")
  while True:
    try:
      dataDump.append(get_sample_of_gpu())
      with open(cfg["pickle_path"], 'wb') as f:
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





