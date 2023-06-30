import pickle
from warnings import warn
from gpu_power_func import get_sample_of_gpu

with (open("configuration.pkl", "rb")) as file:
    while True:
        try:
            cfg = pickle.load(file)
        except EOFError:
            break


# pickle_name = "{}_wattdata_{}.pkl".format(model_t,size)
# print("GPU energy file config: {}".format(pickle_name))

# print(cfg)


if __name__ == '__main__':
    dataDump = []
    while True:
        try:
            dataDump.append(get_sample_of_gpu())
            with open(cfg["pickle_path"], 'wb') as f:
                pickle.dump(dataDump, f)
        except EOFError:
            warn('Pickle ran out of space')
        finally:
            f.close()
