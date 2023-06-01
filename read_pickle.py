import pickle

gpu_data = []
with (open("bayes_wattdata_1.pkl", "rb")) as openfile:
    while True:
        try:
            gpu_data.append(pickle.load(openfile))
        except EOFError:
            break

exp_data = []
with (open("bayes_exp_data_1.pkl", "rb")) as openfile:
    while True:
        try:
            exp_data.append(pickle.load(openfile))
        except EOFError:
            break