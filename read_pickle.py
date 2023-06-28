import pickle

gpu_data = []
with (open("bayesian_wattdata_3.pkl", "rb")) as openfile:
    while True:
        try:
            gpu_data = pickle.load(openfile)
        except EOFError:
            break

#exp_data = []
#with (open("bayes_exp_data_6.pkl", "rb")) as openfile:
#    while True:
#        try:
#            exp_data = pickle.load(openfile)
#        except EOFError:
#            break
