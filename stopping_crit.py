import amd_sample_draw
from time import sleep


def earlyStopping(early_stopping: list, train_acc: float, epoch: int, sensitivity: float=1e-9):
    early_stopping.append(train_acc)
    if epoch % 4 == 0 and epoch > 0:
        print("Value 1: {} > Value 2: {} > \
            Value 3: {}".format(early_stopping[0], \
            abs(early_stopping[1]-sensitivity), \
            abs(early_stopping[2]-sensitivity)))
        if train_acc > 0.5:
            if early_stopping[0] > abs(early_stopping[1]-sensitivity) and \
               early_stopping[1] > abs(early_stopping[2]-sensitivity):
                print("Stopping Early")
                return 1
        del early_stopping[:]
    return 0
        

def energyBound(threshold: float=100000.0):
    try:
        energy = amd_sample_draw.total_watt_consumed()
    except Exception as e:
        sleep(3)
        energy = amd_sample_draw.total_watt_consumed()
    print("Energy used: {}".format(energy))
    if energy > threshold:
        print("Energy bound achieved")
        return 1
    return 0

def accuracyBound(train_acc: float, threshold: float=0.99):
    if train_acc >= threshold:
        print("Accuracy bound achieved")
        return 1
    return 0
