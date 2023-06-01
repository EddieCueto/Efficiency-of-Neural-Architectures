def earlyStopping(early_stopping: list, train_acc: float, sensitivity: float=1e-9):
    early_stopping.append(train_acc)
    if epoch % 4 == 0 and epoch > 0:
        print("Value 1: {} >= {}, Value 2: {} >= {}, \
            Value 2: {} >= {}".format(early_stopping[0], \
            train_acc-sensitivity,early_stopping[1], \
            train_acc-sensitivity, early_stopping[2], train_acc-sensitivity))
        if abs(early_stopping[0]) >= train_acc-sensitivity and \
            abs(early_stopping[1]) >= train_acc-sensitivity and \
            abs(early_stopping[2]) >= train_acc-sensitivity:
            return None
        early_stopping = []
        

def energyBound(threshold: float=100000.0):
    if gpu_sample_draw.total_watt_consumed() > threshold:
        return None


def accuracyBound(train_acc: float, threshold: float=0.99):
    if train_acc >= threshold:
        return None