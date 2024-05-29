import pandas as pd
import numpy as np
import glob
import random
import torch

FILE_PATTERN = '/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/Fixed_Data/*_*_*_TestSub*_ARM_*_*.csv'
WINDOW_SIZE = 1000 # 250 ms
OVERLAP = 760 # 190 ms
STEP_SIZE = WINDOW_SIZE - OVERLAP


def get_data(trainPercentage):
    files = glob.glob(FILE_PATTERN)
    random.shuffle(files)
    train_data = []
    test_data = []
    aux_list = []

    for file in files[: int(len(files) * trainPercentage)]:
        data = pd.read_csv(file, delimiter=';')
        data = data.drop(columns=['TID(SEC)'])
        aux_list.append(data)

    train_data = pd.concat(aux_list, ignore_index=True)
    aux_list = []

    for file in files[int(len(files) * trainPercentage):]:
        data = pd.read_csv(file, delimiter=';')
        data = data.drop(columns=['TID(SEC)'])
        aux_list.append(data)

    test_data = pd.concat(aux_list, ignore_index=True)

    train_windows, train_targets = window_data(train_data)
    test_windows, test_targets = window_data(test_data)

    train_input_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_output_tensor = torch.tensor(train_targets, dtype=torch.long)

    test_input_tensor = torch.tensor(test_windows, dtype=torch.float32)
    test_output_tensor = torch.tensor(test_targets, dtype=torch.long)

    return train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor

def window_data(data):
    windows = []
    targets = []

    for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = data[start:end]
        target = data.iloc[end - 1]['CLASS']
        windows.append(window[['AMP(V) Channel 1', 'AMP(V) Channel 2']].values)
        targets.append(target)

    return np.array(windows), np.array(targets)