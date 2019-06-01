import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F


def list_to_torch(data):
    for i in range(len(data)):
        if isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


def data_to_device(data, device):
    if isinstance(data, dict):
        keys = list(data.keys())
    else:
        keys = range(len(data))
    for i in keys:
        if isinstance(data[i], list) or isinstance(data[i], dict):
            data[i] = data_to_device(data[i], device)
        else:
            if isinstance(data[i], torch.Tensor):
                try:
                    data[i] = data[i].to(device)
                except:
                    print('error', i, data[i], type(data[i]))
                    raise
    return data
