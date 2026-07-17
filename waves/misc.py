import numpy as np

def flatten(t): # flattens out a list of lists (CHECK)
    return [item for sublist in t for item in sublist]

def dict_to_list(t):
    x = []
    for key, val in t.items():
        x.append(val)
    return x

def flatten2D(arr):
    if (len(arr) == 1):
        arr = arr[0]
    else:
        tempArr = arr[0]
        for i in range(1, len(arr)):
            tempArr = np.concatenate((tempArr, arr[i]), axis = 0)
        arr = tempArr
    return arr.tolist()
