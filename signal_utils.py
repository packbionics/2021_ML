from scipy import signal


def filter(filter, arr):
    for column in range(arr.shape[1]):
        arr[:, column] = signal.sosfilt(filter, arr[:, column])
    return arr