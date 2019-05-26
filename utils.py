from glob import glob
from itertools import groupby
from os import path
from random import shuffle

import numpy as np
from scipy.cluster.vq import kmeans

WEIGHTS_FORMAT = "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"


def get_latest_weights_path(p):
    fp = glob(p + "*.h5")[-1]
    _, f_name = path.split(fp)
    tmp = f_name[2:5]
    if tmp.isdecimal():
        epoch = int(tmp)
    else:
        epoch = 0
    return epoch, fp


result = [
    ("1", 0.3, ((14, 1), (50, 50))),
    ("2", 0.9, ((14, 1), (50, 50))),
    ("3", 0.5, ((14, 1), (50, 50))),
    ("4", 0.9, ((45, 1), (50, 50))),
    ("5", 0.8, ((41, 1), (50, 50))),
    ("6", 0.8, ((41, 1), (50, 50))),
    ("7", 0.8, ((80, 1), (50, 50))),
    ("8", 0.8, ((78, 1), (50, 50))),
    ("9", 0.8, ((85, 1), (50, 50))),
    ("10", 0.8, ((115, 1), (50, 50))),
    ("11", 0.7, ((110, 1), (50, 50))),
    ("12", 0.8, ((112, 1), (50, 50))),
    ("13", 0.5, ((113, 1), (50, 50))),
]
# simulate random input
shuffle(result)


def get_captcha(result, n_char: int):
    result.sort(key=lambda node: node[2][0][0])
    xs = np.array([i[2][0][0] for i in result], dtype=np.float32)
    # boxes = np.array([i[2] for i in result], dtype=np.float32)
    code_book, distortion = kmeans(xs, n_char)
    code_book = code_book.reshape((-1, 1))

    gb = groupby(result, key=lambda node: np.argmin(np.linalg.norm(code_book - node[2][0][0], axis=1)))
    for i, l in gb:
        cls_name, confident, boxes = max(l, key=lambda x: x[1])
        yield cls_name, confident, boxes


if __name__ == '__main__':
    print(list(get_captcha(result, 4)))
    pass
