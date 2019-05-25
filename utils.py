from glob import glob

from os import path

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
