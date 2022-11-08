import tensorflow as tf
import numpy as np
from utils import *



def _float_feature(value):
    """Return a float_list form a float/double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Return a int64_list from a bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 创建图像数据的Example
def data_example(label):
    label = np.array(label) / np.abs(label).max()
    k = fft2c_mri(label)

    label_shape = label.shape
    label = label.flatten()

    k_shape = k.shape
    k = k.flatten()

    # don't worry about the dtype of k & label
    # .tolist() convert the data into python float no matter how
    feature = {
        'k_real': _float_feature(k.real.tolist()),
        'k_imag': _float_feature(k.imag.tolist()),
        'label_real': _float_feature(label.real.tolist()),
        'label_imag': _float_feature(label.imag.tolist()),
        'k_shape': _int64_feature(list(k_shape)),
        'label_shape': _int64_feature(list(label_shape))
    }

    exam = tf.train.Example(features=tf.train.Features(feature=feature))

    return exam
