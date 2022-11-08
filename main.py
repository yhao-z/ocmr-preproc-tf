# common import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

# functional import
from utils import *
import read_ocmr as read
from espirit import espirit_csm
from write_record import data_example


def make_record(filename):
    k_orig, _ = read.read_ocmr(filename) # read k-space data
    print(k_orig.shape)
    # [kx, ky, kz, coil, phase, set, slice, rep, avg]
    slices = k_orig.shape[-3]
    avgs = k_orig.shape[-1]
    for s in range(slices):
        for a in range(avgs):
            # set & rep are always 1 in fs data
            k = k_orig[..., 0, s, 0, a] # [kx, ky, kz, coil, phase]
            x = ifft(k, (0, 1, 2)) # the MR multicoil images
            csm = espirit_csm(k[..., 0]) # use the first phase to estimate coil sensitivity(csm)
            # merge the multicoil images into single image
            label = np.zeros_like(x[..., 0, :]) # [kx, ky, kz, phase]
            for coil in range(0, k.shape[3]):
                label = label + x[:, :, :, coil, :] * csm[:, :, :, coil, None].conj()
            
            label = label.squeeze() # squeeze the kz dimension, cuz kz is always 1 in fs data [kx, ky, phase]
            nx = label.shape[0]
            label = label[int(np.ceil(nx/4)):int(np.ceil(nx/4*3)),:,:] # crop the center part of the image, cuz the fs data is oversampled (see ocmr example)
            label = np.transpose(label, (2, 0, 1)) # [phase, kx, ky]
            data = crop(label)
            
            # get the last filename
            f = filename.split('\\')[-1]
            f = f.split('.')[0]
            writer = tf.io.TFRecordWriter('./'+f+'.tfrecord')
            for i in range(len(data)):
                exam = data_example(data[i])
                writer.write(exam.SerializeToString())
            writer.close()
            
            
if __name__ == '__main__':

    filename = r'G:\OCMR-New\OCMR_data\fs_0001_1_5T.h5'
    print(filename)
    make_record(filename)
    
    
    # filenames = glob.glob(r'G:\OCMR-New\OCMR_data\fs_*.h5')
    # print(len(filenames))
    # for filename in filenames:
    #     make_record(filename)



