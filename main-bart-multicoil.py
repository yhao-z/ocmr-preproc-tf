# common import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import time
import os
import scipy.io as scio
# functional import
from utils import *
from bart import bart
import read_ocmr as read
from espirit import espirit_csm
from write_record import data_example_multicoil
from dataset_tfrecord import get_dataset, get_dataset_multicoil



def bart_or_pytn(k):
    # use the first phase to estimate coil sensitivity(csm)
    
    # bart espirit (-r24 -k6 -t0.001 -c0.8) by default
    t0 = time.time()
    csm1 = bart(1, 'ecalib -m1', k[..., 0]) 
    print('calc csm via bart, time: %.3f' % (time.time()-t0))
    
    # python espirit (-r24 -k6 -t0.01 -c0.95) by default
    # these two implements takes diff params, why?
    # if the same, they turn out totally diff results
    # i guess the underlying imple is diff
    # thus i choose a satisfying params for this python code
    t1 = time.time()
    csm2 = espirit_csm(k[..., 0]) 
    print('calc csm via pytn, time: %.3f' % (time.time()-t1))
    print((csm2-csm1).max())
    
    nc = 15
    plt.figure(figsize=(16,20))
    for i in range(nc):
        plt.subplot(1, nc, i+1)
        plt.imshow(abs(csm1[:,:,0,i]).squeeze(), cmap='gray')
        plt.title('Coil sensitivity {}'.format(i))
    plt.savefig('csm1.png')
        
    plt.figure(figsize=(16,20))
    for i in range(nc):
        plt.subplot(1, nc, i+1)
        plt.imshow(abs(csm2[:,:,0,i]).squeeze(), cmap='gray')
        plt.title('Coil image {}'.format(i))
    plt.savefig('csm2.png')

    
def test_tfrecord(filename):
    '''
        visualize the already-to-use dataset to check
        
    '''
    dataset = get_dataset_multicoil(filename, 1)
    tf.print('dataset loaded.')

    len = 0
    for step, sample in enumerate(dataset):
        len = len + 1
        k, csm = sample
        # print(k.shape)
        # print(csm.shape)

        if step == 0:
            scio.savemat('val1.mat', {'k':k.numpy()})
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(np.abs(ifft2c_mri(k[0, 0, 0, ...].numpy())), 'gray')
            plt.subplot(1,2,2)
            plt.imshow(np.abs(csm[0, 0, 0, ...]), 'gray')
            plt.savefig('./test_pic.png')
    print(len)

    
def make_dataset(filedir, outdir, mode):
    filenames = glob.glob(filedir + '/*.h5')
    # print(filenames.split('/')[-1])
    print(len(filenames))
    if mode == 'val' or 'test':
        writer = tf.io.TFRecordWriter(outdir+'ocmr_multicoil_'+mode+'.tfrecord')
    for filename in filenames:   
        f__ = filename.split('/')[-1]
        f__ = f__.split('.')[0]
        if mode == 'train':
            writer = tf.io.TFRecordWriter(outdir+f__+'.tfrecord') 
        print(f"********* data {filename.split('/')[-1]} processing **********")    
        k_orig, _ = read.read_ocmr(filename) # read k-space data
        print(k_orig.shape)
        # [kx, ky, kz, coil, phase, set, slice, rep, avg]
        slices = k_orig.shape[-3]
        avgs = k_orig.shape[-1]
        for s in range(slices):
            for a in range(avgs):
                # set & rep are always 1 in fs data
                k = k_orig[..., 0, s, 0, a] # [kx, ky, kz, coil, phase]
                
                k = np.transpose(k.squeeze(), [2,3,0,1]) # [coil, phase, kx, ky]
                print(k.shape)
                
                x = ifft2c_mri(k)
                nx = x.shape[-2]
                x = x[...,int(np.ceil(nx/4)):int(np.ceil(nx/4*3)),:]
                
                # plt.figure(), plt.imshow(np.abs(singleImg[0,...])), plt.savefig('singleImg.png')
                
                if mode == 'train':
                    # [data enhancement] crop the data into small blocks
                    data = crop_multicoil(x)
                elif mode == 'test' or 'val':
                    data = [x]

                # write the tfrecord
                for i in range(len(data)):
                    img = data[i].sum(1)
                    k0 = fft2c_mri(img)[..., None].transpose((1,2,3,0))
                    csm = bart(1, 'ecalib -m1', k0)
                    csm = np.transpose(csm, [3,2,0,1])
                    
                    exam = data_example_multicoil(fft2c_mri(data[i]), csm)
                    writer.write(exam.SerializeToString())
        writer.close() if mode=='train' else None
    writer.close() if mode=='val' or 'test' else None
    
    
    
if __name__ == '__main__':
    
    mode = 'train'
    filesdir = '/workspace/data/OCMR/orig_h5/train/'
    outdir = '/workspace/E/ocmr/train/'
    if os.path.exists(outdir):
        print('outpath exists')
    else:
        os.makedirs(outdir)
    
    make_dataset(filesdir, outdir, mode)
    test_tfrecord(glob.glob(outdir + '/*.tfrecord'))

    mode = 'val'
    filesdir = '/workspace/data/OCMR/orig_h5/val/'
    outdir = '/workspace/E/ocmr/val/'
    if os.path.exists(outdir):
        print('outpath exists')
    else:
        os.makedirs(outdir)
    
    make_dataset(filesdir, outdir, mode)
    test_tfrecord(glob.glob(outdir + '/*.tfrecord'))
    
    
    mode = 'test'
    filesdir = '/workspace/data/OCMR/orig_h5/test/'
    outdir = '/workspace/E/ocmr/test/'
    if os.path.exists(outdir):
        print('outpath exists')
    else:
        os.makedirs(outdir)
    
    make_dataset(filesdir, outdir, mode)
    test_tfrecord(glob.glob(outdir + '/*.tfrecord'))



