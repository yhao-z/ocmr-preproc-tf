# common import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import time

# functional import
from utils import *
from bart import bart
import read_ocmr as read
from espirit import espirit_csm
from write_record import data_example
from dataset_tfrecord import get_dataset



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
    dataset = get_dataset(filename, 1)
    tf.print('dataset loaded.')

    len = 0
    for step, sample in enumerate(dataset):
        len = len + 1
        k, label = sample

        if step == 0:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(np.abs(k[0, 0, ...]), 'gray')
            plt.subplot(1,2,2)
            plt.imshow(np.abs(label[0, 0, ...]), 'gray')
            plt.savefig('./test_pic.png')
        
        if label.shape != [1, 16, 64, 64]:
            print('strange data')
    print(len)

    
def make_dataset(filedir, outdir, mode):
    filenames = glob.glob(filedir + '/*.h5')
    print(len(filenames))
    writer = tf.io.TFRecordWriter(outdir+'ocmr_'+mode+'.tfrecord')
    for filename in filenames:    
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
                x = ifft(k, (0, 1, 2)) # the MR multicoil images
                
                # use the first phase to estimate coil sensitivity(csm)
                # bart espirit (default) [mriron/bart](https://github.com/mrirecon/bart)
                # cuz bart takes 1s while python one takes 100s. bart is fast
                csm = bart(1, 'ecalib -m1', k[..., 0]) 
                ## python espirit 
                # csm = espirit_csm(k[..., 0]) 
                
                ## compare the bart and the python one by this
                # bart_or_pytn(k)
                
                # merge the multicoil images into single image
                singleImg = np.zeros_like(x[..., 0, :]) # [kx, ky, kz, phase]
                for coil in range(0, k.shape[3]):
                    singleImg = singleImg + x[:, :, :, coil, :] * csm[:, :, :, coil, None].conj()
                
                singleImg = singleImg.squeeze() # squeeze the kz dimension, cuz kz is always 1 in fs data [kx, ky, phase]
                nx = singleImg.shape[0]
                singleImg = singleImg[int(np.ceil(nx/4)):int(np.ceil(nx/4*3)),:,:] # crop the center part of the image, cuz the fs data is oversampled (see ocmr example)
                singleImg = np.transpose(singleImg, (2, 0, 1)) # [phase, kx, ky]
                
                # plt.figure(), plt.imshow(np.abs(singleImg[0,...])), plt.savefig('singleImg.png')
        
                if mode == 'train':
                    # [data enhancement] crop the data into small blocks
                    data = crop(singleImg)
                elif mode == 'test' or 'val':
                    data = [singleImg]

                # write the tfrecord
                for i in range(len(data)):
                    exam = data_example(data[i])
                    writer.write(exam.SerializeToString())
    writer.close()
    
    
if __name__ == '__main__':

    filesdir = '/workspace/data/OCMR/orig_h5/train/'
    outdir = '/workspace/data/OCMR/tfrecord/standard/'
    mode = 'train'
    make_dataset(filesdir, outdir, mode)
    test_tfrecord(outdir + '/ocmr_' + mode + '.tfrecord')



