import numpy as np


fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

# specific fft for our defined dynamic MR data
fft2c_mri  = lambda x, ax=(-2,-1) : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft2c_mri = lambda X, ax=(-2,-1) : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

# crop the data into the fixed size
def crop(img, crop_size=[16,64,64], step=[8,32,32]):
    nt, nx, ny = img.shape
    
    # the fs_0016_3T.h5 file just has 15 phases, thus we need to pad the data
    if nt == 15 and crop_size[0] == 16:
        img = np.concatenate((img[0, ...][None, ...], img), axis=0)
        nt, nx, ny = img.shape 

    dt = (nt - crop_size[0])
    dx = (nx - crop_size[1])
    dy = (ny - crop_size[2])

    data = []
    for t in range(dt//step[0] + 1):
        for x in range(dx//step[1] + 1):
            for y in range(dy//step[2] + 1):
                data.append(img[step[0] * t:step[0] * t + crop_size[0], step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])
        if dx != 0 or dy != 0:      
            data.append(img[step[0] * t:step[0] * t + crop_size[0], -crop_size[1]:, -crop_size[2]:])
                
    # each phase is important, so we need to add the last phases not included in the crop
    if dt % step[0] != 0:
        for x in range(dx//step[1] + 1):
            for y in range(dy//step[2] + 1):
                data.append(img[-crop_size[0]:, step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])
        if dx != 0 or dy != 0: 
            data.append(img[-crop_size[0]:, -crop_size[1]:, -crop_size[2]:])

    return data


