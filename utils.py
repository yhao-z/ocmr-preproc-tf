import numpy as np


fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

# specific fft for our defined dynamic MR data
fft2c_mri  = lambda x, ax=(-2,-1) : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft2c_mri = lambda X, ax=(-2,-1) : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

# crop the data into the fixed size
def crop(img, crop_size=[16,64,64], step=[8,32,32]):
    nt, nx, ny = img.shape

    dt = (nt - crop_size[0])
    dx = (nx - crop_size[1])
    dy = (ny - crop_size[2])

    data = []
    for t in range(dt//step[0] + 1):
        for x in range(dx//step[1] + 1):
            for y in range(dy//step[2] + 1):
                data.append(img[step[0] * t:step[0] * t + crop_size[0], step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])

    if dt % step[0] != 0 or dx % step[1] != 0 or dy % step[2] != 0:
        if not (dt == 0 and dx < 10 and dy < 10):
            data.append(img[-crop_size[0]:, -crop_size[1]:, -crop_size[2]:])
    return data


