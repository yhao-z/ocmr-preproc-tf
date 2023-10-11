import numpy as np


fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

# specific fft for our defined dynamic MR data
fft2c_mri  = lambda x, ax=(-2,-1) : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft2c_mri = lambda X, ax=(-2,-1) : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

# crop the data into the fixed size
def crop(img, crop_size=[16,128,128], step=[8,32,32]):
    nt, nx, ny = img.shape
    
    # # the fs_0016_3T.h5 file just has 15 phases, thus we need to pad the data
    # if nt == 15 and crop_size[0] == 16:
    #     img = np.concatenate((img[0, ...][None, ...], img), axis=0)
    #     nt, nx, ny = img.shape 

    dt = (nt - crop_size[0])
    dx = (nx - crop_size[1])
    dy = (ny - crop_size[2])
    
    wt = dt // step[0]
    wx = dx // step[1]
    wy = dy // step[2]
    
    rt = dt % step[0]
    rx = dx % step[1]
    ry = dy % step[2]
    
    if 0 < rx <= 10:
        img = img[:,rx//2:rx//2-rx,:]
        rx = 0
    if 0 < ry <= 10:
        img = img[:,:,ry//2:ry//2-ry]
        ry = 0
        
    data = []
    for t in range(wt + 1):
        for x in range(wx + 1):
            for y in range(wy + 1):
                data.append(img[step[0] * t:step[0] * t + crop_size[0], step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[step[0] * t:step[0] * t + crop_size[0], step[1] * x:step[1] * x + crop_size[1], -crop_size[2]:])
        if rx != 0: 
            for y in range(wy + 1):
                data.append(img[step[0] * t:step[0] * t + crop_size[0],         -crop_size[1]:                , step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[step[0] * t:step[0] * t + crop_size[0],         -crop_size[1]:                , -crop_size[2]:])
    if rt != 0:
        for x in range(wx + 1):
            for y in range(wy + 1):
                data.append(img[-crop_size[0]:                        , step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[-crop_size[0]:                        , step[1] * x:step[1] * x + crop_size[1], -crop_size[2]:])
        if rx != 0: 
            for y in range(wy + 1):
                data.append(img[-crop_size[0]:                        ,         -crop_size[1]:                , step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[-crop_size[0]:                        ,         -crop_size[1]:                , -crop_size[2]:])

    return data



# crop the data into the fixed size
def crop_multicoil(img, crop_size=[16,128,128], step=[8,32,32]):
    nc, nt, nx, ny = img.shape

    dt = (nt - crop_size[0])
    dx = (nx - crop_size[1])
    dy = (ny - crop_size[2])
    
    wt = dt // step[0]
    wx = dx // step[1]
    wy = dy // step[2]
    
    rt = dt % step[0]
    rx = dx % step[1]
    ry = dy % step[2]
    
    if 0 < rx <= 10:
        img = img[...,rx//2:rx//2-rx,:]
        rx = 0
    if 0 < ry <= 10:
        img = img[...,:,ry//2:ry//2-ry]
        ry = 0
        
    data = []
    for t in range(wt + 1):
        for x in range(wx + 1):
            for y in range(wy + 1):
                data.append(img[..., step[0] * t:step[0] * t + crop_size[0], step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[..., step[0] * t:step[0] * t + crop_size[0], step[1] * x:step[1] * x + crop_size[1], -crop_size[2]:])
        if rx != 0: 
            for y in range(wy + 1):
                data.append(img[..., step[0] * t:step[0] * t + crop_size[0],         -crop_size[1]:                , step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[..., step[0] * t:step[0] * t + crop_size[0],         -crop_size[1]:                , -crop_size[2]:])
    if rt != 0:
        for x in range(wx + 1):
            for y in range(wy + 1):
                data.append(img[..., -crop_size[0]:                        , step[1] * x:step[1] * x + crop_size[1], step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[..., -crop_size[0]:                        , step[1] * x:step[1] * x + crop_size[1], -crop_size[2]:])
        if rx != 0: 
            for y in range(wy + 1):
                data.append(img[..., -crop_size[0]:                        ,         -crop_size[1]:                , step[2] * y:step[2] * y + crop_size[2]])
            if ry != 0:
                data.append(img[..., -crop_size[0]:                        ,         -crop_size[1]:                , -crop_size[2]:])

    return data

