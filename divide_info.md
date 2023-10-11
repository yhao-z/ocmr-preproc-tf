# divide the dataset

Each data in OCMR contains 9 dimensions, which is `[kx, ky, kz, coil, phase, set, slice, rep, avg]`. The first three are spatial dimensions; `coil` is the coil dimension; `phase` is the time dimension; `set & rep` are always 1 that can be ignored; `slice` is the dimension denoting slices; `avg` is the number of repeated images to get the average image with low noise. For example, the dimensions of the first file `fs_0001_1_5T.h5` is `[512, 208, 1, 15, 19, 1, 1, 1, 1]`.

We treat each `slice` and each `avg` as one data, and we denote `num` as the number of data in one specific file. For example,  

* `num` of the `fs_0001_1_5T.h5` `[512, 208, 1, 15, 19, 1, 1, 1, 1]` is 1; 
* `num` of the `fs_0012_3T.h5`   `[288, 112, 1, 30, 18, 1, 11, 1, 1]` is 11;
* `num` of the `fs_0016_3T.h5`     `[384, 150, 1, 30, 15, 1, 1, 1, 2]` is 2;

**The dimensions  and the `num` of each file is listed in [ocmr_data_by_yhao.csv](ocmr_data_by_yhao.csv)**

Through counting, the total `num` of the fully sample data in OCMR is 204. We use 124 for training, 40 for validation, and 40 for test, in the ratio about 7:2:2. 
