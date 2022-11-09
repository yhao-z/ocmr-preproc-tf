# divide the dataset

Each data in OCMR contains 9 dimensions, which is `[kx, ky, kz, coil, phase, set, slice, rep, avg]`. The first three are spatial dimensions; `coil` is the coil dimension; `phase` is the time dimension; `set & rep` are always 1 that can be ignored; `slice` is the dimension denoting slices; `avg` is the number of repeated images to get the average image with low noise. For example, the dimensions of the first file `fs_0001_1_5T.h5` is `[512, 208, 1, 15, 19, 1, 1, 1, 1]`.

We treat each `slice` and each `avg` as one data, and we denote `num` as the number of data in one specific file. For example,  

* `num` of the `fs_0001_1_5T.h5` `[512, 208, 1, 15, 19, 1, 1, 1, 1]` is 1; 
* `num` of the `fs_0012_3T.h5`   `[288, 112, 1, 30, 18, 1, 11, 1, 1]` is 11;
* `num` of the `fs_0016_3T.h5`     `[384, 150, 1, 30, 15, 1, 1, 1, 2]` is 2;

**The dimensions  and the `num` of each file is listed in `ocmr_data_by_yhao.csv`**

Through counting, the total `num` of the fully sample data in OCMR is 204. We use 124 for training, 40 for validation, and 40 for test, in the ratio about 7:2:2. 
The filenames in each subset is listed as follows:

* Train

  > the files not in Validation & Test

* Validation

  > **filename - dimensions - num**
  >
  > fs_0060_1_5T.h5 - [384, 144, 1, 28, 21, 1, 12, 1, 1] - 12
  >
  > fs_0063_1_5T.h5 - [384, 156, 1, 24, 22, 1, 14, 1, 1] - 14
  >
  > fs_0068_1_5T.h5 - [384, 156, 1, 28, 22, 1, 14, 1, 1] - 14

* Test

  > **filename - num**
  >
  > fs_0051_1_5T.h5 - 1
  >
  > fs_0052_1_5T.h5 - 1
  >
  > fs_0054_1_5T.h5 - 1
  >
  > fs_0055_1_5T.h5 - 1
  >
  > fs_0058_1_5T.h5 - 1
  >
  > fs_0059_1_5T.h5 - 1
  >
  > fs_0061_1_5T.h5 - 1
  >
  > fs_0062_1_5T.h5 - 1
  >
  > fs_0064_1_5T.h5 - 1
  >
  > fs_0065_1_5T.h5 - 1
  >
  > fs_0066_1_5T.h5 - 1
  >
  > fs_0067_1_5T.h5 - 1
  >
  > fs_0069_1_5T.h5 - 12
  >
  > fs_0070_1_5T.h5 - 1
  >
  > fs_0071_1_5T.h5 - 1
  >
  > fs_0072_1_5T.h5 - 1
  >
  > fs_0073_1_5T.h5 - 1
  >
  > fs_0074_1_5T.h5 - 12

