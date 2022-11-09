# ocmr-preproc-tf

![GitHub release (latest by date)](https://img.shields.io/github/v/release/yhao-z/ocmr-preproc-tf&display_name=tag&style=flat) ![tf](https://img.shields.io/badge/Tensorflow-2.9.0-blue) ![data](https://img.shields.io/badge/Dataset-OCMR-brightgreen)

This is a pipeline in `Tensorflow` framework to preprocess the `OCMR (Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging)` dataset into the `.tfrecord` files.

You can attach the details of the `OCMR` dataset by the following materials.

> Paper:
>
> Chen, Chong, et al. "OCMR (v1. 0)--Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging." *arXiv preprint arXiv:2008.03410* (2020).
>
> GitHub:
>
> [MRIOSU/OCMR: OCMR (Open-Access Repository for Multi-Coil k-space Data for Cardiovascular Magnetic Resonance Imaging) (github.com)](https://github.com/MRIOSU/OCMR)
>
> Homepage:
>
> [https://ocmr.info/](https://ocmr.info/)

We note that the `num` of fully sampled data of `OCMR` is 204, and we divide the data into 124 for training; 40 for validation; and 40 for test, see [devide_info.md](divide_info.md) for the definition of `num` and more info. The ratio of train, validation and test is about 7:2:2.

We use ESPIRiT method to estimate the coil sensitivity maps of the multi-coil images and then merge them into single-coil images.

We perform the data augmentation for the training data, i.e., crop the single-coil data  into `64×64×16 (x,y,t)` with the step of `32×32×8`. Eventually, we get 2491 training images.

## 1. get the preprocessed tfrecord file

You can download the preprocessed tfrecord files via [my onedrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/yhao-zhang_stu_hit_edu_cn/Ev1ZhrDUVU1EmJHg81y1-eYBdMRRbzb1SpXxQJtodMGsfg?e=NfFFXI)

## 2. Run the code

run `main-bart.py` after the following steps done. u may change the `dirs` by yourself.

### 2.1 Download the data

click [HERE](https://ocmr.s3.amazonaws.com/data/ocmr_cine.tar.gz) to download the original full data

> u can get more info in the OCMR Homepage [https://ocmr.info/](https://ocmr.info/), also the download link.

### 2.2 divide the data into train/val/test

see [devide_info.md](divide_info.md)

### 2.3 env & requirements

#### Linux ( suggested )

* **if you are using `Docker`, you can easily pull the image and create a container via**

  ```
  docker pull yhaoz/tf:2.9.0-bart
  ```

* else

  * install `bart`

    > We use bart to estimate the coil sensitivity maps (csm) to combine the multi-coil image into a single-coil image
    >
    > For more info, 
    >
    > GitHub: [mrirecon/bart: BART: Toolbox for Computational Magnetic Resonance Imaging (github.com)](https://github.com/mrirecon/bart)
    >
    > Homepage: [BART Toolbox (mrirecon.github.io)](https://mrirecon.github.io/bart/)
  
  * `pip install -r requirements.txt`
  
    ```
    ismrmrd==1.12.5
    matplotlib==3.5.2
    numpy==1.22.3
    tensorflow==2.9.0
    ```

  * u may also need to install the `ismrmrd-python-tools` manually
  
    > GitHub: [ismrmrd/ismrmrd-python-tools: ISMRMRD Python Toolbox (github.com)](https://github.com/ismrmrd/ismrmrd-python-tools)	
  
    ```
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git
    pip install ismrmrd-python-tools-master
    ```

#### windows ( NOT suggested )

Except `bart`, u should install other requirements, `requiements.txt` & `ismrmrd-python-tools`. Then, u should finetune the code `main-bart.py`

* remove the `bart` corresponding codes

* comment the line using `bart` calculate the coil sensitivity maps

  ```python
  # csm = bart(1, 'ecalib -m1', k[..., 0]) 
  ```

* use the espirit implementation in python via
  ```python
  csm = espirit_csm(k[..., 0])
  ```

  > ESPIRiT is a method to estimate the coil sensitivity maps (csm)
  >
  > For more info, 
  >
  > Paper: 
  > Uecker, Martin, et al. "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA." *Magnetic resonance in medicine* 71.3 (2014): 990-1001.
  >
  > GitHub:
  >
  > [mikgroup/espirit-python: Python ESPIRiT implementation. (github.com)](https://github.com/mikgroup/espirit-python)

* Note that using `bart` to calculate the csm takes **1 second** while using the python implementation of espirit takes **more than 100 seconds**

## 3. Thanks

We thank the following GitHub repos referred to which we finally complete the preprocessing code.

> [wenqihuang/LS-Net-Dynamic-MRI: This the code repository for paper "Deep Low-rank plus Sparse Network for Dynamic MR Imaging". (github.com)](https://github.com/wenqihuang/LS-Net-Dynamic-MRI)
>
> [Keziwen/SLR-Net: Code for our work: "Learned Low-rank Priors in Dynamic MR Imaging" (github.com)](https://github.com/Keziwen/SLR-Net)
>
> [mrirecon/bart: BART: Toolbox for Computational Magnetic Resonance Imaging (github.com)](https://github.com/mrirecon/bart)
>
> [MRIOSU/OCMR: OCMR (Open-Access Repository for Multi-Coil k-space Data for Cardiovascular Magnetic Resonance Imaging) (github.com)](https://github.com/MRIOSU/OCMR)
>
> [mikgroup/espirit-python: Python ESPIRiT implementation. (github.com)](https://github.com/mikgroup/espirit-python)

## 4. License

This repo is with the [GPL-3.0 license](https://github.com/yhao-z/ocmr-preproc-tf/blob/main/LICENSE).

