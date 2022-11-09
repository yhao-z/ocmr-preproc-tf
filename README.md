# ocmr-preproc-tf

This is a pipeline in `Tensorflow` framework to preprocess the `OCMR (Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging)` dataset into the `.tfrecord` files.

## 1. OCMR

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

## 2. Run the code

run `main-bart.py` after the following steps done. u may change the dir by yourself.

### 2.1 Download the data

click [HERE](https://ocmr.s3.amazonaws.com/data/ocmr_cine.tar.gz) to download the full data

> u can get more info in the OCMR Homepage [https://ocmr.info/](https://ocmr.info/), also the download link.

### 2.2 divide the data into train/val/test

see [devide_info.md](divide_info.md)

### 2.3 env & requirements

#### Linux ( suggested )

* **if you are using `Docker`, you can easily create the env via**

  ```
  docker pull yhaoz/tf:2.9.0-bart
  ```

* else

  * install `bart`

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

* remove the `bart` corresponding code

* comment the line using `bart` calculate the coil sensitivity maps

  ```python
  csm = bart(1, 'ecalib -m1', k[..., 0]) 
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

* Note that using `bart` calculates the csm takes **1 second** while using the python implementation of espirit takes **more than 100 seconds**

