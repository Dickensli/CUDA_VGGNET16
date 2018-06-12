# CUDA_VGGNET16
Implement VGGNet16 using cuda. Implement 4 conv kernels, 2 pool kernels.

## Requirement
Cuda-9.0, Cudnn-7.5, cuda-toolkit, nvcc

## Usage
### cmake
Under `root` directory `cmake .` to add dependency and import NVTX and NVML.

### import weights and bias
download vgg16 caffemodel "https://drive.google.com/open?id=1b5O7U50hLuOWFdQViomDU7WAsM2LwuaF" and unzip it by `run -xvf data.zip`.

### compile
Under `CUDA_VGGNET16/Src/vgg` run `make` to generate runnable file and ptx file.

### find an image
find an image and run `python Convert_image_to_txt.py` to generate 224 * 224 size image.
 
### run
format: `./vgg image_path`
