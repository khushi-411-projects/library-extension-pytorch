## Creating Library Extensions in PyTorch.

Thanks to [Kushashwa Ravi Shrimali](https://github.com/krshrimali) for inspiring me and mentoring me throughout this project. 

### Motivation

1. Perform inference in C++ (get an idea of how PyTorch C++ API is used, trying to get closer to open-source).
2. Using a pre-trained model (resnet18) in C++.
3. Trying to create Python Bindings using pybind11 (realization: not the ideal way, extending torch ops is preferred using `TORCH_LIBRARY` class)

### Files

1. `CMakeLists.txt`: Used to configure the project (example: finding OpenCV, PyTorch libraries, including their paths in the compiler flags, linking them to the project)
2. `MNISTInference.cpp`: Contains inference code to perform inference on given input image and using given model path (also contains template of using `pybind11`)

### Instructions

```
mkdir build 
cd build
cmake -DCMAKE_PREFIX_PATH=/home/khushi/Documents/libtorch ..
make
```
