## Creating Library Extensions in PyTorch.

### Motivation

1. Perform inference in C++ (get an idea of how PyTorch C++ API is used, trying to get closer to open-source).
2. Using a pre-trained model (resnet18) in C++.
3. Trying to create Python Bindings using pybind11 (realization: not the ideal way, extending torch ops is preferred using `TORCH_LIBRARY` class)

### Files

1. `CMakeLists.txt`: Used to configure the project (example: finding OpenCV, PyTorch libraries, including their paths in the compiler flags, linking them to the project)
2. `MNISTInference.cpp`: Contains inference code to perform inference on given input image and using given model path (also contains template of using `pybind11`)

### Notes

1. My goal was to learn and get closer to PyTorch code-base, and get a feel of development.
2. Using OpenCV is not the most recommended way I feel, if someone has to just load images - then normal fstream should work. (a code sample is shared in another file)
3. ABI issues are real (I still need to understand what they are exactly, but have faced them a lot already) - lesson: building PyTorch from source is still better if I've to play around with these python bindings. 

### Instructions

```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/home/khushi/Documents/libtorch ..
make
```
