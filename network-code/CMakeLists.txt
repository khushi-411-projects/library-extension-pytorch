cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(khushu)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Define our library target
add_library(khushu SHARED KhushiFunctions.cpp)

# Enable C++14
target_compile_features(khushu PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(khushu "${TORCH_LIBRARIES}")
# Link against OpenCV
message(STATUS "foo include dir: ${OpenCV_LIBRARIES}")
target_link_libraries(khushu ${OpenCV_LIBRARIES})
target_link_libraries(khushu ${PYTHON_LIBRARIES})
