cmake_minimum_required(VERSION 3.1)
project(inference)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Torch_INCLUDE_DIRS})

add_executable(run unet_inference.cpp)
target_link_libraries(run ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
target_compile_features(run PUBLIC cxx_range_for)
set_property(TARGET run PROPERTY CXX_STANDARD 14)
