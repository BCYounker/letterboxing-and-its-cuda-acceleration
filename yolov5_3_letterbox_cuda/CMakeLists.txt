cmake_minimum_required(VERSION 3.0.0)
#set(CMAKE_BUILD_TYPE "")

#project(yolov5 VERSION 0.1.0)
project(yolov5 LANGUAGES CXX CUDA VERSION 0.1.0)

find_package(OpenCV REQUIRED)
message("OPENCV INCLUDE DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OPENCV LINK Libs: ${OpenCV_LIBS}")


add_executable(main yolov5_letterbox_cuda.cu)

include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

