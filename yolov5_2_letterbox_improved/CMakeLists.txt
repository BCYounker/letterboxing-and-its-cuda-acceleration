cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE "")

project(yolov5 VERSION 0.1.0)

find_package(OpenCV REQUIRED)
message("OPENCV INCLUDE DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OPENCV LINK Libs: ${OpenCV_LIBS}")

add_executable(main yolov5_letterbox_improved.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
