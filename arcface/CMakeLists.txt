cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.

include_directories(../include)
add_definitions(-std=c++11)
link_directories(../lib/ncnn ../lib/dlib ../lib/ncnn/ippicv_lnx/lib/intel64)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING
   "Choose the type of build, options are: Debug Release
   RelWithDebInfo MinSizeRel." FORCE)

FIND_PACKAGE( OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})

   add_executable(main main.cpp mtcnn.cpp base.cpp arcface.cpp)
   target_link_libraries(main dlib ncnn ippicv ${OpenCV_LIBS})
else()
   message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
endif()
