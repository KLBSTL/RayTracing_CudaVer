cmake_minimum_required(VERSION 3.31)
project(RT_CudeVer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(RT_CudeVer main.cpp
        src/vec3.h
        src/ray.h
        src/camera.h
        src/hitable.h
        src/hitable_list.h
        src/material.h
        src/sphere.h
        src/primitive.h
        src/scene.h
        src/stb_image_write.h)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
