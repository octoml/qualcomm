cmake_minimum_required(VERSION 3.10.2)
project("opencl_cpp_runner")

set(CMAKE_CXX_STANDARD 14)

if (BUILD_FOR_MOBILE)
    add_subdirectory(libopencl-stub)

    set(OpenCL_INCLUDE_DIRS
            libopencl-stub/include/
            libopencl-stub/include/CL/
        )
    set(OpenCL_LIBRARIES OpenCL)
else()
    message(WARNING "If you want to build this project for Android, pass -DBUILD_FOR_MOBILE=ON option to CMake")
    find_package(OpenCL)
    if(NOT ${OpenCL_FOUND})
        message(FATAL_ERROR "Cannot find OpenCL")
    endif()
endif()

include_directories(
        common/
        implementations/
        ${OpenCL_INCLUDE_DIRS}
)

file(GLOB src_files
        main.cpp
        common/*.cpp
        implementations/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME}
    ${src_files}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OpenCL_LIBRARIES}
)
