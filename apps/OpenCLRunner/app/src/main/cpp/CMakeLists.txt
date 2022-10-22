
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

add_subdirectory(libopencl-stub)

# Declares and names the project.

project("openclrunner")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

include_directories(
        libopencl-stub/include/
        libopencl-stub/include/CL/
        common/
        implementations/
)

file(GLOB src_files
        openclrunner.cpp
        common/common_functions.cpp
        implementations/vector_add.cpp
        implementations/mace_inceptionv3_kernel.cpp
        implementations/tvm_resnet50v2_conv_kernel.cpp
        implementations/simple_mad.cpp
)

add_library( # Sets the name of the library.
             ${PROJECT_NAME}

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${src_files} )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log
              android-lib
              android
            )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       ${PROJECT_NAME}

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       ${android-lib}
                       android
                       OpenCL
                    )
