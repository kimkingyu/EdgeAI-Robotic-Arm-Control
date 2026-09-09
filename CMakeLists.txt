cmake_minimum_required(VERSION 3.16)
project(EdgeAI_Robotic_Arm_Control LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Compiler flags for ARM NEON and optimization
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_definitions(-DARCH_AARCH64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn_api/include
)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Threads
find_package(Threads REQUIRED)

# Link directories for RKNN API
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn_api/lib
)

# Output binaries to bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Main executable
add_executable(edge_arm_control
    src/main.cpp src/kinematics.cpp
)

target_link_libraries(edge_arm_control
    PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# Offline driver/kinematics regression tests. All device syscalls are replaced
# by test doubles at link time, so ctest does not access an I2C bus or camera.
include(CTest)
if(BUILD_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_cpp_control tools/test_cpp_control.cpp src/kinematics.cpp)
    target_compile_options(test_cpp_control PRIVATE -Wall -Wextra -Wpedantic)
    target_link_options(test_cpp_control PRIVATE
        "LINKER:--wrap=open" "LINKER:--wrap=ioctl" "LINKER:--wrap=read"
        "LINKER:--wrap=write" "LINKER:--wrap=close" "LINKER:--wrap=usleep"
        "LINKER:--wrap=__read_chk" # glibc fortified read, including sanitizer builds
    )
    add_test(NAME cpp_control_offline COMMAND test_cpp_control)

    # 手眼变换导出程序，供 tools/test_handeye_parity.py 做跨语言比对
    add_executable(handeye_dump tools/handeye_dump.cpp)
    target_compile_options(handeye_dump PRIVATE -Wall -Wextra)
    target_link_libraries(handeye_dump PRIVATE ${OpenCV_LIBS})
endif()

# Optional linking for RKNN runtime on aarch64
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn_api/lib/librknnrt.so")
    target_link_libraries(edge_arm_control PRIVATE rknnrt)
endif()
