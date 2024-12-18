cmake_minimum_required(VERSION 3.0.2)

project(laneDet LANGUAGES CXX)

add_definitions(-std=c++14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_BUILD_TYPE Release)


message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/lib/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/bin/")

option(MNN "MNN deploy" ON)

include_directories(/usr/include/opencv4)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


if (TENSORRT)
    
endif()

if (MNN)
    include_directories(~/envs/MNN/install/include)
    link_directories(~/envs/MNN/install/lib)

    add_library(laneDetMNN SHARED 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/implement/mnn.cpp
    )
    target_link_libraries(laneDetMNN PUBLIC
        MNN
        ${OpenCV_LIBRARIES}
        yaml-cpp
    )

    add_executable(laneDetDemo
        ${CMAKE_CURRENT_SOURCE_DIR}/src/demo/demo.cpp
    )

    target_link_libraries(laneDetDemo
        laneDetMNN
        ${OpenCV_LIBRARIES}
        yaml-cpp
    )


endif()

if(ROCKCHIP)

endif()

if (ASCEND)
    
endif()