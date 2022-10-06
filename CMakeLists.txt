cmake_minimum_required(VERSION 3.14)
project(gc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.5.5 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(g2o REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/lib/cmake/g2o)
find_package(Sophus REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/sophus/install/share/sophus/cmake)
find_package(CSparse REQUIRED)

if(g2o_FOUND)
        set(g2o_INCLUDE_DIR /home/chl/codeTest/slamStudy/week1/Thirdparty/g2o/install/include)
        set(g2o_LIBRARIES /home/chl/codeTest/slamStudy/week1/Thirdparty/g2o/install/lib/libg2o_core.so)
        message("g2o include : " ${g2o_INCLUDE_DIR})
endif()

if(Sophus_FOUND)
        set(Sophus_INCLUDE_DIR /home/chl/codeTest/slamStudy/week1/Thirdparty/sophus/install/include)
        message("Sophus include : " ${Sophus_INCLUDE_DIR})
endif()

add_subdirectory(module)

add_executable(${PROJECT_NAME} examples/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
        module/include
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${g2o_INCLUDE_DIR}
        ${Sophus_INCLUDE_DIR}
        ${CSPARSE_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
        module
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${g2o_LIBRARIES}
        Sophus::Sophus
        ${CSPARSE_LIBRARY}
)
