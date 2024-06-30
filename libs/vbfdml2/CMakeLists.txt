cmake_minimum_required(VERSION 3.15)

project(vbfdml2)

if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
endif()

find_package(CGAL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(glm REQUIRED)

if(APPLE)
    include_directories("/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(vbfdml2 src/vbfdml2_py.cpp)
target_link_libraries(vbfdml2 PRIVATE CGAL::CGAL OpenMP::OpenMP_CXX fmt::fmt)

install(TARGETS vbfdml2 DESTINATION vbfdml2)
