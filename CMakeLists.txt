cmake_minimum_required(VERSION 3.11-3.25)

# Metadata
project(
    VBFDML2
    VERSION 1.0
    DESCRIPTION "New version of (voxel-based approximation of) few distance-measurement localization, with support for uncertain environments."
)
set(VBFDML_PROJECT_AUTHORS "Michael Bilevich")
set(VBFDML_PROJECT_EMAILS "michaelmoshe@mail.tau.ac.il")
set(VBFDML_PROJECT_URL "http://acg.cs.tau.ac.il/projects/in-house-projects/localization-with-few-distance-measurements")
####

option(VBFDML_WITH_PYBINDINGS "With Python bindings" ON)


set(VBFDML_MODULES_REL_DIR cmake/modules)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/${VBFDML_MODULES_REL_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(libs)
add_subdirectory(apps)