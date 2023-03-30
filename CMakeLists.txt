cmake_minimum_required(VERSION 3.5)

project(UNet_Studio VERSION 0.1 LANGUAGES CUDA CXX;C)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)


find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(ZLIB REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch)

set(PROJECT_SOURCES
        unet.cpp
        unet.hpp
        main.cpp
        train.hpp
        train.cpp
        evaluate.hpp
        evaluate.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

set(UNet_STUDIO_CUDA
        cuda.cu)

if(WIN32)
    set(UNET_STUDIO_EXEC
        WIN32)
endif()
if(APPLE)
    set(UNET_STUDIO_EXEC
        MACOSX_BUNDLE)
endif()

qt_add_executable(unet_studio MANUAL_FINALIZATION ${PROJECT_SOURCES} ${UNet_STUDIO_CUDA})

target_link_libraries(unet_studio PUBLIC Qt6::Widgets ZLIB::ZLIB ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_include_directories(unet_studio PUBLIC ${TIPL_DIR})
set_target_properties(unet_studio PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER unet_studio.labsolver.org
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

string(REPLACE "--Werror" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REPLACE "cross-execution-space-call" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
target_compile_definitions(unet_studio PUBLIC TIPL_USE_QT)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23.0")
    set_property(TARGET unet_studio PROPERTY CUDA_ARCHITECTURES all)
else()
    set_property(TARGET unet_studio PROPERTY CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86")
endif()

qt_finalize_executable(unet_studio)

