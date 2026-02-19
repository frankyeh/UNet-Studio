cmake_minimum_required(VERSION 3.5)

project(UNet_Studio VERSION 0.1 LANGUAGES CXX;C)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Core Gui Network Widgets REQUIRED)
find_package(ZLIB REQUIRED)
set(ZLIB_USE_STATIC_LIBS "ON")

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "Torch_DIR: ${Torch_DIR}")

find_package(Torch REQUIRED)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    project(UNet_Studio VERSION "2026.02.18" LANGUAGES CUDA CXX;C)
endif()
set(UNET_STUDIO_SOURCES
        unet.cpp
        unet.hpp
        main.cpp
        train.hpp
        train.cpp
        console.h
        console.cpp
        console.ui
        evaluate.hpp
        evaluate.cpp
        optiontablewidget.hpp
        optiontablewidget.cpp
        mainwindow.cpp
        mainwindow_training.cpp
        mainwindow_evaluation.cpp
        visual_perception_augmentation.cpp
        mainwindow.h
        mainwindow.ui
)

if(CUDAToolkit_FOUND)
    set(UNET_STUDIO_CUDA
            cuda.cu
            visual_perception_augmentation.cu
)
endif()

set(UNET_STUDIO_RESOURCES icons.qrc)

if(WIN32)
    set(UNET_STUDIO_EXEC
        WIN32
        unet_studio.rc)
endif()
if(APPLE)
    set(UNET_STUDIO_EXEC
        MACOSX_BUNDLE)     
endif()


add_executable(unet_studio ${UNET_STUDIO_EXEC} ${UNET_STUDIO_SOURCES} ${UNET_STUDIO_CUDA} ${UNET_STUDIO_RESOURCES})

if(WIN32)
    set_target_properties(unet_studio PROPERTIES
        WIN32_EXECUTABLE TRUE
        LINK_FLAGS "/ICON:unet-studio.ico")
endif()

if(APPLE)
    set_target_properties(unet_studio PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER unet-studio.labsolver.org
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        MACOSX_BUNDLE TRUE)
endif()


target_include_directories(unet_studio PUBLIC ${TIPL_DIR})
target_compile_definitions(unet_studio PUBLIC TIPL_USE_QT)

if(CUDAToolkit_FOUND)
    string(REPLACE "--Werror" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    string(REPLACE "cross-execution-space-call" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    target_compile_definitions(unet_studio PUBLIC TIPL_USE_CUDA)
    set_property(TARGET unet_studio PROPERTY CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86")
    target_link_libraries(unet_studio PUBLIC Qt6::Widgets Qt6::Core Qt6::Gui ZLIB::ZLIB ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
else()

    target_link_libraries(unet_studio PUBLIC Qt6::Widgets Qt6::Core Qt6::Gui ZLIB::ZLIB ${TORCH_LIBRARIES})

endif(CUDAToolkit_FOUND)


