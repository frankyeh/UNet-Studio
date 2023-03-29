cmake_minimum_required(VERSION 3.5)

project(UNetStudio VERSION 0.1 LANGUAGES CUDA CXX;C)

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

set(UNetStudio_CUDA
        cuda.cu)

qt_add_executable(UNetStudio MANUAL_FINALIZATION ${PROJECT_SOURCES} ${UNetStudio_CUDA})

target_link_libraries(UNetStudio PUBLIC Qt6::Widgets ZLIB::ZLIB ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_include_directories(UNetStudio PUBLIC ${TIPL_DIR})
set_target_properties(UNetStudio PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER UNetStudio.labsolver.org
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

string(REPLACE "--Werror" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REPLACE "cross-execution-space-call" "" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
target_compile_definitions(UNetStudio PUBLIC TIPL_USE_QT)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23.0")
    set_property(TARGET UNetStudio PROPERTY CUDA_ARCHITECTURES all)
else()
    set_property(TARGET UNetStudio PROPERTY CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86")
endif()


qt_finalize_executable(UNetStudio)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(TARGET UNetStudio POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:UNetStudio>)

