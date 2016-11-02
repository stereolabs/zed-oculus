
SET(OCULUS_PATH "OCULUS_SDK_ROOT_DIR-NOTFOUND" REQUIRED CACHE STRING "Path of OVR SDK folder. eg: C:\\ovr_sdk_win_0.8.0.0\\OculusSDK")
SET(SDL_PATH "SDL2_ROOT_DIR-NOTFOUND" REQUIRED CACHE STRING "Path of SDL2 folder. eg: C:\\SDL2-2.0.3")

SET(execName ZED_Oculus_Viewer)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ".")

if(MSVC12)
    SET(VS_VERSION VS2013)
else(MSVC12)
    if(MSVC11)
        SET(VS_VERSION VS2012)
    else(MSVC11)
        if(MSVC10)
            SET(VS_VERSION VS2010)
        endif(MSVC10)
    endif(MSVC11)
endif(MSVC12)

SET(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
SET(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})

SET(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})
SET(OCULUS_LIBRARY_DIRS ${OCULUS_PATH}/LibOVR/Lib/Windows/x64/Release/${VS_VERSION})
SET(SDL_LIBRARY_DIRS ${SDL_PATH}/lib/x64)

SET(GLEW_LIBRARY_DIRS $ENV{GLEW_LIBRARY_DIRS})
SET(GLEW_LIBRARIES $ENV{GLEW_LIBRARY})
SET(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIRS})

SET(GLUT_INCLUDE_DIRS $ENV{GLUT_INCLUDE_DIRS})
SET(GLUT_LIBRARY_DIRS $ENV{GLUT_LIBRARY_DIRS})
SET(GLUT_LIBRARY $ENV{GLUT_LIBRARY})

SET(OCULUS_INCLUDE_DIRS ${OCULUS_PATH}/LibOVR/Include ${OCULUS_PATH}/LibOVRKernel/Src)
SET(OCULUS_LIBRARIES LibOVR)

SET(SDL_INCLUDE_DIRS ${SDL_PATH}/include)
SET(SDL_LIBRARIES SDL2 SDL2main)


# Mandatory
find_package(CUDA 8.0 REQUIRED)
find_package(OpenGL 3.0 REQUIRED)
find_package(OpenCV 3.1 REQUIRED)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OCULUS_INCLUDE_DIRS})
include_directories(${SDL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories($ENV{EIGEN_DIR})


link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${SDL_LIBRARY_DIRS})
link_directories(${OCULUS_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED Oculus Viewer")
add_definitions(-std=c++0x)

#add_definitions(-DOPENGL_GPU_INTEROP)

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
			${GLEW_LIBRARIES}
			${OpenCV_LIBRARIES}
			${SDL_LIBRARIES}
			${OCULUS_LIBRARIES}
			${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_npp_LIBRARY}
 )
