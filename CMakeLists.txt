SET(OCULUS_PATH "OCULUS_SDK_ROOT_DIR-NOTFOUND" REQUIRED CACHE STRING "Path of OVR SDK folder. eg: C:\\ovr_sdk_win_0.17.0\\OculusSDK")
SET(SDL_PATH "SDL2_ROOT_DIR-NOTFOUND" REQUIRED CACHE STRING "Path of SDL2 folder. eg: C:\\SDL2-2.0.3")

SET(execName ZED_Stereo_Passthrough)
project(${execName})
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

# Mandatory
find_package(ZED 2.2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(cuda REQUIRED)
find_package(OpenGL 3.0 REQUIRED)

SET(OCULUS_LIBRARY_DIRS ${OCULUS_PATH}/LibOVR/Lib/Windows/x64/Release/VS2015)
SET(SDL_LIBRARY_DIRS ${SDL_PATH}/lib/x64)

SET(OCULUS_INCLUDE_DIRS ${OCULUS_PATH}/LibOVR/Include ${OCULUS_PATH}/LibOVRKernel/Src)
SET(OCULUS_LIBRARIES LibOVR)

SET(SDL_INCLUDE_DIRS ${SDL_PATH}/include)
SET(SDL_LIBRARIES SDL2 SDL2main)

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OCULUS_INCLUDE_DIRS})
include_directories(${SDL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${SDL_LIBRARY_DIRS})
link_directories(${OCULUS_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

SET(SRC_FOLDER src)
FILE(GLOB_RECURSE SRC_FILES "${SRC_FOLDER}/*.cpp")

ADD_EXECUTABLE(${execName} ${SRC_FILES})

add_definitions(-std=c++11 -O3)
set_property(TARGET ${execName} PROPERTY OUTPUT_NAME "ZED Stereo Passthrough")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")

# Add the required libraries for linking:
TARGET_LINK_LIBRARIES(${execName}
			${ZED_LIBRARIES}
			${OPENGL_LIBRARIES}
            ${GLUT_LIBRARY}
			${GLEW_LIBRARIES}
			${SDL_LIBRARIES}
			${OCULUS_LIBRARIES}
			${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED})
