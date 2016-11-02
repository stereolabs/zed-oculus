# Stereolabs ZED - Oculus

This sample shows how to capture stereo video from the ZED and display it in an Oculus Rift headset. Works with Oculus DK2 and CV1.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, check out our [tutorial](https://www.stereolabs.com/blog/index.php/2015/11/17/using-the-zed-with-oculus/) on using the ZED and Oculus Rift and read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).


### Prerequisites

- Windows 7 64 bits or later
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads), [OpenCV 3.1](http://opencv.org/downloads.html))
- [Oculus SDK 1.3.0](https://developer.oculus.com/downloads/pc/1.3.0/Oculus_SDK_for_Windows/) or later
- [GLEW](http://glew.sourceforge.net)
- [SDL](http://libsdl.org/download-2.0.php)


## Build the program

Download the sample and follow these instructions:

  - Create a folder called "build" in the source folder
  - Open cmake-gui and select the source and build folders
  - Generate the Visual Studio `Win64` solution
  - Open the resulting solution and change configuration to `Release`.
  You may have to modify the path of the dependencies to match your configuration
  - Build solution


## Features

This sample show how to render the ZED stereo images in a Rift.
- Two modes are available :
  - The first one, enabled by default, takes the ZED images from a CPU buffer and renders them into the Rift.
  - The second mode captures the images in a CUDA GPU buffer that is rendered directly into the Rift, using OpenGL/CUDA interoperability. This mode improves video framerate and latency.

## Keyboard shortcuts

 Main Hotkeys                    |           Description                                       
 ------------------------------|-------------------------------------------------------------   
 'c'                   | Pause/Unpause the rendering                                                                                                       
  'q'                     | Quit the application.                                                       


### OpenGL/CUDA interoperability

To get a significant performance boost, you can enable the OpenGL/CUDA interoperability. This will avoid costly GPU to CPU readback and unnecessary memory copies. However, the code is more complex if you are not familiar with CUDA.

 - Add the preprocessor definition OPENGL_GPU_INTEROP [here](https://github.com/stereolabs/zed-oculus/blob/master/CMakeLists.txt#L84).
