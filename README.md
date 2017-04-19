# Stereolabs ZED - Oculus

This sample shows how to capture stereo video from the ZED and display it in an Oculus Rift headset. Works with Oculus DK2 and CV1.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, check out our [tutorial](https://www.stereolabs.com/blog/index.php/2015/11/17/using-the-zed-with-oculus/) on using the ZED and Oculus Rift and read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).


### Prerequisites

- Windows 64 bits
- [ZED SDK 2.0](https://www.stereolabs.com/developers/) and its dependencies ([CUDA 8.0](https://developer.nvidia.com/cuda-downloads), [OpenCV 3.1](http://opencv.org/downloads.html))
- [Oculus SDK](https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/) (recommended 1.13)
- [GLEW 1.12](http://glew.sourceforge.net or in the dependencies ZED SDK folder)
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
It uses the CUDA-OpenGL interoperability to send images through opengl in the most effective way (without going back to CPU)



## Keyboard shortcuts

 Main Hotkeys                    |           Description                                       
 ------------------------------|-------------------------------------------------------------   
 'c'                   | Pause/Unpause the rendering                                                                                                       
  'q'                     | Quit the application.                                                       


