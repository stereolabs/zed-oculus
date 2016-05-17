# zed-oculus

_Note: This sample works with Oculus DK2 and CV1._

**This sample is designed to work with the ZED stereo camera only and requires the ZED SDK. For more information: https://www.stereolabs.com**

It demonstrates how to grab stereo images with the ZED SDK and display the results in a Oculus Rift headset.

**Prerequisites**
 - ZED SDK 0.9.2 or later
 - [Oculus PC Runtime 1.3.0](https://www.oculus.com/en-us/setup/) (This URL may change in near future)
 - [Oculus PC SDK 1.3.0](https://developer.oculus.com/downloads/pc/1.3.0/Oculus_SDK_for_Windows/) 
 - OpenGL 3+
 - [GLEW 1.12.0](http://glew.sourceforge.net)
 - [SDL2 2.0.3](http://libsdl.org/download-2.0.php)

**OpenGL GPU Interoperability**

/!\ You must have NVIDIA GPU Computing Toolkit installed on your computer.

To enable Opengl GPU Interoperability (which increases performance):
 - add the preprocessor definition OPENGL_GPU_INTEROP;
 - link the CUDA library cudart_static.lib (commonly at "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64\cudart_static.lib").

##Keyboard shortcuts
 
               Main Hotkeys              |
---------------------------------------- |
 'q'   : Exit the application            |
 'c'   : Pause/Unpause the rendering     |
 'r'   : Reset HIT to zero               |
 'Mouse wheel'   : Increase/Decrease HIT |
