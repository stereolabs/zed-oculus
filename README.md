# zed-oculus

**This sample is designed to work with the ZED stereo camera only and requires the ZED SDK. For more information: https://www.stereolabs.com**

It demonstrates how to grab stereo images with the ZED SDK and display the results in a Oculus Rift headset.

**Warning:**
 - GPU->CPU readback is time-consuming

**Prerequisites**
 - ZED SDK 0.9.2
 - [Oculus PC Runtime 0.8.0.0-Beta](https://developer.oculus.com/downloads/pc/0.8.0.0-beta/Oculus_Runtime_for_Windows)
 - [Oculus PC SDK 0.8.0.0-Beta](https://developer.oculus.com/downloads/pc/0.8.0.0-beta/Oculus_SDK_for_Windows)
 - OpenGL 3+
 - [GLEW 1.12.0](http://glew.sourceforge.net)
 - [SDL2 2.0.3](http://libsdl.org/download-2.0.php)
 
For more information on this sample, please read our tutorial [Using the ZED camera with Oculus Rift](https://www.stereolabs.com/blog/index.php/2015/11/17/516/).

##Keyboard shortcuts
 
               Main Hotkeys              |
---------------------------------------- |
 'q'   : Exit the application            |
 'c'   : Pause/unpause the rendering     |
 'r'   : Reset hit to zero               |
 'Mouse wheel'   : increase/decrease hit |
