///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2016, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/**********************************
 ** Using the ZED with Oculus Rift
 **********************************/

#include <iostream>
#include <Windows.h>

#include <GL/glew.h>

#include <stddef.h>

#include <SDL.h>
#include <SDL_syswm.h>

#include <Extras/OVR_Math.h>
#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <sl/Camera.hpp>

#include "Shader.hpp"

#define MAX_FPS 120

GLchar* OVR_ZED_VS =
        "#version 330 core\n \
			layout(location=0) in vec3 in_vertex;\n \
			layout(location=1) in vec2 in_texCoord;\n \
			uniform float hit; \n \
			uniform uint isLeft; \n \
			out vec2 b_coordTexture; \n \
			void main()\n \
			{\n \
				if (isLeft == 1U)\n \
				{\n \
					b_coordTexture = in_texCoord;\n \
					gl_Position = vec4(in_vertex.x - hit, in_vertex.y, in_vertex.z,1);\n \
				}\n \
				else \n \
				{\n \
					b_coordTexture = vec2(1.0 - in_texCoord.x, in_texCoord.y);\n \
					gl_Position = vec4(-in_vertex.x + hit, in_vertex.y, in_vertex.z,1);\n \
				}\n \
			}";

GLchar* OVR_ZED_FS =
        "#version 330 core\n \
			uniform sampler2D u_textureZED; \n \
			in vec2 b_coordTexture;\n \
			out vec4 out_color; \n \
			void main()\n \
			{\n \
				out_color = vec4(texture(u_textureZED, b_coordTexture).bgr,1); \n \
			}";



//objects
sl::Camera zed;
sl::Mat zed_image_Left;
sl::Mat zed_image_Right;



int main(int argc, char **argv) {
    // Initialize SDL2's context
    SDL_Init(SDL_INIT_VIDEO);
    // Initialize Oculus' context
    ovrResult result = ovr_Initialize(nullptr);
    if (OVR_FAILURE(result)) {
        std::cout << "ERROR: Failed to initialize libOVR" << std::endl;
        SDL_Quit();
        return -1;
    }

    ovrSession session;
    ovrGraphicsLuid luid;
    // Connect to the Oculus headset
    result = ovr_Create(&session, &luid);
    if (OVR_FAILURE(result)) {
        std::cout << "ERROR: Oculus Rift not detected" << std::endl;
        ovr_Shutdown();
        SDL_Quit();
        return -1;
    }

    int x = SDL_WINDOWPOS_CENTERED, y = SDL_WINDOWPOS_CENTERED;
    int winWidth = 1280;
    int winHeight = 720;
    Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
    // Create SDL2 Window
    SDL_Window* window = SDL_CreateWindow("OVR ZED App", x, y, winWidth, winHeight, flags);
    // Create OpenGL context
    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    // Initialize GLEW
    glewInit();
    // Turn off vsync to let the compositor do its magic
    SDL_GL_SetSwapInterval(0);

    // Initialize the ZED Camera
	sl::InitParameters init_parameters;
	init_parameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;


	sl::ERROR_CODE err_ = zed.open(init_parameters);
 

    if (err_ != sl::SUCCESS) {
        std::cout << "ERROR: " << sl::errorCode2str(err_) << std::endl;
        ovr_Destroy(session);
        ovr_Shutdown();
        SDL_GL_DeleteContext(glContext);
        SDL_DestroyWindow(window);
        SDL_Quit();
		zed.close();
        return -1;
    }

	int zedWidth = zed.getResolution().width;
	int zedHeight = zed.getResolution().height;

    GLuint zedTextureID_L, zedTextureID_R;
    // Generate OpenGL texture for left images of the ZED camera
    glGenTextures(1, &zedTextureID_L);
    glBindTexture(GL_TEXTURE_2D, zedTextureID_L);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, zedWidth, zedHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Generate OpenGL texture for right images of the ZED camera
    glGenTextures(1, &zedTextureID_R);
    glBindTexture(GL_TEXTURE_2D, zedTextureID_R);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, zedWidth, zedHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsResource* cimg_L;
    cudaGraphicsResource* cimg_R;
    cudaError_t errL, errR;
    errL = cudaGraphicsGLRegisterImage(&cimg_L, zedTextureID_L, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
    errR = cudaGraphicsGLRegisterImage(&cimg_R, zedTextureID_R, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
    if (errL != cudaSuccess || errR != cudaSuccess) {
        std::cout << "ERROR: cannot create CUDA texture : " << errL << "|" << errR << std::endl;
    }

    ovrHmdDesc hmdDesc = ovr_GetHmdDesc(session);
    // Get the texture sizes of Oculus eyes
    ovrSizei textureSize0 = ovr_GetFovTextureSize(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0], 1.0f);
    ovrSizei textureSize1 = ovr_GetFovTextureSize(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1], 1.0f);
    // Compute the final size of the render buffer
    ovrSizei bufferSize;
    bufferSize.w = textureSize0.w + textureSize1.w;
    bufferSize.h = std::max(textureSize0.h, textureSize1.h);
    // Initialize OpenGL swap textures to render
    ovrTextureSwapChain textureChain = nullptr;
    // Description of the swap chain
    ovrTextureSwapChainDesc descTextureSwap = {};
    descTextureSwap.Type = ovrTexture_2D;
    descTextureSwap.ArraySize = 1;
    descTextureSwap.Width = bufferSize.w;
    descTextureSwap.Height = bufferSize.h;
    descTextureSwap.MipLevels = 1;
    descTextureSwap.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
    descTextureSwap.SampleCount = 1;
    descTextureSwap.StaticImage = ovrFalse;
    // Create the OpenGL texture swap chain
    result = ovr_CreateTextureSwapChainGL(session, &descTextureSwap, &textureChain);

    int length = 0;
    ovr_GetTextureSwapChainLength(session, textureChain, &length);

    if (OVR_SUCCESS(result)) {
        for (int i = 0; i < length; ++i) {
            GLuint chainTexId;
            ovr_GetTextureSwapChainBufferGL(session, textureChain, i, &chainTexId);
            glBindTexture(GL_TEXTURE_2D, chainTexId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    } else {
        std::cout << "ERROR: failed creating swap texture" << std::endl;
        ovr_Destroy(session);
        ovr_Shutdown();
        SDL_GL_DeleteContext(glContext);
        SDL_DestroyWindow(window);
        SDL_Quit();
		zed.close();
        return -1;
    }
    // Generate frame buffer to render
    GLuint fboID;
    glGenFramebuffers(1, &fboID);
    // Generate depth buffer of the frame buffer
    GLuint depthBuffID;
    glGenTextures(1, &depthBuffID);
    glBindTexture(GL_TEXTURE_2D, depthBuffID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GLenum internalFormat = GL_DEPTH_COMPONENT24;
    GLenum type = GL_UNSIGNED_INT;
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, bufferSize.w, bufferSize.h, 0, GL_DEPTH_COMPONENT, type, NULL);

    // Create a mirror texture to display the render result in the SDL2 window
    ovrMirrorTextureDesc descMirrorTexture;
    memset(&descMirrorTexture, 0, sizeof (descMirrorTexture));
    descMirrorTexture.Width = winWidth;
    descMirrorTexture.Height = winHeight;
    descMirrorTexture.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;

    ovrMirrorTexture mirrorTexture = nullptr;
    result = ovr_CreateMirrorTextureGL(session, &descMirrorTexture, &mirrorTexture);
    if (!OVR_SUCCESS(result)) {
        std::cout << "ERROR: Failed to create mirror texture" << std::endl;
    }
    GLuint mirrorTextureId;
    ovr_GetMirrorTextureBufferGL(session, mirrorTexture, &mirrorTextureId);

    GLuint mirrorFBOID;
    glGenFramebuffers(1, &mirrorFBOID);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBOID);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTextureId, 0);
    glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    // Frame index used by the compositor
    // it needs to be updated each new frame
    long long frameIndex = 0;

    // FloorLevel will give tracking poses where the floor height is 0
    ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);

    // Initialize a default Pose
    ovrPosef eyeRenderPose[2];

    // Get the render description of the left and right "eyes" of the Oculus headset
    ovrEyeRenderDesc eyeRenderDesc[2];
    eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
    eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);
    // Get the Oculus view scale description
    ovrVector3f hmdToEyeOffset[2];
    double sensorSampleTime;

    // Create and compile the shader's sources
    Shader shader(OVR_ZED_VS, OVR_ZED_FS);

    // Get the ZED image field of view with the ZED parameters
	float zedFovH = zed.getCameraInformation().calibration_parameters.left_cam.h_fov * M_PI /180.f;
    // Compute the Horizontal Oculus' field of view with its parameters
    float ovrFovH = (atanf(hmdDesc.DefaultEyeFov[0].LeftTan) + atanf(hmdDesc.DefaultEyeFov[0].RightTan));

	std::cout << " FOV comparison : " << zedFovH << " : " << ovrFovH << std::endl;
    // Compute the useful part of the ZED image
	
	// compute the usefulwidth seems to be more coherent, but images look too zoomed in. You can switch between both with to check the difference.
	//unsigned int usefulWidth = zedWidth *ovrFovH / zedFovH;
	unsigned int usefulWidth = zedWidth;

    // Compute the size of the final image displayed in the headset with the ZED image's aspect-ratio kept
    unsigned int widthFinal = bufferSize.w / 2;
    float heightGL = 1.f;
    float widthGL = 1.f;
    if (usefulWidth > 0.f) {
        unsigned int heightFinal = zedHeight * widthFinal / usefulWidth;
        // Convert this size to OpenGL viewport's frame's coordinates
        heightGL = (heightFinal) / (float) (bufferSize.h);
        widthGL = ((zedWidth * (heightFinal / (float)zedHeight)) / (float) widthFinal);
    } else {
        std::cout << "WARNING: ZED parameters got wrong values."
                "Default vertical and horizontal FOV are used.\n"
                "Check your calibration file or check if your ZED is not too close to a surface or an object."
                << std::endl;
    }

    // Compute the Vertical Oculus' field of view with its parameters
    float ovrFovV = (atanf(hmdDesc.DefaultEyeFov[0].UpTan) + atanf(hmdDesc.DefaultEyeFov[0].DownTan));

    // Compute the center of the optical lenses of the headset
    float offsetLensCenterX = ((atanf(hmdDesc.DefaultEyeFov[0].LeftTan)) / ovrFovH) * 2.f - 1.f;
    float offsetLensCenterY = ((atanf(hmdDesc.DefaultEyeFov[0].UpTan)) / ovrFovV) * 2.f - 1.f;


    // Create a rectangle with the computed coordinates and push it in GPU memory.

    struct GLScreenCoordinates {
        float left, up, right, down;
    } screenCoord;
    screenCoord.up = heightGL + offsetLensCenterY;
    screenCoord.down = heightGL - offsetLensCenterY;
    screenCoord.right = widthGL + offsetLensCenterX;
    screenCoord.left = widthGL - offsetLensCenterX;

    float rectVertices[12] = {-screenCoord.left, -screenCoord.up, 0,
        screenCoord.right, -screenCoord.up, 0,
        screenCoord.right, screenCoord.down, 0,
        -screenCoord.left, screenCoord.down, 0};
    GLuint rectVBO[3];
    glGenBuffers(1, &rectVBO[0]);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof (rectVertices), rectVertices, GL_STATIC_DRAW);

    float rectTexCoord[8] = {0, 1, 1, 1, 1, 0, 0, 0};
    glGenBuffers(1, &rectVBO[1]);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof (rectTexCoord), rectTexCoord, GL_STATIC_DRAW);

    unsigned int rectIndices[6] = {0, 1, 2, 0, 2, 3};
    glGenBuffers(1, &rectVBO[2]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rectVBO[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof (rectIndices), rectIndices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Initialize hit value
    float hit = 0.0f;
    // Initialize a boolean that will be used to stop the application�s loop and another one to pause/unpause rendering
    bool end = false;
    bool refresh = true;
    // SDL variable that will be used to store input events
    SDL_Event events;
    // Initialize time variables. They will be used to limit the number of frames rendered per second.
    // Frame counter
    unsigned int riftc = 0, zedc = 1;
    // Chronometer
    unsigned int rifttime = 0, zedtime = 0, zedFPS = 0;
    int time1 = 0, timePerFrame = 0;
    int frameRate = (int) (1000 / MAX_FPS);

    // This boolean is used to test if the application is focused
    bool isVisible = true;

    // Enable the shader
    glUseProgram(shader.getProgramId());
    // Bind the Vertex Buffer Objects of the rectangle that displays ZED images
    // vertices
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO[0]);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rectVBO[2]);
    // texture coordinates
    glEnableVertexAttribArray(Shader::ATTRIB_TEXTURE2D_POS);
    glBindBuffer(GL_ARRAY_BUFFER, rectVBO[1]);
    glVertexAttribPointer(Shader::ATTRIB_TEXTURE2D_POS, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // Main loop
    while (!end) {
        // Compute the time used to render the previous frame
        timePerFrame = SDL_GetTicks() - time1;
        // If the previous frame has been rendered too fast
        if (timePerFrame < frameRate) {
            // Pause the loop to have a max FPS equal to MAX_FPS
            SDL_Delay(frameRate - timePerFrame);
            timePerFrame = frameRate;
        }
        // Increment the ZED chronometer
        zedtime += timePerFrame;
        // If ZED chronometer reached 1 second
        if (zedtime > 1000) {
            zedFPS = zedc;
            zedc = 0;
            zedtime = 0;
        }
        // Increment the Rift chronometer and the Rift frame counter
        rifttime += timePerFrame;
        riftc++;
        // If Rift chronometer reached 200 milliseconds
        if (rifttime > 200) {
            // Display FPS
            std::cout << "\rRIFT FPS: " << 1000 / (rifttime / riftc) << " | ZED FPS: " << zedFPS;
            // Reset Rift chronometer
            rifttime = 0;
            // Reset Rift frame counter
            riftc = 0;
        }
        // Start frame chronometer
        time1 = SDL_GetTicks();

        // While there is an event catched and not tested
        while (SDL_PollEvent(&events)) {
            // If a key is released
            if (events.type == SDL_KEYUP) {
                // If Q quit the application
                if (events.key.keysym.scancode == SDL_SCANCODE_Q)
                    end = true;
            }
        }

        // Get texture swap index where we must draw our frame
        GLuint curTexId;
        int curIndex;
        ovr_GetTextureSwapChainCurrentIndex(session, textureChain, &curIndex);
        ovr_GetTextureSwapChainBufferGL(session, textureChain, curIndex, &curTexId);

        // Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyeOffset) may change at runtime.
        eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
        eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);
        hmdToEyeOffset[0] = eyeRenderDesc[0].HmdToEyeOffset;
        hmdToEyeOffset[1] = eyeRenderDesc[1].HmdToEyeOffset;
        // Get eye poses, feeding in correct IPD offset
        ovr_GetEyePoses(session, frameIndex, ovrTrue, hmdToEyeOffset, eyeRenderPose, &sensorSampleTime);

		sl::RuntimeParameters runtime_parameters;
		runtime_parameters.enable_depth = false;

        // If the application is focused
        if (isVisible) {
            // If successful grab a new ZED image
            if (zed.grab(runtime_parameters)==sl::SUCCESS) {
                // Update the ZED frame counter
                zedc++;
                if (refresh) {

					cudaArray_t arrIm;
					if (zed.retrieveImage(zed_image_Left, sl::VIEW_LEFT, sl::MEM_GPU) == sl::SUCCESS) {
						cudaGraphicsMapResources(1, &cimg_L, 0);
						cudaGraphicsSubResourceGetMappedArray(&arrIm, cimg_L, 0, 0);
						cudaMemcpy2DToArray(arrIm, 0, 0, zed_image_Left.getPtr<sl::uchar1>(sl::MEM_GPU), zed_image_Left.getStepBytes(sl::MEM_GPU), zed_image_Left.getWidth() * 4, zed_image_Left.getHeight(), cudaMemcpyDeviceToDevice);
						cudaGraphicsUnmapResources(1, &cimg_L, 0);
					}


					if (zed.retrieveImage(zed_image_Right, sl::VIEW_RIGHT, sl::MEM_GPU) == sl::SUCCESS) {
						cudaGraphicsMapResources(1, &cimg_R, 0);
						cudaGraphicsSubResourceGetMappedArray(&arrIm, cimg_R, 0, 0);
						cudaMemcpy2DToArray(arrIm, 0, 0, zed_image_Right.getPtr<sl::uchar1>(sl::MEM_GPU), zed_image_Right.getStepBytes(sl::MEM_GPU), zed_image_Right.getWidth() * 4, zed_image_Right.getHeight(), cudaMemcpyDeviceToDevice);
						cudaGraphicsUnmapResources(1, &cimg_R, 0);
					}

 
                    // Bind the frame buffer
                    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
                    // Set its color layer 0 as the current swap texture
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
                    // Set its depth layer as our depth buffer
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffID, 0);
                    // Clear the frame buffer
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                    glClearColor(0, 0, 0, 1);

                    // Render for each Oculus eye the equivalent ZED image
                    for (int eye = 0; eye < 2; eye++) {
                        // Set the left or right vertical half of the buffer as the viewport
                        glViewport(eye == ovrEye_Left ? 0 : bufferSize.w / 2, 0, bufferSize.w / 2, bufferSize.h);
                        // Bind the left or right ZED image
                        glBindTexture(GL_TEXTURE_2D, eye == ovrEye_Left ? zedTextureID_L : zedTextureID_R);

                        // Bind the hit value
                        glUniform1f(glGetUniformLocation(shader.getProgramId(), "hit"), eye == ovrEye_Left ? hit : -hit);
                        // Bind the isLeft value
                        glUniform1ui(glGetUniformLocation(shader.getProgramId(), "isLeft"), eye == ovrEye_Left ? 1U : 0U);
                        // Draw the ZED image
                        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
                    }

                    // Avoids an error when calling SetAndClearRenderSurface during next iteration.
                    // Without this, during the next while loop iteration SetAndClearRenderSurface
                    // would bind a framebuffer with an invalid COLOR_ATTACHMENT0 because the texture ID
                    // associated with COLOR_ATTACHMENT0 had been unlocked by calling wglDXUnlockObjectsNV.
                    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
                    // Commit changes to the textures so they get picked up frame
                    ovr_CommitTextureSwapChain(session, textureChain);
                }

                // Do not forget to increment the frameIndex!
                frameIndex++;
            }
        }
        /*
        Note: Even if we don't ask to refresh the framebuffer or if the Camera::grab()
              doesn't catch a new frame, we have to submit an image to the Rift; it
                  needs 75Hz refresh. Else there will be jumbs, black frames and/or glitches
                  in the headset.
         */

        ovrLayerEyeFov ld;
        ld.Header.Type = ovrLayerType_EyeFov;
        // Tell to the Oculus compositor that our texture origin is at the bottom left
        ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft; // Because OpenGL | Disable head tracking
        // Set the Oculus layer eye field of view for each view
        for (int eye = 0; eye < 2; ++eye) {
            // Set the color texture as the current swap texture
            ld.ColorTexture[eye] = textureChain;
            // Set the viewport as the right or left vertical half part of the color texture
            ld.Viewport[eye] = OVR::Recti(eye == ovrEye_Left ? 0 : bufferSize.w / 2, 0, bufferSize.w / 2, bufferSize.h);
            // Set the field of view
            ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
            // Set the pose matrix
            ld.RenderPose[eye] = eyeRenderPose[eye];
        }

        ld.SensorSampleTime = sensorSampleTime;

        ovrLayerHeader* layers = &ld.Header;
        // Submit the frame to the Oculus compositor
        // which will display the frame in the Oculus headset
        result = ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1);

        if (!OVR_SUCCESS(result)) {
            std::cout << "ERROR: failed to submit frame" << std::endl;
            glDeleteBuffers(3, rectVBO);
            ovr_DestroyTextureSwapChain(session, textureChain);
            ovr_DestroyMirrorTexture(session, mirrorTexture);
            ovr_Destroy(session);
            ovr_Shutdown();
            SDL_GL_DeleteContext(glContext);
            SDL_DestroyWindow(window);
            SDL_Quit();
			zed.close();
            return -1;
        }

        if (result == ovrSuccess && !isVisible) {
            std::cout << "The application is now shown in the headset." << std::endl;
        }
        isVisible = (result == ovrSuccess);

        // This is not really needed for this application but it may be useful for an more advanced application
        ovrSessionStatus sessionStatus;
        ovr_GetSessionStatus(session, &sessionStatus);
        if (sessionStatus.ShouldRecenter) {
            std::cout << "Recenter Tracking asked by Session" << std::endl;
            ovr_RecenterTrackingOrigin(session);
        }

        // Copy the frame to the mirror buffer
        // which will be drawn in the SDL2 image
        glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBOID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        GLint w = winWidth;
        GLint h = winHeight;
        glBlitFramebuffer(0, h, w, 0,
                0, 0, w, h,
                GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        // Swap the SDL2 window
        SDL_GL_SwapWindow(window);
    }

    // Disable all OpenGL buffer
    glDisableVertexAttribArray(Shader::ATTRIB_TEXTURE2D_POS);
    glDisableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    // Delete the Vertex Buffer Objects of the rectangle
    glDeleteBuffers(3, rectVBO);
    // Delete SDL, OpenGL, Oculus and ZED context
    ovr_DestroyTextureSwapChain(session, textureChain);
    ovr_DestroyMirrorTexture(session, mirrorTexture);
    ovr_Destroy(session);
    ovr_Shutdown();
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
	zed.close();
    // Quit
    return 0;
}
