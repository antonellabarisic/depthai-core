#!/usr/bin/env python3

# this is the github clean version. version before it was cleaned for github is saved to file stereo_astra_aligned_nn_with_comments.

# combined from stereo_astra_aligned_nn.py (that is a result of aligning added to stereo_astra.py) and depth_nn_stereo_aligned.py!

import cv2
import depthai as dai
import numpy as np
from transformers import pipeline
import torch
from get_calib_info import get_focal_length
from PIL import Image
import copy
import time
from datetime import timedelta
FPS = 25.0 # otherwise very slow and laggy

class FPSCounter:
    def __init__(self):
        self.frameTimes = []

    def tick(self):
        now = time.time()
        self.frameTimes.append(now)
        self.frameTimes = self.frameTimes[-10:]

    def getFps(self):
        if len(self.frameTimes) <= 1:
            return 0
        return (len(self.frameTimes) - 1) / (self.frameTimes[-1] - self.frameTimes[0])

class CombinedDepthApp:
    def __init__(self):
        self.lowerRes = (640, 400)
        # print focal length of camera currently in use
        self.focal_px = get_focal_length(dai.CameraBoardSocket.CAM_B, resolution=self.lowerRes, verbose=True)

        self.pipeline = dai.Pipeline()
        # set up nn pipeline
        # device = 0 if torch.cuda.is_available() else -1
        # self.nn_pipeline = pipeline(device=device)
        self.maxDisparity = 1 # isn't used
        self.colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

        # Stereo parameters (should match your camera calibration)
        self.baseline_mm = 75  # baseline in mm
        # OAK D PRO W
        # self.focal_px = 568.924
        # OAK D PRO
        # self.focal_px = 798.6192626953125 # focal length in pixels
        # OAK D PRO W at low res (640, 400)
        # self.focal_px = 284.4622802734375
        # OAK D PRO at low res (640, 400)
        # self.focal_px = 399.30963134765625

        self.subpixel_bits = 3    # subpixel bits (FAST_ACCURACY and FAST_DENSITY dont use subpixel, 5 for face and high detail, and 3 for default and robotics)

        # Depth range parameters (in mm)
        self.depth_min = 1000 # 0
        self.depth_max = 4000 # 15000

        # === RGB / Monocular NN setup ===
        self.nn_depth_min = 1.0 # 0.0 # doesn't work when set to self.depth_min_stereo
        self.nn_depth_max = 4.0 # 15.0 # 10.0 # doesn't work when set to self.depth_max_stereo
        device = 0 if torch.cuda.is_available() else -1
        self.nn_depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            device=device
        )
        self.latest_nn_depth = None
        
        self.setup_stereo_pipeline()
        self.latest_stereo_depth = None  # Store latest real (stereo) depth frame

        # Visualization flags
        self.show_left = False
        self.show_right = False
        self.show_disparity = False
        self.show_confidence = False # True
        self.show_stereo_depth = True
        self.show_nn_depth = True
        self.show_aligned = True

        self.use_lower_res = False

    def setup_stereo_pipeline(self):
        monoLeft = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        monoRight = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        # add alignment node
        rgbCam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        sync = self.pipeline.create(dai.node.Sync)
        # stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1/(2*FPS))) # important for FPS 
        # platform = self.pipeline.getDefaultDevice().getPlatform()
        # print("platform:", platform) # platform: Platform.RVC2
        # if platform == dai.Platform.RVC4:
        #     align = self.pipeline.create(dai.node.ImageAlign)

        # StereoDepthConfig initialization
        # config = dai.StereoDepthConfig()
        # config.costMatching.confidenceThreshold = 200
        # stereo.initialConfig.set(config)

        # OR SELECT ONE OF THE PRESETS
        # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

        # CONFIG
        stereo.initialConfig.setConfidenceThreshold(55) # 255 is highest confidence, 0 is lowest
        stereo.setSubpixel(True)
        stereo.initialConfig.setSubpixelFractionalBits(self.subpixel_bits)
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)

        # Set depth range in threshold filter
        stereo.initialConfig.postProcessing.thresholdFilter.minRange = self.depth_min
        stereo.initialConfig.postProcessing.thresholdFilter.maxRange = self.depth_max
        print("depth set to:", stereo.initialConfig.postProcessing.thresholdFilter.minRange, stereo.initialConfig.postProcessing.thresholdFilter.maxRange)

        # monoLeftOut = monoLeft.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        # monoRightOut = monoRight.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        # above is too heavy, opting for lower resolution
        monoLeftOut = monoLeft.requestOutput(size=self.lowerRes, fps=FPS)
        monoRightOut = monoRight.requestOutput(size=self.lowerRes, fps=FPS)
        monoLeftOut.link(stereo.left)
        monoRightOut.link(stereo.right)
        # link alignment node
        # rgbOut = rgbCam.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12) # is this the right type for RGB? didn't find explanation in https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_frame
        # above version (stereo_astra) results in error (?) so setting to the one from depth_alignment_v3
        rgbOut = rgbCam.requestOutput(size = (1280, 960), fps = FPS, enableUndistortion=True)
        rgbOut.link(sync.inputs["rgb"])
        # if platform == dai.Platform.RVC4:
        #     stereo.depth.link(align.input)
        #     rgbOut.link(align.inputAlignTo)
        #     align.outputAligned.link(sync.inputs["depth_aligned"])
        # else:
        #     stereo.depth.link(sync.inputs["depth_aligned"])
        #     rgbOut.link(stereo.inputAlignTo)

        stereo.depth.link(sync.inputs["depth_aligned"])
        rgbOut.link(stereo.inputAlignTo)

        self.syncedLeftQueue = stereo.syncedLeft.createOutputQueue()
        self.syncedRightQueue = stereo.syncedRight.createOutputQueue()
        self.disparityQueue = stereo.disparity.createOutputQueue(blocking=False) # attempt to reduce load for better fps
        self.confidenceQueue = stereo.confidenceMap.createOutputQueue(blocking=False)
        self.depthQueue = stereo.depth.createOutputQueue()
        # define alignment queue
        self.syncAlignedQueue = sync.out.createOutputQueue()

    # === Stereo visualization functions ===
    def setup_rgb_pipeline(self):
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 480) # check if this alone cuts the image
        camRgb.setInterleaved(False)
        self.rgbQueue = camRgb.preview.createOutputQueue()

    def process_disparity(self, npDisparity):
        # Visualization scaling:
        # Standard mode: max_disp = 95
        # Extended disparity: max_disp = 190
        # Subpixel mode: max_disp = 760 (3 bits), 1520 (4 bits), 3040 (5 bits)
        # To use fixed scaling, code it
        # By default, use dynamic scaling for visualization:
        normDisparity = cv2.normalize(npDisparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colorizedDisparity = cv2.applyColorMap(normDisparity, self.colorMap)
        return colorizedDisparity

    def process_confidence(self, npConfidence):
        # Fixed mapping: 0 = black, 255 = white
        normConfidence = np.clip(npConfidence, 0, 255).astype(np.uint8)
        colorizedConfidence = cv2.applyColorMap(normConfidence, cv2.COLORMAP_JET)
        return colorizedConfidence

    def process_stereo_depth(self, npDepth):
        # Visualization scaling based on depth range
        # normDepth = np.clip((npDepth - self.depth_min) / max(1, (self.depth_max - self.depth_min)) * 255, 0, 255).astype(np.uint8)
        # colorizedDepth = cv2.applyColorMap(normDepth, cv2.COLORMAP_HOT)
        # return colorizedDepth
        # checking if thresholdFilter will do the clipping on its own
        # return cv2.applyColorMap(npDepth.astype(np.uint8), cv2.COLORMAP_HOT) # not normalized
        normDepth = ((npDepth - npDepth.min()) / max(1, (npDepth.max() - npDepth.min())) * 255).astype(np.uint8)
        colorizedDepth = cv2.applyColorMap(normDepth, cv2.COLORMAP_HOT)
        return colorizedDepth
        # conclusion: the threshold (displayed in depth image) is handled by the postProcessing.thresholdFilter - no need to do it ourselves.

    def calculate_stereo_depth(self, npDisparity):
        # Use instance parameters for calculation
        subpixel_factor = 2 ** self.subpixel_bits
        disp = npDisparity.astype(np.float32) / subpixel_factor
        with np.errstate(divide="ignore"):
            depth = (self.baseline_mm * self.focal_px) / disp
            depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        # temporary exploratory test: clip in this function
        # Mask out values outside the valid range
        mask = (depth >= self.depth_min) & (depth <= self.depth_max)
        depth = np.where(mask, depth, 0)
        # conclusion: no significant improvement in speed was noticed (but that is expected bc alignment doesn't take this into account). 
        return depth
    
    # === NN visualization functions ===
    def process_nn_depth(self, npDepth):
        normDepth = np.clip(
            (npDepth - self.nn_depth_min) / max(1e-6, (self.nn_depth_max - self.nn_depth_min)) * 255,
            0, 255
        ).astype(np.uint8)
        return cv2.applyColorMap(normDepth, cv2.COLORMAP_HOT)
    
    # alignment related method for adjusting how much of the rgb or depth you see in the aligned window
    def updateBlendWeights(self, percentRgb):
        """
        Update the rgb and depth weights used to blend depth/rgb image

        @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
        """
        self.rgbWeight = float(percentRgb) / 100.0
        self.depthWeight = 1.0 - self.rgbWeight

    def add_legend(self, image, min_val, max_val, label, colormap=cv2.COLORMAP_HOT):
        # Create a legend bar with width matching the image
        legend_height = 30
        legend_width = image.shape[1]
        legend = np.linspace(0, 255, legend_width, dtype=np.uint8)
        legend = np.tile(legend, (legend_height, 1))
        legend_color = cv2.applyColorMap(legend, colormap)
        # Add text
        cv2.putText(legend_color, f"{min_val}", (0, legend_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(legend_color, f"{max_val}", (legend_width-40, legend_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(legend_color, label, (legend_width//2-40, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # Stack legend below image
        return np.vstack((image, legend_color))

    def run(self):
        self.clickInfo = {"pt": None, "stereo_depth": None, "nn_depth": None} # stores stereo and NN depth of last clicked pixel in milimeters
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # y may be out of bounds if legend is stacked below image
                img_height = self.latest_stereo_depth.shape[0]
                if y >= img_height:
                    print("Clicked outside image area.")
                    return
                self.clickInfo["pt"] = (x, y)
                depth_string = f"Depth at pixel ({x},{y}) is "
                if self.latest_nn_depth is not None and y < self.latest_nn_depth.shape[0]:
                    nn_depth_m = self.latest_nn_depth[y, x]
                    nn_depth_mm = nn_depth_m * 1000
                    depth_string += f"|NN: {nn_depth_m*100:.2f} cm "
                    self.clickInfo["nn_depth"] = nn_depth_mm
                if self.latest_stereo_depth is not None and y < self.latest_stereo_depth.shape[0]:
                    stereo_depth_mm = self.latest_stereo_depth[y, x]
                    depth_string += f" |Stereo: {stereo_depth_mm/10:.2f} cm"
                    self.clickInfo["stereo_depth"] = stereo_depth_mm
                print(depth_string)

        if self.show_stereo_depth:
            stereoWindowName = "stereo-depth"
            cv2.namedWindow(stereoWindowName)
            cv2.setMouseCallback(stereoWindowName, mouse_callback)
            alignedWindowName = "aligned-rgb-depth"
            cv2.namedWindow(alignedWindowName)
            cv2.setMouseCallback(alignedWindowName, mouse_callback)

        # alignment related variables
        self.rgbWeight = 0.4
        self.depthWeight = 0.6

        with self.pipeline:
            self.pipeline.start()

            # Configure windows; trackbar adjusts blending ratio of rgb/depth
            # Set the window to be resizable and the initial size
            #cv2.namedWindow(alignedWindowName, cv2.WINDOW_NORMAL)
            #cv2.resizeWindow(alignedWindowName, 1280, 720)
            cv2.createTrackbar(
                "RGB Weight %",
                alignedWindowName,
                int(self.rgbWeight * 100),
                100,
                self.updateBlendWeights,
            )
            fpsCounter = FPSCounter()

            while self.pipeline.isRunning():
                fpsCounter.tick()
                leftSynced = self.syncedLeftQueue.get()
                rightSynced = self.syncedRightQueue.get()
                disparity = self.disparityQueue.get()
                confidence = self.confidenceQueue.get()
                depth = self.depthQueue.get()
                assert isinstance(leftSynced, dai.ImgFrame)
                assert isinstance(rightSynced, dai.ImgFrame)
                assert isinstance(disparity, dai.ImgFrame)
                assert isinstance(confidence, dai.ImgFrame)
                assert isinstance(depth, dai.ImgFrame)
                # get same for aligned
                alignedMessageGroup = self.syncAlignedQueue.get()
                assert isinstance(alignedMessageGroup, dai.MessageGroup)
                alignedRgb = alignedMessageGroup["rgb"]
                assert isinstance(alignedRgb, dai.ImgFrame)
                
                # NN depth
                alignedRgbCv = alignedRgb.getCvFrame()
                alignedRgbCvRgb = cv2.cvtColor(alignedRgbCv, cv2.COLOR_BGR2RGB)
                alignedRgbPil = Image.fromarray(alignedRgbCvRgb)
                predictions = self.nn_depth_pipe(alignedRgbPil)
                depth_tensor = predictions["predicted_depth"]
                npDepth_nn = depth_tensor.cpu().numpy()
                self.latest_nn_depth = npDepth_nn
                colorized_nn = self.process_nn_depth(npDepth_nn)
                colorized_nn = self.add_legend(colorized_nn, f"{self.nn_depth_min}m", f"{self.nn_depth_max}m", "NN Depth (m)")

                alignedDepth = alignedMessageGroup["depth_aligned"]
                assert isinstance(alignedDepth, dai.ImgFrame)
                if self.show_left:
                    cv2.imshow("left", leftSynced.getCvFrame())
                if self.show_right:
                    cv2.imshow("right", rightSynced.getCvFrame())
                npDisparity = disparity.getFrame()
                colorizedDisparity = self.process_disparity(npDisparity)
                colorizedDisparity = self.add_legend(colorizedDisparity, "0", "190", "Disparity (pixels)", self.colorMap)
                if self.show_disparity:
                    cv2.imshow("disparity", colorizedDisparity)
                colorizedConfidence = self.process_confidence(confidence.getFrame())
                colorizedConfidence = self.add_legend(colorizedConfidence, "0", "255", "Confidence", cv2.COLORMAP_JET)
                if self.show_confidence:
                    cv2.imshow("confidence", colorizedConfidence)

                # show the NN depth that was calculated earlier; TODO: rearrange the code so that everything is in a logical place...
                if self.show_nn_depth:
                    # draw red dot (circle) on NN depth window
                    cv2.circle(colorized_nn, self.clickInfo["pt"], 5, (0, 0, 255), -1)
                    cv2.imshow("nn_depth", colorized_nn)

                npDepth = depth.getFrame()
                min_depth = np.min(npDepth)
                max_depth = np.max(npDepth)
                # print(f"Depth min: {min_depth} mm, max: {max_depth} mm")
                colorizedDepth = self.process_stereo_depth(npDepth)
                colorizedDepth = self.add_legend(colorizedDepth, f"{self.depth_min}mm", f"{self.depth_max}mm", "Depth (mm)", cv2.COLORMAP_HOT)
                if self.show_stereo_depth:
                    # Draw dot on depth window
                    if self.clickInfo["pt"] is not None:
                        #cv2.circle(colorizedAlignedDepth, self.clickInfo["pt"], 5, (0, 0, 255), -1) # TODO: figure out why not do this here...
                        cv2.circle(colorizedDepth, self.clickInfo["pt"], 5, (0, 0, 255), -1)
                    cv2.imshow(stereoWindowName, colorizedDepth)
                # Calculate and print real (stereo) depth from disparity
                stereo_depth = self.calculate_stereo_depth(npDisparity)
                self.latest_stereo_depth = stereo_depth  # Update latest real (stereo) depth for mouse callback
                min_stereo = np.min(stereo_depth)
                max_stereo = np.max(stereo_depth)
                # show aligned window
                if self.show_aligned:
                    if alignedDepth is not None:
                        cvFrame = alignedRgb.getCvFrame() # how it's done in aligned code
                        npAlignedDepth = alignedDepth.getFrame() # how it's done in stereo_astra code
                        colorizedAlignedDepth = self.process_stereo_depth(npAlignedDepth) # TODO: calculated twice...!
                        # colorizedAlignedDepth = self.add_legend(colorizedAlignedDepth, f"{self.depth_min}mm", f"{self.depth_max}mm", "Depth (mm)", cv2.COLORMAP_HOT)
                        blended = cv2.addWeighted(
                            cvFrame, self.rgbWeight, colorizedAlignedDepth, self.depthWeight, 0
                        )
                        cv2.putText(
                            blended,
                            f"FPS: {fpsCounter.getFps():.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )
                        # Draw red dot if a pixel was selected
                        if self.clickInfo["pt"] is not None:
                            cv2.circle(blended, self.clickInfo["pt"], 5, (0, 0, 255), -1)
                            cv2.circle(colorizedDepth, self.clickInfo["pt"], 5, (0, 0, 255), -1)
                        cv2.imshow(alignedWindowName, blended)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.pipeline.stop()
                    break

if __name__ == "__main__":
    app = CombinedDepthApp()
    app.run()
