#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

class StereoDepthApp:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.maxDisparity = 1
        self.colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

        # Stereo parameters (should match your camera calibration)
        self.baseline_mm = 75  # baseline in mm
        # OAK D PRO W
        self.focal_px = 568.924
        # OAK D PRO
        # self.focal_px = 798.6192626953125 # focal length in pixels

        self.subpixel_bits = 3    # subpixel bits (FAST_ACCURACY and FAST_DENSITY dont use subpixel, 5 for face and high detail, and 3 for default and robotics)

        # Depth range parameters (in mm)
        self.depth_min = 0
        self.depth_max = 15000
        
        self.setup_pipeline()
        self.latest_real_depth = None  # Store latest real depth frame

        # Visualization flags
        self.show_left = False
        self.show_right = False
        self.show_disparity = False
        self.show_confidence = True
        self.show_depth = True

    def setup_pipeline(self):
        monoLeft = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        monoRight = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)

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

        monoLeftOut = monoLeft.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        monoRightOut = monoRight.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
        monoLeftOut.link(stereo.left)
        monoRightOut.link(stereo.right)

        self.syncedLeftQueue = stereo.syncedLeft.createOutputQueue()
        self.syncedRightQueue = stereo.syncedRight.createOutputQueue()
        self.disparityQueue = stereo.disparity.createOutputQueue()
        self.confidenceQueue = stereo.confidenceMap.createOutputQueue()
        self.depthQueue = stereo.depth.createOutputQueue()

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

    def process_depth(self, npDepth):
        # Visualization scaling based on depth range
        normDepth = np.clip((npDepth - self.depth_min) / max(1, (self.depth_max - self.depth_min)) * 255, 0, 255).astype(np.uint8)
        colorizedDepth = cv2.applyColorMap(normDepth, cv2.COLORMAP_HOT)
        return colorizedDepth
    
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
    
    def calculate_real_depth(self, npDisparity):
        # Use instance parameters for calculation
        subpixel_factor = 2 ** self.subpixel_bits
        disp = npDisparity.astype(np.float32) / subpixel_factor
        with np.errstate(divide="ignore"):
            depth = (self.baseline_mm * self.focal_px) / disp
            depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
        return depth

    def run(self):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self.latest_real_depth is not None:
                # y may be out of bounds if legend is stacked below image
                img_height = self.latest_real_depth.shape[0]
                if y >= img_height:
                    print("Clicked outside image area.")
                    return
                depth_mm = self.latest_real_depth[y, x]
                depth_cm = depth_mm / 10.0
                print(f"Clicked at ({x},{y}): Depth = {depth_cm:.2f} cm")

        if self.show_depth:
            cv2.namedWindow("depth")
            cv2.setMouseCallback("depth", mouse_callback)

        with self.pipeline:
            self.pipeline.start()
            while self.pipeline.isRunning():
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
                npDepth = depth.getFrame()
                min_depth = np.min(npDepth)
                max_depth = np.max(npDepth)
                # print(f"Depth min: {min_depth} mm, max: {max_depth} mm")
                colorizedDepth = self.process_depth(npDepth)
                colorizedDepth = self.add_legend(colorizedDepth, f"{self.depth_min}mm", f"{self.depth_max}mm", "Depth (mm)", cv2.COLORMAP_HOT)
                if self.show_depth:
                    cv2.imshow("depth", colorizedDepth)
                # Calculate and print real depth from disparity
                real_depth = self.calculate_real_depth(npDisparity)
                self.latest_real_depth = real_depth  # Update latest real depth for mouse callback
                min_real = np.min(real_depth)
                max_real = np.max(real_depth)
                # print(f"Real depth from disparity: min={min_real:.2f} mm, max={max_real:.2f} mm")
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.pipeline.stop()
                    break

if __name__ == "__main__":
    app = StereoDepthApp()
    app.run()
