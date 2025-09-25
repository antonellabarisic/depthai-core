#!/usr/bin/env python3

import depthai as dai

def get_default_resolution(cam, verbose=False):
    """cam is object dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, or dai.CameraBoardSocket.CAM_C"""
    with dai.Device(dai.UsbSpeed.HIGH) as device:
        calibData = device.readCalibration()
        M_def, def_w, def_h = calibData.getDefaultIntrinsics(cam)
        if verbose:
            # print("Calibration intrinsics (3x3):\n", M_def)
            print("Calibration resolution:", def_w, def_h)
        return def_w, def_h

def get_focal_length(cam, resolution=None, verbose=False):
    """cam is object dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, or dai.CameraBoardSocket.CAM_C"""
    if resolution is None:
        # get resolution of camera to get that specific focal length
        res_w, res_h = get_default_resolution(cam)
    else:
        res_w, res_h = resolution
    with dai.Device(dai.UsbSpeed.HIGH) as device:
        calibData = device.readCalibration()
        intrinsics = calibData.getCameraIntrinsics(cam, res_w, res_h)
        focal_px = intrinsics[0][0]
        if verbose:
            print(cam, 'focal length in pixels:', focal_px)
    return focal_px
        # # expected values:
        # self.baseline_mm = 75  # baseline in mm
        # # OAK D PRO W
        # self.focal_px = 568.924
        # # OAK D PRO
        # # self.focal_px = 798.6192626953125 # focal length in pixels

if __name__ == "__main__":
    width, height = get_default_resolution(dai.CameraBoardSocket.CAM_A, verbose=True)
    print("this is the width:", width)
    print("and this is the height:", height)
    get_default_resolution(dai.CameraBoardSocket.CAM_B, verbose=True)
    get_default_resolution(dai.CameraBoardSocket.CAM_C, verbose=True)

    get_focal_length(dai.CameraBoardSocket.CAM_A, verbose=True)
    get_focal_length(dai.CameraBoardSocket.CAM_B, (1280, 800), verbose=True)
    get_focal_length(dai.CameraBoardSocket.CAM_B, (640, 400), verbose=True)
    get_focal_length(dai.CameraBoardSocket.CAM_C, verbose=True)
