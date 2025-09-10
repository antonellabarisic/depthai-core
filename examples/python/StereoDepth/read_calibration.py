#!/usr/bin/env python3

import depthai as dai
import json


device = dai.Device(dai.UsbSpeed.HIGH)
output = {}
output['is_eeprom_available'] = device.isEepromAvailable()

# User calibration
try:
    user_calib = device.readCalibration2().eepromToJson()
    output['user_calibration'] = user_calib
except Exception as ex:
    output['user_calibration_error'] = str(ex)

# Factory calibration
try:
    factory_calib = device.readFactoryCalibration().eepromToJson()
    output['factory_calibration'] = factory_calib
except Exception as ex:
    output['factory_calibration_error'] = str(ex)

# Save to JSON file
with open('calibration_output.json', 'w') as f:
    json.dump(output, f, indent=2)
