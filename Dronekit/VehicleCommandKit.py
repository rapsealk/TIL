#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

from dronekit import connect, VehicleMode

import time

def _log(message):
    print("[%s] %s" % (time.time(), message))

class VehicleCommandKit(object):

    def __init__(self):
        _log("== VehicleCommandKit ==")

        self.port = 'COM8'
        self.vehicle = connect(self.port, wait_ready=True, baud=57600)

        _log("Got some vehicle attribute values:")
        _log(" GPS: " + self.vehicle.gps_0)
        _log(" Battery: " + self.vehicle.battery)
        _log(" Last Heartbeat: " + self.vehicle.last_heartbeat)
        _log(" Is Armable?: " + self.vehicle.is_armable)
        _log(" System status: " + self.vehicle.system_status.state)
        _log(" Mode: " + self.vehicle.mode.name)

        # Watch location update
        self.vehicle.add_attribute_listener('global_frame', self.location_callback)
        _log("Global: {}".format(self.vehicle.location.global_frame))
        _log("Global Latitude: {}".format(self.vehicle.location.global_frame.lat))
        _log("Global Longitude: {}".format(self.vehicle.location.global_frame.lon))
        _log("Global Altitude: {}".format(self.vehicle.location.global_frame.alt))

        _log("Relative: {}".format(self.vehicle.location.global_relative_frame))
        _log("Relative Latitude: {}".format(self.vehicle.location.global_relative_frame.lat))
        _log("Relative Longitude: {}".format(self.vehicle.location.global_relative_frame.lon))
        _log("Relative Altitude: {}".format(self.vehicle.location.global_relative_frame.alt))

        _log("Local NED: {}".format(self.vehicle.location.local_frame))

    def __del__(self):
        #self.vehicle.close()
        _log("Vehicle has closed..")

    def location_callback(self, attr_name, msg):
        _log("Location (Global): {}".format(msg))

=    def arm(self):
        while not self.vehicle.armed:
            _log("Vehicle is not armed..")
            _log("Command \"Arm\"..")
            self.vehicle.armed = True
            time.sleep(1)
        _log("Vehicle is now armed..")

    def disarm(self):
        while self.vehicle.armed:
            _log("Vehicle is armed..")
            _log("Command \"Disarm\"")
            self.vehicle.armed = False
            time.sleep(1)
        _log("Vehicle is now disarmed..")

    def close(self):
        self.vehicle.close()

    def command(self, cmd):
        if cmd.upper() == 'ARM':
            self.arm()
        elif cmd.upper() == 'DISARM':
            self.disarm()
        else:
            _log("Invalid command: " + cmd)


if __name__ == '__main__':
    kit = VehicleCommandKit()
    while True:
        cmd = raw_input("Command: ")
        if cmd == 'x':
            break
        kit.command(cmd)
    kit.close()
    _log("Python process has finished.")