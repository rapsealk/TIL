#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

from dronekit import connect, VehicleMode

import time

def _log(message):
    print("[%s] %s" % (time.time(), message))

class VehicleCommandKit(object):

    def __init__(self, port='COM8'):
        _log("== VehicleCommandKit ==")

        self.vehicle = connect(port, wait_ready=True, baud=57600)

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

        self.commands = ['arm', 'disarm', 'takeoff', 'run']
        _log("Commands available: {}".format(self.commands))

    def __del__(self):
        #self.vehicle.close()
        _log("Vehicle has closed..")

    def location_callback(self, attr_name, msg):
        _log("Location (Global): {}".format(msg))

    """
    def mission(self):
        # Get commands object from Vehicle.
        cmds = self.vehicle.commands
        # Call clear() on Vehicle.commands and upload the command to the vehicle.
        cmds.clear()

        cmd1 = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, 10)
        cmd2 = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0, 0, 10, 10, 10)
        cmds.add(cmd1)
        cmds.add(cmd2)

        cmds.upload()
    """

    def arm(self):
        while not self.vehicle.is_armable:
            _log("Waiting for vehicle to initilize..")
            time.sleep(1)
        while not self.vehicle.armed:
            _log("Vehicle is not armed..")
            _log("Command \"Arm\"..")
            # self.vehicle.mode = VehicleMode("GUIDED") # OFFBOARD
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

    def takeoff(self, altitude=5):
        if not self.vehicle.armed:
            _log("Vehicle must be armed before takeoff..")
            return
        _log("Command \"Takeoff\"")
        _log("Takeoff target altitude: {}".format(altitude))
        self.vehicle.simple_takeoff(altitude)

    """
    def _offboard_move(self):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,      # time_boot_ms (not used)
            0, 0,   # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,    # frame
            0b0000111111111000, # type_mask (only positions enabled)
            north, east, down,  # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame)
            0, 0, 0,    # x, y, z velocity in m/s   (not used)
            0, 0, 0,    # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
            0, 0        # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
        )
        # send command to vehicle
        self.vehicle.send_mavlink(msg)
    """

    def run(self):
        # MAV_CMD_MISSION_START: if the vehicle is on the ground (only)
        self.vehicle.mode = VehicleMode("AUTO")

    def close(self):
        self.vehicle.close()

    def command(self, cmd):
        # exec('self.' + command + '()')
        if cmd.upper() == 'ARM':
            self.arm()
        elif cmd.upper() == 'DISARM':
            self.disarm()
        elif cmd.upper() == 'TAKEOFF':
            self.takeoff()
        elif cmd.upper() == 'RUN':
            self.run()
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