#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

from dronekit import connect, VehicleMode, Command, LocationGlobal
from pymavlink import mavutil

import time
import math
import threading

def LOG(message):
    timestamp = time.time()
    print("[%s] %s" % (timestamp, message))
    with open('log.txt', 'w') as log:
        log.write("[{}] {}\n".format(timestamp, message))

MAV_MODE_AUTO = 4

home_position_set = False

class VehicleCommandKit(object):

    def __init__(self, port='COM8'):
        LOG("== VehicleCommandKit ==")

        self.vehicle = connect(port, wait_ready=True, baud=57600)

        LOG("Got some vehicle attribute values:")
        LOG(" GPS: {}".format(self.vehicle.gps_0))
        LOG(" Battery: {}".format(self.vehicle.battery))
        LOG(" Last Heartbeat: {}".format(self.vehicle.last_heartbeat))
        LOG(" Is Armable?: {}".format(self.vehicle.is_armable))
        LOG(" System status: {}".format(self.vehicle.system_status.state))
        LOG(" Mode: {}".format(self.vehicle.mode.name))

        def gps_callback(vehicle):
            while True:
                gps = vehicle.location.global_frame
                LOG("GPS_CALLBACK")
                LOG("GPS: {}".format(gps))
                LOG("LAT: {}, LON: {}, ALT: {}".format(gps.lat, gps.lon, gps.alt))
                LOG("GPS_0: {}\n".format(vehicle.gps_0))
                time.sleep(1)

        self.gthread = threading.Thread(target=gps_callback, args=(self.vehicle,))
        self.gthread.daemon = True
        self.gthread.start()

        LOG("Global: {}".format(self.vehicle.location.global_frame))
        LOG("Global Latitude: {}".format(self.vehicle.location.global_frame.lat))
        LOG("Global Longitude: {}".format(self.vehicle.location.global_frame.lon))
        LOG("Global Altitude: {}".format(self.vehicle.location.global_frame.alt))

        LOG("Relative: {}".format(self.vehicle.location.global_relative_frame))
        LOG("Relative Latitude: {}".format(self.vehicle.location.global_relative_frame.lat))
        LOG("Relative Longitude: {}".format(self.vehicle.location.global_relative_frame.lon))
        LOG("Relative Altitude: {}".format(self.vehicle.location.global_relative_frame.alt))

        LOG("Local NED: {}".format(self.vehicle.location.local_frame))

        self.commands = ['arm', 'disarm', 'upload', 'takeoff', 'run']
        LOG("Commands available: {}".format(self.commands))

    def __del__(self):
        #self.vehicle.close()
        LOG("Vehicle has closed..")

    def get_location_offset_meters(self, original_location, dNorth, dEast, alt):
        """
        Returns a LocationGlobal object containing the latitude/longitude 'dNorth' and 'dEast' meters from the specified 'original_location'.
        The returned Location adds the entered 'alt' value to the altitude of the 'original_location'.
        The function is useful when you want to move the vehicle around specifying locations relative to the current vehicle position.
        The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
        For more information see:
        http://gis.stackexchange.com/questions/2951/algoithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
        """
        earth_radius = 6378137.0    # Radius of "spherical" earth
        # Coordinate offsets in radians
        dLat = dNorth / earth_radius
        dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

        # New position in decimal degrees
        newlat = original_location.lat + (dLat * 100/math.pi)
        newlon = original_location.lon + (dLon * 100/math.pi)
        return LocationGlobal(newlat, newlon, original_location.alt + alt)

    def PX4setMode(mavMode):
        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system, self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavMode,
            0, 0, 0, 0, 0, 0
        )

    """
    def location_callback(self, attr_name, msg):
        LOG("Location (Global): {}".format(msg))
    """

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

    def load_mission(self, filename="mission.txt"):
        LOG("Reading mission from file: {}".format(filename))
        cmds = self.vehicle.commands
        missionlist = []
        with open(filename) as fhandle:
            for i, line in enumerate(fhandle):
                if i == 0:
                    if not line.startswith("QGC WPL 110\n"):
                        # raise Exception("File is not supported WP version.")
                        LOG("File is not supported WP version..")
                        return
                else:
                    values = line.split('\t')
                    ln_index = int(values[0])
                    ln_current_wp = int(values[1])
                    ln_frame = int(values[2])
                    ln_command = int(values[3])
                    ln_param1 = float(values[4])
                    ln_param2 = float(values[5])
                    ln_param3 = float(values[6])
                    ln_param4 = float(values[7])
                    ln_param5 = float(values[8])
                    ln_param6 = float(values[9])
                    ln_param7 = float(values[10])
                    ln_autocontinue = int(values[11].strip())
                    cmd = Command(0, 0, 0, ln_frame, ln_command, ln_current_wp, ln_autocontinue,
                                ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
                    missionlist.append(cmd)
        return missionlist

    def upload(self, filename="mission.txt"):
        missionlist = self.load_mission(filename)

        LOG("Upload mission from a file: {}".format(filename))
        
        # Clear existing mission from vehicle.
        LOG("Clear mission..")
        cmds = self.vehicle.commands
        cmds.clear()

        # Add new mission to vehicle
        for command in missionlist:
            cmds.add(command)
        LOG("Upload mission..")
        self.vehicle.commands.upload()

    def download_mission(self):
        """
        Download the current mission and returns it in a list.
        It is used in save_mission() to get the file information to save.
        """
        LOG("Download mission from vehicle..")
        missionlist = []
        cmds = self.vehicle.commands
        cmds.download()
        cmds.wait_ready()
        for cmd in cmds:
            missionlist.append(cmd)
        return missionlist

    def save_mission(self, filename='mission_{}.txt'.format(time.time())):
        """
        Save a mission in the Waypoint file format (http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).
        """
        LOG("Save mission from Vehicle to file: {}".format(filename))
        missionlist = self.download_mission()
        # Add file format information
        output = "QGC WPL 110\n"
        # Add home location as 0th waypoint
        home = self.vehicle.home_location
        output += "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0, 1, 0, 16, 0, 0, 0, 0, home.lat, home.lon, home.alt, 1)
        # Add commands
        for cmd in missionlist:
            commandline = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq, cmd.current, cmd.frame, cmd.command,
                                                                                cmd.param1, cmd.param2, cmd.param3, cmd.param4,
                                                                                cmd.x, cmd.y, cmd.z, cmd.autocontinue)
            output += commandline
        with open(filename, 'w') as fhandle:
            LOG(" Write mission to file..")
            fhandle.write(output)

    def arm(self):
        """
        while not self.vehicle.is_armable:
            LOG("Waiting for vehicle to initilize..")
            time.sleep(1)
        """
        while not self.vehicle.armed:
            LOG("Vehicle is not armed..")
            LOG("Command \"Arm\"..")
            # self.vehicle.mode = VehicleMode("GUIDED") # OFFBOARD
            self.vehicle.armed = True
            time.sleep(1)
        LOG("Vehicle is now armed..")

    def disarm(self):
        while self.vehicle.armed:
            LOG("Vehicle is armed..")
            LOG("Command \"Disarm\"")
            self.vehicle.armed = False
            time.sleep(1)
        LOG("Vehicle is now disarmed..")

    def takeoff(self, altitude=5):
        if not self.vehicle.armed:
            LOG("Vehicle must be armed before takeoff..")
            return
        LOG("Command \"Takeoff\"")
        LOG("Takeoff target altitude: {}".format(altitude))
        self.vehicle.simple_takeoff(altitude)

    def _offboard_move(self):
        north = east = down = 0 # TODO: Retrieve from mission
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

    def run(self):

        if not self.vehicle.armed:
            self.takeoff()

        self.upload()
        # MAV_CMD_MISSION_START: if the vehicle is on the ground (only)
        #self.vehicle.mode = VehicleMode("AUTO")
        
        # monitor mission execution
        wp = self.vehicle.commands.next
        while wp < len(self.vehicle.commands):
            if self.vehicle.commands.next > wp:
                LOG("Moving to waypoint {}".format(self.vehicle.commands.next + 1))
            time.sleep(1)
        # wait for the vehicle to land
        while self.vehicle.commands.next > 0:
            time.sleep(1)
        
    def close(self):
        self.vehicle.close()

    def command(self, cmd):
        if cmd.lower() in self.commands:
            exec('self.' + cmd.lower() + '()')
        else:
            LOG("Invalid command: " + cmd)


if __name__ == '__main__':
    kit = VehicleCommandKit()

    while True:
        cmd = raw_input("Command: ")
        if cmd == 'x':
            break
        kit.command(cmd)
    kit.close()
    LOG("Python process has finished.")