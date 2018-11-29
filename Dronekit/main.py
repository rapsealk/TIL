#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dronekit import connect, VehicleMode
import time

connection_string = '/dev/ttyUSB0'
print "Connecting to vehicle on: %s" % (connection_string)
vehicle = connect(connection_string, wait_ready=True, baud=57600)

print "Get some vehicle attribute values:"
print " GPS: %s" % vehicle.gps_0
print " Battery: %s" % vehicle.battery
print " Last Heartbeat: %s" % vehicle.last_heartbeat
print " Is Armable?: %s" % vehicle.is_armable
print " System status: %s" % vehicle.system_status.state
print " Mode: %s" % vehicle.mode.name
"""
raw_input("Press enter to arm.")
vehicle.armed = True
time.sleep(3)
raw_input("Press enter to disarm.")
vehicle.disarmed = False
time.sleep(3)
"""

# Wait for the drone to be armed.
while not vehicle.armed:
	print "Vehicle is not armed at ", time.time()
	time.sleep(1)
while vehicle.armed:
	print "Disarm vehicle at ", time.time()
	vehicle.armed = False
	time.sleep(3)
raw_input("Press enter to disconnect.")
vehicle.close()
