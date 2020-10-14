#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import concurrent

import grpc
import numpy as np

import captain_pb2
import captain_pb2_grpc

sys.path.append(os.path.dirname(os.path.join(os.path.dirname(__file__), '../../../')))
from model import TurtleNet


class CaptainServicer(captain_pb2_grpc.CaptainServicer):

    def __init__(self):
        self.model = TurtleNet.new()

    def MakeDecision(self, request, context):
        print('Captain.MakeDecision: %d -*-*-' % os.getpid())
        """
        print('- player_health:', request.player_health)
        print('- player_yaw:', request.player_yaw)
        print('- player_position:', request.player_position)
        print('- player_speed:', request.player_speed)
        print('- enemy_health:', request.enemy_health)
        print('- enemy_yaw:', request.enemy_yaw)
        print('- enemy_position:', request.enemy_position)
        print('- enemy_visible:', request.enemy_visible)
        print('- guns_embedding:', request.guns_embedding)
        print('- guns_loading:', request.guns_loading)
        print('- crosshair:', len(request.crosshair))
        print('- camera_rotation:', request.camera_rotation)
        """
        """
        tid = int(time.time() * 1000)
        args = (request.player_health, request.player_yaw, request.player_position.x, request.player_position.y,
                request.player_speed, request.enemy_health, request.enemy_yaw, request.enemy_position.x, request.enemy_position.y,
                request.enemy_visible, *request.guns_embedding, *request.guns_loading, '%s.png' % tid)
        fmt = '%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%s\n'
        with open('data/%d.csv' % tid, 'a') as f:
            f.write(fmt % args)
        """

        player = np.array([request.player_health, request.player_yaw, request.player_position.x, request.player_position.y], dtype=np.float32).reshape((-1, 4))
        knot = np.array([request.player_speed], dtype=np.float32).reshape((-1, 1))
        enemy = np.array([request.enemy_health, request.enemy_yaw, request.enemy_position.x, request.enemy_position.y, request.enemy_visible], dtype=np.float32).reshape((-1, 5))
        guns = np.array([request.guns_embedding, request.guns_loading], dtype=np.float32).reshape((-1, 2, 6))
        camera = np.array([request.camera_rotation], dtype=np.float32).reshape((-1, 1))

        data = {'player': player, 'knot': knot, 'enemy': enemy,
                'guns': guns, 'camera': camera}

        output = self.model(dict(data))

        steering = np.argmax(output[2])
        forwarding = np.argmax(output[3])
        fire = np.argmax(output[4]) == 1

        action = captain_pb2.Action(value=output[0], x=output[1], steering=steering, forwarding=forwarding, fire=fire)
        # action = captain_pb2.Action(value=0, x=0, y=0, steering=0, forwarding=0, fire=True)

        return action


def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=8))
    captain_pb2_grpc.add_CaptainServicer_to_server(CaptainServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()