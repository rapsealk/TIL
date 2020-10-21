#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import json
import socket
import struct

from util import recvall

ENDIAN = ">" if sys.byteorder == 'big' else "<"
FORMAT = ENDIAN + "I"

ADDR = ('127.0.0.1', 50637)


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(ADDR)

    print('Endian:', sys.byteorder)

    action = {
        "Type": "Request",
        "Target": "Unit"
    }

    message = json.dumps(action)
    header = struct.pack(FORMAT, len(message))      # int.to_bytes(len(message), byteorder=sys.byteorder)
    print('header: %r (%d bytes)' % (header, int.from_bytes(header, byteorder=sys.byteorder)))
    # [1] client -> server
    sock.sendall(header)    # Header (4 bytes) - 전송할 데이터의 크기 (bytes)
    sock.sendall(message.encode("utf-8"))   # Payload (n bytes) - 실제 데이터

    # [2] server -> client
    header = recvall(sock, 4)   # Header (4 bytes) - 전달받을 데이터의 크기 (bytes)
    data_size = struct.unpack(FORMAT, header)[0]    # int.from_bytes(header, byteorder=sys.byteorder)
    print('header: %r (%d bytes)' % (header, data_size))
    message = recvall(sock, data_size)  # Payload (n bytes) - 실제 데이터
    print('message: %r' % message)
    data = json.loads(message.decode("utf-8"))
    print('recv(id=%d, health=%d, location=%s):' % (data["Units"][0]["Id"], data["Units"][0]["Health"], data["Units"][0]["Location"]))

    data = {
        "Type": "Request",
        "Target": "Unit"
    }

    message = json.dumps(data)
    header = struct.pack(FORMAT, len(message))      # int.to_bytes(len(message), byteorder=sys.byteorder)
    print('header: %r (%d bytes)' % (header, int.from_bytes(header, byteorder=sys.byteorder)))
    # [3] client -> server
    sock.sendall(header)    # Header (4 bytes) - 전송할 데이터의 크기 (bytes)
    sock.sendall(message.encode("utf-8"))   # Payload (n bytes) - 실제 데이터

    # [4] Close
    sock.close()
