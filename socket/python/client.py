#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import json
import random
import socket
import struct

ENDIAN = ">" if sys.byteorder == 'big' else "<"
FORMAT = ENDIAN + "I"


def recvall(sock: socket.socket, bufsize: int) -> bytes:
    buffer = b''
    while len(buffer) < bufsize:
        chunk = sock.recv(bufsize - len(buffer))
        if not chunk:
            raise EOFError('소켓이 닫혔습니다.')
        buffer += chunk
    return buffer


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = ('127.0.0.1', 50637)
    sock.connect(addr)

    print('Endian:', sys.byteorder)

    data = {
        "Units": [
            {
                "Id": random.randint(1, 100),
                "Health": 18400 - random.randint(100, 10000),
                "MaxHealth": 18400,
                "Sensor": [
                    {
                        "Id": 0,
                        "Type": 0,
                        "Activated": True
                    }
                ],
                "Heading": 0.0,
                "Location": [random.random(), random.random(), random.random()]
            }
        ]
    }

    message = json.dumps(data)
    header = struct.pack(FORMAT, len(message))
    print('header: %r (%d bytes)' % (header, int.from_bytes(header, byteorder=sys.byteorder)))
    # Header (4 bytes) - 전송할 데이터의 크기 (bytes)
    sock.sendall(header)
    # Payload (n bytes) - 실제 데이터
    sock.sendall(message.encode("utf-8"))

    # Header (4 bytes) - 전달받을 데이터의 크기 (bytes)
    header = recvall(sock, 4)
    data_size = struct.unpack(FORMAT, header)[0]    # int.from_bytes(header, byteorder=sys.byteorder)
    # Payload (n bytes) - 실제 데이터
    message = recvall(sock, data_size)
    print('message: %r' % message)
    data = json.loads(message.decode("utf-8"))
    print('recv(id=%d, health=%d, location=%s):' % (data["Units"][0]["Id"], data["Units"][0]["Health"], data["Units"][0]["Location"]))

    sock.close()
