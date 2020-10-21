#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import socket


class Protocol:
    pass


class ProtocolImpl(Protocol):
    pass


def recvall(sock: socket.socket, bufsize: int) -> bytes:
    buffer = b''
    while len(buffer) < bufsize:
        chunk = sock.recv(bufsize - len(buffer))
        if not chunk:
            raise EOFError('소켓이 닫혔습니다.')
        buffer += chunk
    return buffer


if __name__ == "__main__":
    pass
