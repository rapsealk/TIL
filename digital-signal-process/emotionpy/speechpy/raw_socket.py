import socket
# from socket import socket

class TCPSocket():

    def __init__(self):        
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    def connect(self, host, port):
        self.host = host
        self.port = port
        self.socket.connect((self.host, self.port))

    def send(self, message):
        print(message.encode())
        self.socket.sendall(message.encode())
        print('message sent!')

    def receive(self):
        response = self.socket.recv(1024)
        print('response:', response.decode('utf-8'))
        return float(response.decode('utf-8'))

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    TCPSocket()