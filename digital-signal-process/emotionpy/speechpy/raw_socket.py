import socket

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

    def read_gapi(self, url):
        ADDRESS = '192.168.195.186'
        self.connect(ADDRESS, 5000)
        self.send(url + '\n')
        gapi = float(self.receive())
        self.close()
        return gapi


if __name__ == '__main__':
    TCPSocket()