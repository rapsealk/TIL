import socket

class TCPSocket():

    def __init__(self):
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    def connect(self, host, port):
        self.host = host
        self.port = port
        self.socket.connect((self.host, self.port))
        print('== connected ==')

    def send(self, message):
        self.socket.sendall(message)
        # self.socket.sendall(message.encode())
        print('message sent!')

    def receive(self):
        response = self.socket.recv(1024)
        print('response:', response)

        self.socket.close()

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    sockett = TCPSocket()
    address = '192.168.162.195' # '192.168.166.153' # '192.168.161.103'
    sockett.connect(address, 5000)
    message = 'https://firebasestorage.googleapis.com/v0/b/kaubrain418.appspot.com/o/wav%2F%5B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%A2%E1%84%8B%E1%85%A7%E1%86%BC%5D%20Angry23.WAV?alt=media&token=ed74c009-08e9-4f55-8d71-68263dcff228'
    # message = input('message: ')
    sockett.send(message+'\n')
    # sockett.send(message)
    sockett.receive()
    sockett.close()