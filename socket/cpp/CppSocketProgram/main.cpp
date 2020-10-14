#include <iostream>
#include <cstdlib>
//#include <thread>
//#include <mutex>
#include <stdint.h>
#include <vector>

#ifdef _WIN32
//#include <Windows.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
// link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")
#elif linux
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#define DEFAULT_PORT "50637"
#define DEFAULT_BUFLEN 512
#define HEADER_SIZE 4

/*
typedef struct {
	int a;
} action_packet_t;
*/

int recv_s(SOCKET s, char* buf, int len, int flags);
int send_s(SOCKET s, const char* buf, int len, int flags);

DWORD WINAPI SocketThreadFn(LPVOID lpParameter);

DWORD WINAPI thread_fn(LPVOID lpParameter)
{
	unsigned int& counter = *((unsigned int*)lpParameter);
	while (counter < 0xFFFFFFFF) ++counter;
	return 0;
}

struct Endian {
	constexpr static bool IsBig() {
		union {
			uint32_t i;
			uint8_t c[4];
		} check = { 0x01000000 };
		return check.c[0];
	}

	constexpr static bool IsLittle() {
		return !IsBig();
	}
};

/**
 * https://docs.microsoft.com/en-us/windows/win32/winsock/winsock-server-application
 */
int main(int argc, char* argv[])
{
	// Step 01. Creating a Socket for the Server
	struct addrinfo* result = NULL;
	struct addrinfo* ptr = NULL;
	struct addrinfo hints;

	WORD wVersionRequested = MAKEWORD(2, 2);	// WinSock 2.2
	WSADATA wsaData;
	int iErrorStatus = WSAStartup(wVersionRequested, &wsaData);
	if (iErrorStatus != 0) {
		std::cout << "Failed to WSAStartup.." << std::endl;
		return 1;
	}

	if ((LOBYTE(wsaData.wVersion) != LOBYTE(wVersionRequested))
		|| (HIBYTE(wsaData.wVersion) != HIBYTE(wVersionRequested))) {
		std::cout << "WSA Unsupported version." << std::endl;
		return 1;
	}

	std::cout << "Version: " << static_cast<int>LOBYTE(wsaData.wVersion) << "." << static_cast<int>HIBYTE(wsaData.wVersion) << std::endl;

	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;

	// Resolve the local address and port to be used by the server.
	int iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
	if (iResult != 0) {
		std::cout << "getaddrinfo failed: " << iResult << std::endl;
		WSACleanup();
		return 1;
	}

	SOCKET ListenSocket = INVALID_SOCKET;
	ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (ListenSocket == INVALID_SOCKET) {
		std::cout << "Error at socket(): " << WSAGetLastError() << std::endl;
		freeaddrinfo(result);
		WSACleanup();
		return 1;
	}
	std::cout << "Socket is listening.." << std::endl;

	// Step 02. Binding a Socket
	iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
	if (iResult == SOCKET_ERROR) {
		std::cout << "Binding failed with error: " << WSAGetLastError() << std::endl;
		freeaddrinfo(result);
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}

	// Step 03. Listening on a Socket
	if (listen(ListenSocket, SOMAXCONN) == SOCKET_ERROR) {
		std::cout << "Listen failed with error: " << WSAGetLastError() << std::endl;
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}

	// Step 04. Accepting a Connection
	std::vector<HANDLE> threads;

	while (true) {
		// Accept a client socket
		SOCKET ClientSocket = accept(ListenSocket, NULL, NULL);
		if (ClientSocket == INVALID_SOCKET) {
			std::cout << "accept failed: " << WSAGetLastError() << std::endl;
			closesocket(ListenSocket);
			WSACleanup();
			return 1;
		}

		DWORD threadId;
		HANDLE handle = CreateThread(NULL, 0, SocketThreadFn, &ClientSocket, 0, &threadId);
		threads.push_back(handle);
	}

	for (const auto& thread : threads) {
		if (thread != NULL) {
			CloseHandle(thread);
		}
	}

	// Step 05. Receiving and Sending Data on the Server
	
	WSACleanup();

	return 0;
}

int recv_s(SOCKET s, char* buf, int len, int flags)
{
	int iTotalBytes = 0;
	while (iTotalBytes < len) {
		int iResult = recv(s, buf + iTotalBytes, len - iTotalBytes, flags);
		if (iResult <= 0) {
			return iResult;
		}
		iTotalBytes += iResult;
	}
	return iTotalBytes;
}

int send_s(SOCKET s, const char* buf, int len, int flags)
{
	int iTotalBytes = 0;
	while (iTotalBytes < len) {
		int iSendResult = send(s, buf + iTotalBytes, len - iTotalBytes, flags);
		if (iSendResult == SOCKET_ERROR) {
			return iSendResult;
		}
		iTotalBytes += iSendResult;
	}
	return iTotalBytes;
}

DWORD WINAPI SocketThreadFn(LPVOID lpParameter)
{
	SOCKET& socket = *((SOCKET*)lpParameter);

	int iResult = 0;

	char headerbuf[HEADER_SIZE + 1];
	char recvbuf[DEFAULT_BUFLEN];
	int iSendResult;
	int recvbuflen = DEFAULT_BUFLEN;

	// Receive until the peer shuts down the connection
	do {
		// Header (4 bytes) - 전달받을 데이터의 크기 (bytes)
		iResult = recv_s(socket, headerbuf, HEADER_SIZE, 0);
		if (iResult == 0) {
			std::cout << "Connection closed.." << std::endl;
			break;
		}
		else if (iResult < 0) {
			std::cout << "recv failed: " << WSAGetLastError() << std::endl;
			closesocket(socket);
			WSACleanup();
			return 1;
		}
		headerbuf[4] = NULL;
		std::printf("Header: %x %x %x %x\n", headerbuf[0], headerbuf[1], headerbuf[2], headerbuf[3]);
		unsigned int bytes = 0;
		if (Endian::IsLittle()) {
			for (int i = 0; i < HEADER_SIZE; i++) {
				bytes += ((unsigned int)headerbuf[i] & 0xFF) << (i * 8);
			}
		}
		else {
			for (int i = HEADER_SIZE; i > 0; i--) {
				bytes += ((unsigned int)headerbuf[i - 1] & 0xFF) << ((HEADER_SIZE - i - 1) * 8);
			}
		}
		std::cout << "Header: " << bytes << " bytes" << std::endl;

		// Payload (n bytes) - 실제 데이터
		iResult = recv_s(socket, recvbuf, bytes, 0);
		if (iResult == 0) {
			std::cout << "Connection closed.." << std::endl;
			break;
		}
		else if (iResult < 0) {
			std::cout << "recv failed: " << WSAGetLastError() << std::endl;
			closesocket(socket);
			WSACleanup();
			return 1;
		}
		recvbuf[bytes + 1] = NULL;
		//std::string message(recvbuf);
		//std::cout << "Message: " << message << std::endl;

		// Header (4 bytes) - 전달할 데이터의 크기 (bytes)
		if (Endian::IsLittle()) {
			for (int i = 0; i < HEADER_SIZE; i++) {
				headerbuf[i] = bytes >> (i * 8) & 0xFF;
			}
		}
		else {
			for (int i = HEADER_SIZE; i > 0; i--) {
				headerbuf[i] = bytes >> ((HEADER_SIZE - i - 1) * 8) & 0xFF;
			}
		}
		iSendResult = send_s(socket, headerbuf, 4, 0);
		if (iSendResult == SOCKET_ERROR) {
			std::cout << "send failed: " << WSAGetLastError() << std::endl;
			closesocket(socket);
			WSACleanup();
			return 1;
		}

		// Payload (n bytes) - 실제 데이터
		iSendResult = send_s(socket, recvbuf, iResult, 0);
		if (iSendResult == SOCKET_ERROR) {
			std::cout << "send failed: " << WSAGetLastError() << std::endl;
			closesocket(socket);
			WSACleanup();
			return 1;
		}
		std::cout << "Bytes sent: " << iSendResult << " / " << iResult << std::endl;
	} while (iResult > 0);

	// Step 06. Disconnecting the Server
	// shutdown the send half of the connection since no more data will be sent
	iResult = shutdown(socket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		std::cout << "shutdown failed: " << WSAGetLastError() << std::endl;
		closesocket(socket);
		WSACleanup();
		return 1;
	}

	// cleanup
	closesocket(socket);

	return 0;
}