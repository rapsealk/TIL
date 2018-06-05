import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;
 
public class JServer {
	// 연결할 포트를 지정합니다.
	private static final int PORT = 8000;
	public static void main(String[] args) {
 
		try {
			// 서버소켓 생성
			ServerSocket serverSocket = new ServerSocket(PORT);
            System.out.println("socket at port: " + PORT);
 
			// 소켓서버가 종료될때까지 무한루프
			while (true) {
                System.out.println("waiting..");
				// 소켓 접속 요청이 올때까지 대기합니다.
				Socket socket = serverSocket.accept();
                System.out.println("socket accepted!");
				try {
                    InputStream istream = socket.getInputStream();
                    BufferedReader br = new BufferedReader(new InputStreamReader(istream));
                    String response = br.readLine();
                    System.out.println("response: " + response);
                    // System.out.println(istream.read(1024));
					// 응답을 위해 스트림을 얻어옵니다.
					OutputStream stream = socket.getOutputStream();
					// 그리고 현재 날짜를 출력해줍니다.
					stream.write(new Date().toString().getBytes());
				} catch (Exception e) {
					e.printStackTrace();
				} finally {
					// 반드시 소켓은 닫습니다.
					socket.close();
                    break;
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}