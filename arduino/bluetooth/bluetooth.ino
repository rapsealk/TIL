#include <SoftwareSerial.h> // 시리얼통신 라이브러리

int Tx = 3; // 보내는 핀
int Rx = 2; // 받는 핀
SoftwareSerial serial(Tx, Rx);  // 시리얼통신을 위한 객체

void setup() {
  Serial.begin(9600); // 시리얼 모니터
  serial.begin(9600); // 블루투스 시리얼
}

void loop() {
  if (serial.available()) {
    Serial.write(serial.read());  // 블루투스 내용을 시리얼 모니터에 출력
  }
  if (Serial.available()) {
    serial.write(Serial.read());  // 시리얼 모니터 내용을 블루투스에 WRITE
  }
}
