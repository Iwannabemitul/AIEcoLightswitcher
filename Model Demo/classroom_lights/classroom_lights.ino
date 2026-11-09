#include <Arduino.h>

// Define the pin where your classroom LEDs are connected
const int ledPin = 13; 

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW); // Start with lights off
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == '1') {
      digitalWrite(ledPin, HIGH);
      Serial.println("LED_ON");
    } else if (command == '0') {
      digitalWrite(ledPin, LOW);
      Serial.println("LED_OFF");
    } else if (command == '?') {
      Serial.println("AIEcoLightswitcher_v2");
    }
  }
}
