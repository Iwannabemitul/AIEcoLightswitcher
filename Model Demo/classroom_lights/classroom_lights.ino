/**
 * AIEcoLightswitcher — Multi-Pin Hardware Controller Firmware
 * Target: Arduino Uno / Nano (ATmega328P)
 * Supported Pins: Digital Pins 2–19 (A0–A5 mapped to digital 14–19)
 * Reserved Pins: Pins 0 & 1 (Hardware UART RX/TX)
 * Memory: Strict 0 dynamic heap allocations (0 bytes String, 0 malloc/free/new)
 */

#include <Arduino.h>
#include <string.h>
#include <stdlib.h>

// --- Configuration Constants ---
static const uint8_t PIN_MIN = 2;
static const uint8_t PIN_MAX = 19;
static const uint8_t LEGACY_PIN = 13;
static const unsigned long BAUDRATE = 9600;
static const unsigned long DEFAULT_FAILSAFE_MS = 15000; // 15 seconds watchdog

// --- State Variables (Static RAM) ---
static uint8_t pin_states[20]; // Direct indexing by pin number (20 bytes static SRAM)
static char rx_buffer[64];     // Static linear command buffer (64 bytes static SRAM)
static uint8_t rx_index = 0;
static unsigned long last_heartbeat = 0;
static bool failsafe_armed = true;

// --- Forward Declarations ---
void process_command(char* cmd);
void set_pin_state(uint8_t pin, uint8_t state);
void reset_failsafe();
void dump_status();

void setup() {
  Serial.begin(BAUDRATE);

  // Initialize all controllable digital pins (Pins 2-19) as OUTPUT and LOW
  for (uint8_t pin = PIN_MIN; pin <= PIN_MAX; pin++) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
    pin_states[pin] = 0;
  }

  last_heartbeat = millis();
  failsafe_armed = true;

  // Signal ready to host
  Serial.println(F("READY"));
}

void loop() {
  unsigned long now = millis();

  // --- Watchdog / Failsafe Check ---
  if (failsafe_armed && (now - last_heartbeat >= DEFAULT_FAILSAFE_MS)) {
    for (uint8_t pin = PIN_MIN; pin <= PIN_MAX; pin++) {
      digitalWrite(pin, LOW);
      pin_states[pin] = 0;
    }
    failsafe_armed = false; // Trigger once until next heartbeat
    Serial.println(F("FAILSAFE_TRIGGERED"));
  }

  // --- Serial Character Ingestion ---
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    // 1. Dual-mode legacy single-character compatibility (no newline required)
    if (rx_index == 0) {
      if (c == '1') {
        set_pin_state(LEGACY_PIN, 1);
        Serial.println(F("LED_ON"));
        reset_failsafe();
        continue;
      } else if (c == '0') {
        set_pin_state(LEGACY_PIN, 0);
        Serial.println(F("LED_OFF"));
        reset_failsafe();
        continue;
      } else if (c == '?') {
        Serial.println(F("AIEcoLightswitcher_v2"));
        reset_failsafe();
        continue;
      } else if (c == '\r' || c == '\n') {
        // Discard blank line delimiters
        continue;
      }
    }

    // 2. Framed command collection
    if (c == '\r') {
      continue; // Strip carriage returns
    }

    if (c == '\n') {
      // Delimiter reached: null terminate and dispatch
      rx_buffer[rx_index] = '\0';
      if (rx_index > 0) {
        process_command(rx_buffer);
      }
      rx_index = 0;
    } else {
      // Buffer storage with overflow guard
      if (rx_index < sizeof(rx_buffer) - 1) {
        rx_buffer[rx_index++] = c;
      } else {
        // Overflow: flush buffer and alert
        rx_index = 0;
        Serial.println(F("ERR OVERFLOW"));
      }
    }
  }
}

void reset_failsafe() {
  last_heartbeat = millis();
  failsafe_armed = true;
}

void set_pin_state(uint8_t pin, uint8_t state) {
  digitalWrite(pin, state ? HIGH : LOW);
  pin_states[pin] = state ? 1 : 0;
}

void dump_status() {
  Serial.print(F("STATUS"));
  for (uint8_t pin = PIN_MIN; pin <= PIN_MAX; pin++) {
    Serial.print(F(" "));
    Serial.print(pin);
    Serial.print(F(":"));
    Serial.print(pin_states[pin]);
  }
  Serial.println();
}

void process_command(char* cmd) {
  // Strip leading whitespace
  while (*cmd == ' ' || *cmd == '\t') cmd++;
  if (*cmd == '\0') return;

  // --- PING ---
  if (strcmp(cmd, "PING") == 0) {
    Serial.println(F("PONG"));
    reset_failsafe();
    return;
  }

  // --- IDENT ---
  if (strcmp(cmd, "IDENT") == 0) {
    Serial.println(F("AIEcoLightswitcher_v2"));
    reset_failsafe();
    return;
  }

  // --- STATUS ---
  if (strcmp(cmd, "STATUS") == 0) {
    dump_status();
    reset_failsafe();
    return;
  }

  // --- SET <pin> <state> ---
  if (strncmp(cmd, "SET ", 4) == 0) {
    char* p = cmd + 4;
    while (*p == ' ') p++;
    if (*p == '\0') {
      Serial.println(F("ERR SYNTAX"));
      return;
    }

    int pin = atoi(p);
    // Find space delimiter between pin and state
    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;
    if (*p == '\0') {
      Serial.println(F("ERR SYNTAX"));
      return;
    }

    int state = atoi(p);

    // Validate pin boundaries (Pins 2 to 19 inclusive)
    if (pin < PIN_MIN || pin > PIN_MAX) {
      Serial.println(F("ERR INVALID_PIN"));
      return;
    }

    // Validate state (must be 0 or 1)
    if (state != 0 && state != 1) {
      Serial.println(F("ERR INVALID_STATE"));
      return;
    }

    set_pin_state((uint8_t)pin, (uint8_t)state);
    reset_failsafe();
    Serial.print(F("OK SET "));
    Serial.print(pin);
    Serial.print(F(" "));
    Serial.println(state);
    return;
  }

  // --- GET <pin> ---
  if (strncmp(cmd, "GET ", 4) == 0) {
    char* p = cmd + 4;
    while (*p == ' ') p++;
    if (*p == '\0') {
      Serial.println(F("ERR SYNTAX"));
      return;
    }

    int pin = atoi(p);
    if (pin < PIN_MIN || pin > PIN_MAX) {
      Serial.println(F("ERR INVALID_PIN"));
      return;
    }

    reset_failsafe();
    Serial.print(F("PIN "));
    Serial.print(pin);
    Serial.print(F(" "));
    Serial.println(pin_states[pin]);
    return;
  }

  // --- Unknown Command ---
  Serial.println(F("ERR UNKNOWN_COMMAND"));
}
