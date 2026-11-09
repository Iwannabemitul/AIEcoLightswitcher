# AIEcoLightswitcher

An intelligent Edge AI energy management system that processes video feeds from smartphones (**DroidCam** via Wi-Fi/USB), webcams, or IP cameras to automatically switch physical room appliances (LEDs/relays) on and off based on occupant presence (detecting humans and toy/doll proxies) using **YOLOv8** on laptops with **NVIDIA RTX 4050 GPU** and **Arduino**.

---

## Key Features

- **Occupant & Proxy Detection:** Uses YOLOv8 Nano (`yolov8n.pt`) configured for multi-class detection of humans (`person`, COCO class 0) as well as doll/toy proxies (`teddy bear`, COCO class 77).
- **Phone as Wi-Fi Camera (DroidCam):** Supports both DroidCam Virtual Webcam device index (e.g. `1` or `2`) and direct HTTP MJPEG streams (`http://<phone-ip>:4747/video`).
- **Hardware Acceleration:** Auto-detects NVIDIA RTX 4050 (`CUDA:0`) with automatic graceful fallback to CPU.
- **Power-Saving Duty Cycle:** Operates on an evaluation window (e.g., 2 seconds) followed by a power-saving sleep interval (5 seconds) to minimize edge compute energy consumption.
- **Microcontroller Integration:** Serial interface with Arduino Uno / Nano over USB (`COM3`, `COM4`, etc.) controlling Pin 13 LED or a relay module.
- **Interactive CustomTkinter Dashboard:**
  - Real-time video preview with bounding boxes and confidence scores.
  - Arduino COM port scanner and one-click connect/disconnect.
  - Manual overrides (`AUTO MODE`, `FORCE ON`, `FORCE OFF`).
  - School / Office Hours time simulator.
  - Fail-safe simulated camera fallback when hardware is disconnected.

---

## Hardware Requirements

- **Laptop:** NVIDIA GeForce RTX 4050 Laptop GPU (or standard CPU).
- **Camera:** Smartphone with DroidCam app installed (connected to same Wi-Fi as laptop or via USB).
- **Microcontroller:** Arduino UNO R3, Nano, or compatible microcontroller.
- **Output:** LED with 220Ω/330Ω resistor connected to Pin 13 (or a 5V relay module for AC appliances).

---

## Quick Start Guide

### 1. Install Software Dependencies
Ensure Python 3.10+ is installed. Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Setup DroidCam on Phone & Laptop
1. Install **DroidCam** on your Android or iOS device from the App Store / Play Store.
2. Install the **DroidCam Windows Client** on your laptop.
3. Connect both phone and laptop to the same Wi-Fi network.
4. Open DroidCam on your phone, note the **Phone IP** (e.g., `192.168.1.100`) and **DroidCam Port** (default `4747`).
5. In the DroidCam PC Client, click **Start**.

### 3. Flash Arduino Firmware
1. Open `Model Demo/classroom_lights.ino` in the Arduino IDE.
2. Select your Arduino board and COM port.
3. Click **Upload**.

### 4. Run the AI Vision Dashboard
Run the main dashboard application:
```bash
python "Model Demo/main.py"
```

In the sidebar:
- Select your Arduino COM port (e.g., `COM3` or `COM4`) and click **Connect**.
- Choose your camera source:
  - **DroidCam Dev**: Input device index `1` (or `2` depending on Windows camera index).
  - **DroidCam IP**: Input `http://<your-phone-ip>:4747/video`.
  - **Laptop Cam**: Input `0`.
- Click **Switch Camera Source**.
