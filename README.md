# AIEcoLightswitcher

Making an AI which takes feed from multiple CCTV/laptop/phone camera(s) and turns appliances on/off acc to the ppl in the room.

For the Model Demo, one must install the Ultralytics library in Python along with OpenCV, PySerial, and CustomTkinter to run the AI vision dashboard.

## Project Architecture
This showcase demonstrates edge computing and resource management by bridging a Python-based Computer Vision system (YOLOv8) with an Arduino microcontroller.

*   **AI Occupant Detection:** Uses a pre-trained YOLOv8 Nano model (`yolov8n.pt`) to identify occupants (configured to recognize specific proxy objects for the miniature demo).
*   **Optimized Duty Cycle:** Operates on a 1.5-second evaluation window followed by a 5-second power-saving sleep interval to drastically minimize CPU load.
*   **Time Simulation:** Built-in GUI slider to simulate "School Hours" (08:00 to 16:00). The system automatically halts processing and cuts power outside of these hours.
*   **Hardware Fail-safes:** The UI features a simulated camera feed and manual overrides (FORCE ON / FORCE OFF) that keep the presentation running even if the physical webcam or Arduino disconnects mid-demo.

## Hardware Requirements
*   Arduino UNO R3 (CH340G or official)
*   8mm LED
*   220Ω or 330Ω Resistor
*   USB A to B Cable

## Software Dependencies
Ensure Python 3.x is installed. Run the following command to install all required libraries:

```bash
pip install opencv-python pyserial customtkinter Pillow numpy ultralytics
