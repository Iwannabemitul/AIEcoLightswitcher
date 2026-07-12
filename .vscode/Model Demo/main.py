import cv2
import numpy as np
import serial
import time
import customtkinter as ctk
from PIL import Image, ImageTk

# --- CONFIGURATION ---
ARDUINO_PORT = 'COM3'  # Update this to your Arduino's COM port
BAUD_RATE = 9600

# HSV Color Range for Toys (Replace with your calibrated values)
LOWER_COLOR = np.array([0, 120, 70])
UPPER_COLOR = np.array([10, 255, 255])
MIN_AREA = 500  

# Attempt hardware connection
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    print(f"Arduino connected on {ARDUINO_PORT}")
except Exception as e:
    arduino = None
    print(f"Warning: Arduino not connected on {ARDUINO_PORT}")

class AIEcoLightSwitcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AIEcoLightswitcher - Control Panel")
        self.geometry("850x700")
        ctk.set_appearance_mode("dark")

        # --- GUI ELEMENTS ---
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)

        # Controls Frame
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(pady=10, padx=20, fill="x")

        # Time Slider
        self.time_display = ctk.CTkLabel(self.controls_frame, text="Simulated Time: 12:00", font=("Arial", 16, "bold"))
        self.time_display.pack(pady=(10, 0))

        self.time_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=24, number_of_steps=24, command=self.update_time)
        self.time_slider.set(12)
        self.time_slider.pack(pady=10, padx=20, fill="x")

        # Override Buttons Frame
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=10)

        self.btn_auto = ctk.CTkButton(self.button_frame, text="AUTO MODE", fg_color="blue", command=self.set_mode_auto)
        self.btn_auto.grid(row=0, column=0, padx=10)

        self.btn_on = ctk.CTkButton(self.button_frame, text="FORCE ON", fg_color="green", command=self.set_mode_on)
        self.btn_on.grid(row=0, column=1, padx=10)

        self.btn_off = ctk.CTkButton(self.button_frame, text="FORCE OFF", fg_color="red", command=self.set_mode_off)
        self.btn_off.grid(row=0, column=2, padx=10)

        # Status Label
        self.status_label = ctk.CTkLabel(self, text="Status: Booting...", text_color="green", font=("Arial", 16, "bold"))
        self.status_label.pack(pady=10)

        # --- SYSTEM STATE ---
        self.cap = cv2.VideoCapture(0)
        self.current_time = 12
        self.school_start = 8
        self.school_end = 16
        
        self.mode = "AUTO"  # AUTO, FORCE_ON, FORCE_OFF
        self.state = "WAKE" 
        self.state_timer = time.time()
        self.frames_checked = 0
        self.frames_with_detection = 0

        # Start the loop
        self.update_loop()

    # --- UI CALLBACKS ---
    def update_time(self, value):
        self.current_time = int(value)
        self.time_display.configure(text=f"Simulated Time: {self.current_time:02d}:00")

    def set_mode_auto(self):
        self.mode = "AUTO"
        self.state = "WAKE"
        self.state_timer = time.time()

    def set_mode_on(self):
        self.mode = "FORCE_ON"
        self.send_command(True)

    def set_mode_off(self):
        self.mode = "FORCE_OFF"
        self.send_command(False)

    def send_command(self, state):
        if arduino:
            try:
                arduino.write(b'1' if state else b'0')
            except:
                pass # Fail silently if unplugged during runtime

    # --- MAIN LOGIC LOOP ---
    def update_loop(self):
        ret, frame = self.cap.read()
        
        # Hardware Fail-safe: No Camera Connected
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA NOT CONNECTED", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "UI Controls Still Active", (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        display_frame = frame.copy()

        # 1. Handle Manual Overrides First
        if self.mode == "FORCE_ON":
            self.status_label.configure(text="MANUAL OVERRIDE: LIGHTS ON", text_color="green")
            self.render_frame(display_frame)
            self.after(30, self.update_loop)
            return
            
        if self.mode == "FORCE_OFF":
            self.status_label.configure(text="MANUAL OVERRIDE: LIGHTS OFF", text_color="red")
            self.render_frame(display_frame)
            self.after(30, self.update_loop)
            return

        # 2. Enforce School Hours (Auto Mode Only)
        if not (self.school_start <= self.current_time < self.school_end):
            self.status_label.configure(text="SYSTEM SLEEP: OUT OF SCHOOL HOURS", text_color="red")
            self.send_command(False)
            
            # Apply visual grey-out
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (20, 20, 20), -1)
            display_frame = cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0)
            cv2.putText(display_frame, "OUT OF HOURS", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            self.render_frame(display_frame)
            self.after(30, self.update_loop)
            return

        # 3. Core OpenCV Processing Cycle (Only runs if camera works and in Auto mode)
        if ret:
            elapsed_time = time.time() - self.state_timer

            if self.state == "WAKE":
                self.status_label.configure(text="Status: EVALUATING CLASSROOM (1.5s window)", text_color="#00FF00")

                # Apply Color Masking
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detection_in_frame = False
                for cnt in contours:
                    if cv2.contourArea(cnt) > MIN_AREA:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        detection_in_frame = True

                self.frames_checked += 1
                if detection_in_frame:
                    self.frames_with_detection += 1

                # Transition to SLEEP
                if elapsed_time >= 1.5:
                    if self.frames_checked > 0 and self.frames_with_detection >= (self.frames_checked * 0.2): 
                        self.send_command(True)
                    else:
                        self.send_command(False)

                    self.state = "SLEEP"
                    self.state_timer = time.time()

            elif self.state == "SLEEP":
                self.status_label.configure(text="Status: IDLE (5s interval)", text_color="orange")
                
                # Apply visual grey-out effect during sleep phase
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (50, 50, 50), -1)
                display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
                cv2.putText(display_frame, "POWER SAVING SLEEP", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Transition back to WAKE
                if elapsed_time >= 5.0:
                    self.frames_checked = 0
                    self.frames_with_detection = 0
                    self.state = "WAKE"
                    self.state_timer = time.time()
        else:
            self.status_label.configure(text="Status: WAITING FOR CAMERA", text_color="orange")

        self.render_frame(display_frame)
        self.after(30, self.update_loop)

    def render_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

if __name__ == "__main__":
    app = AIEcoLightSwitcher()
    app.mainloop()