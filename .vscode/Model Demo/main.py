import cv2
import numpy as np
import serial
import time
import customtkinter as ctk
from PIL import Image, ImageTk

# --- CONFIGURATION ---
ARDUINO_PORT = 'COM3'  # Update this to your Arduino's COM port
BAUD_RATE = 9600

# HSV Color Range for Toys (Example: Bright Red)
LOWER_COLOR = np.array([0, 120, 70])
UPPER_COLOR = np.array([10, 255, 255])
MIN_AREA = 500  # Minimum pixel area to be considered a toy

# Attempt hardware connection
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
except Exception as e:
    arduino = None
    print(f"Warning: Arduino not connected on {ARDUINO_PORT}")

class AIEcoLightSwitcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AIEcoLightswitcher - Control Panel")
        self.geometry("800x650")
        ctk.set_appearance_mode("dark")

        # --- GUI ELEMENTS ---
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)

        self.time_slider = ctk.CTkSlider(self, from_=0, to=24, number_of_steps=24, command=self.update_time)
        self.time_slider.set(12)
        self.time_slider.pack(pady=10)

        self.time_display = ctk.CTkLabel(self, text="Simulated Time: 12:00", font=("Arial", 16, "bold"))
        self.time_display.pack()

        self.status_label = ctk.CTkLabel(self, text="Status: Booting...", text_color="green", font=("Arial", 14))
        self.status_label.pack(pady=10)

        # --- SYSTEM STATE ---
        self.cap = cv2.VideoCapture(0)
        self.current_time = 12
        self.school_start = 8
        self.school_end = 16
        
        self.state = "WAKE" 
        self.state_timer = time.time()
        self.frames_checked = 0
        self.frames_with_detection = 0

        # Start the loop
        self.update_loop()

    def update_time(self, value):
        self.current_time = int(value)
        self.time_display.configure(text=f"Simulated Time: {self.current_time:02d}:00")

    def send_command(self, state):
        if arduino:
            arduino.write(b'1' if state else b'0')

    def update_loop(self):
        # 1. Enforce School Hours
        if not (self.school_start <= self.current_time < self.school_end):
            self.status_label.configure(text="SYSTEM SLEEP: OUT OF SCHOOL HOURS", text_color="red")
            self.send_command(False)
            
            # Show blank/sleeping screen
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "OUT OF HOURS", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            self.render_frame(blank_frame)
            
            self.after(100, self.update_loop)
            return

        # 2. Core Processing Cycle (Wake/Sleep)
        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self.update_loop)
            return

        display_frame = frame.copy()
        elapsed_time = time.time() - self.state_timer

        if self.state == "WAKE":
            self.status_label.configure(text="Status: EVALUATING CLASSROOM (1.5s window)", text_color="green")

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

            # Transition to SLEEP after 1.5 seconds
            if elapsed_time >= 1.5:
                # If a toy was found in at least 20% of the checked frames, turn lights ON
                if self.frames_with_detection >= (self.frames_checked * 0.2): 
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

            # Transition back to WAKE after 5 seconds
            if elapsed_time >= 5.0:
                self.frames_checked = 0
                self.frames_with_detection = 0
                self.state = "WAKE"
                self.state_timer = time.time()

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