import cv2
import numpy as np
import serial
import time
import tkinter as tk
import customtkinter as ctk
from PIL import Image

# --- CONFIGURATION ---
ARDUINO_PORT = 'COM3'  
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

# --- TOOLTIP CLASS ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True) 
        self.tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#2b2b2b", foreground="#e0e0e0", relief='solid', borderwidth=1,
                       font=("Arial", 11, "normal"), padx=8, pady=4)
        label.pack(ipadx=1)

    def leave(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None

class AIEcoLightSwitcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AIEcoLightswitcher - AI Vision Dashboard")
        self.geometry("1050x550")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(0, weight=1)  
        self.grid_columnconfigure(1, weight=0)  
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT: VIDEO FRAME ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.grid(row=0, column=0)

        # --- RIGHT: SIDEBAR CONTROLS ---
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=10)
        self.sidebar_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)

        self.title_label = ctk.CTkLabel(self.sidebar_frame, text="Control Panel", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(20, 30))

        # Status Display
        self.status_header = ctk.CTkLabel(self.sidebar_frame, text="SYSTEM STATUS", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.status_header.pack(anchor="w", padx=20)
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Booting...", text_color="#2ECC71", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(anchor="w", padx=20, pady=(0, 30))

        # Time Simulator
        self.time_header = ctk.CTkLabel(self.sidebar_frame, text="TIME SIMULATOR", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.time_header.pack(anchor="w", padx=20)

        self.time_display = ctk.CTkLabel(self.sidebar_frame, text="12:00 PM (School Hours)", font=ctk.CTkFont(size=16))
        self.time_display.pack(anchor="w", padx=20, pady=(5, 5))

        self.time_slider = ctk.CTkSlider(self.sidebar_frame, from_=0, to=24, number_of_steps=24, command=self.update_time)
        self.time_slider.set(12)
        self.time_slider.pack(pady=10, padx=20, fill="x")
        ToolTip(self.time_slider, "Drag to simulate time.\n08:00 AM - 03:59 PM = School Hours\n04:00 PM - 07:59 AM = Power Sleep")

        # Manual Overrides
        self.override_header = ctk.CTkLabel(self.sidebar_frame, text="SYSTEM OVERRIDE", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.override_header.pack(anchor="w", padx=20, pady=(30, 10))

        self.btn_auto = ctk.CTkButton(self.sidebar_frame, text="AUTO MODE", fg_color="#2980B9", hover_color="#3498DB", command=self.set_mode_auto)
        self.btn_auto.pack(pady=5, padx=20, fill="x")
        ToolTip(self.btn_auto, "Hands-free operation.\nLights are controlled by AI/CV detection\nduring active school hours.")

        self.btn_on = ctk.CTkButton(self.sidebar_frame, text="FORCE ON", fg_color="#27AE60", hover_color="#2ECC71", command=self.set_mode_on)
        self.btn_on.pack(pady=5, padx=20, fill="x")
        ToolTip(self.btn_on, "Manual Override: ON\nBypasses AI and forces Arduino to\nkeep the lights turned ON.")

        self.btn_off = ctk.CTkButton(self.sidebar_frame, text="FORCE OFF", fg_color="#C0392B", hover_color="#E74C3C", command=self.set_mode_off)
        self.btn_off.pack(pady=5, padx=20, fill="x")
        ToolTip(self.btn_off, "Manual Override: OFF\nBypasses AI and forces Arduino to\nkeep the lights turned OFF.")

        # --- SYSTEM STATE ---
        self.cap = cv2.VideoCapture(0)
        self.current_time = 12
        self.school_start = 8   # 8:00 AM
        self.school_end = 16    # 4:00 PM
        
        self.mode = "AUTO" 
        self.system_state = "WAKE" 
        self.state_timer = time.time()
        self.frames_checked = 0
        self.frames_with_detection = 0

        # Start the loop
        self.update_loop()

    # --- UI CALLBACKS ---
    def format_ampm(self, hour):
        """Converts 24hr integer to AM/PM string format"""
        if hour == 0 or hour == 24:
            return "12:00 AM"
        elif hour < 12:
            return f"{hour:02d}:00 AM"
        elif hour == 12:
            return "12:00 PM"
        else:
            return f"{(hour - 12):02d}:00 PM"

    def update_time(self, value):
        self.current_time = int(value)
        
        # Check against 24hr logic for system rules
        if self.school_start <= self.current_time < self.school_end:
            status_txt = "(School Hours)"
        else:
            status_txt = "(Out of Hours)"
            
        # Display as AM/PM
        display_time = self.format_ampm(self.current_time)
        self.time_display.configure(text=f"{display_time} {status_txt}")

    def set_mode_auto(self):
        self.mode = "AUTO"
        self.system_state = "WAKE"
        self.state_timer = time.time()

    def set_mode_on(self):
        self.mode = "FORCE_ON"
        self.send_command(True)

    def set_mode_off(self):
        self.mode = "FORCE_OFF"
        self.send_command(False)

    def send_command(self, status):
        if arduino:
            try:
                arduino.write(b'1' if status else b'0')
            except:
                pass 

    # --- MAIN LOGIC LOOP ---
    def update_loop(self):
        ret, frame = self.cap.read()
        
        # Hardware Fail-safe
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA NOT CONNECTED", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "UI Controls Still Active", (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        display_frame = frame.copy()

        # 1. Manual Overrides
        if self.mode == "FORCE_ON":
            self.status_label.configure(text="OVERRIDE: ON", text_color="#2ECC71")
            self.render_frame(display_frame)
            self.after(30, self.update_loop)
            return
            
        if self.mode == "FORCE_OFF":
            self.status_label.configure(text="OVERRIDE: OFF", text_color="#E74C3C")
            self.render_frame(display_frame)
            self.after(30, self.update_loop)
            return

        # 2. Enforce School Hours
        if not (self.school_start <= self.current_time < self.school_end):
            self.status_label.configure(text="SLEEP: OUT OF HOURS", text_color="#E74C3C")
            self.send_command(False)
            
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (20, 20, 20), -1)
            display_frame = cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0)
            cv2.putText(display_frame, "SYSTEM SLEEP", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            self.render_frame(display_frame)
            self.after(30, self.update_loop)
            return

        # 3. Core Processing Cycle 
        if ret:
            elapsed_time = time.time() - self.state_timer

            if self.system_state == "WAKE":
                self.status_label.configure(text="EVALUATING (1.5s)", text_color="#2ECC71")

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

                if elapsed_time >= 1.5:
                    if self.frames_checked > 0 and self.frames_with_detection >= (self.frames_checked * 0.2): 
                        self.send_command(True)
                    else:
                        self.send_command(False)

                    self.system_state = "SLEEP"
                    self.state_timer = time.time()

            elif self.system_state == "SLEEP":
                self.status_label.configure(text="IDLE (5s interval)", text_color="#F39C12")
                
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (50, 50, 50), -1)
                display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
                cv2.putText(display_frame, "POWER SAVING SLEEP", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if elapsed_time >= 5.0:
                    self.frames_checked = 0
                    self.frames_with_detection = 0
                    self.system_state = "WAKE"
                    self.state_timer = time.time()
        else:
            self.status_label.configure(text="NO CAMERA FEED", text_color="#E74C3C")

        self.render_frame(display_frame)
        self.after(30, self.update_loop)

    def render_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img  

if __name__ == "__main__":
    app = AIEcoLightSwitcher()
    app.mainloop()