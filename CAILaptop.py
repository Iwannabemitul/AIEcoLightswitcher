import cv2
import time
import datetime
import subprocess
import winsound
import numpy as np
from ultralytics import YOLO

def speak(text):
    safe_text = text.replace('"', "'")
    ps_command = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{safe_text}")'
    subprocess.Popen(["powershell", "-Command", ps_command], creationflags=subprocess.CREATE_NO_WINDOW)

# --- SYSTEM INITIALIZATION ---
model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the laptop camera.")
    exit()

# --- BUILD NIGHT VISION FILTER (Gamma Correction LUT) ---
# Gamma < 1.0 shifts the image towards lighter values. 0.4 is aggressive night vision.
gamma_value = 0.4 
invGamma = 1.0 / gamma_value
# Build a 256-element array mapping every possible dark pixel to a brighter value
night_vision_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# --- SYSTEM CONFIGURATION ---
CYCLE_DELAY = 7        
SNIPPET_DURATION = 2   
next_capture_time = time.time()

print("==================================================================")
print("System Active: V5 with Digital Night Vision & Calibrated Tamper")
print("==================================================================")

while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("AI Eco Light Switcher Prototype", frame)
    
    if current_time >= next_capture_time:
        print(f"\n=== STARTING 2-SECOND VIDEO SNIPPET CAPTURE @ {datetime.datetime.now().strftime('%H:%M:%S')} ===")
        
        snippet_frames = []
        record_start = time.time()
        
        while time.time() - record_start < SNIPPET_DURATION:
            ret, s_frame = cap.read()
            if ret:
                snippet_frames.append(s_frame)
                cv2.imshow("AI Eco Light Switcher Prototype", s_frame)
                cv2.waitKey(1)
        
        if len(snippet_frames) > 1:
            brightness_list = []
            frame_variances = []
            yolo_person_counts = []
            pixel_changes_over_time = []
            
            prev_gray = cv2.cvtColor(snippet_frames[0], cv2.COLOR_BGR2GRAY)
            
            # --- METRICS PASS ---
            for f in snippet_frames:
                brightness_list.append(f.mean())
                
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                frame_variances.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                
                frame_diff = cv2.absdiff(gray, prev_gray)
                pixel_changes_over_time.append(frame_diff.mean())
                prev_gray = gray

            avg_snippet_brightness = np.mean(brightness_list)
            avg_snippet_texture = np.mean(frame_variances)
            avg_temporal_motion = np.mean(pixel_changes_over_time)
            
            # --- AI INFERENCE PASS WITH CONDITIONAL NIGHT VISION ---
            is_dark_room = avg_snippet_brightness < 45.0
            
            for idx, f in enumerate(snippet_frames):
                if idx % 5 == 0:
                    # If the room is dark, apply the mathematical night vision filter
                    if is_dark_room:
                        frame_for_ai = cv2.LUT(f, night_vision_table)
                    else:
                        frame_for_ai = f

                    results = model(frame_for_ai, verbose=False)
                    current_frame_people = 0
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.6:
                            current_frame_people += 1
                    yolo_person_counts.append(current_frame_people)

            max_people_seen = max(yolo_person_counts) if yolo_person_counts else 0
            
            print("-" * 60)
            print("[DEBUG] SNIPPET BATCH METRICS:")
            print(f"  -> Avg Brightness over 2s : {avg_snippet_brightness:.2f}")
            print(f"  -> Avg Texture (Spatial)  : {avg_snippet_texture:.2f}")
            print(f"  -> Avg Motion (Temporal)  : {avg_temporal_motion:.4f}")
            print(f"  -> Night Vision Active    : {'YES' if is_dark_room else 'NO'}")
            print(f"  -> Max Occupants Detected : {max_people_seen}")
            print("-" * 60)

            # --- DECISION LOGIC ---
            
            # Calibrated based directly on your hardware logs
            if avg_snippet_brightness < 45.0 and avg_temporal_motion < 0.60 and avg_snippet_texture < 4.5:
                print(">>> FINAL UTTERANCE: TAMPERED (Lens Obstruction Confirmed) <<<")
                speak("Warning. Camera tampering detected.")
                for _ in range(3):
                    winsound.Beep(1000, 400)
                    winsound.Beep(700, 400)
            
            else:
                current_clock_time = datetime.datetime.now().strftime("%I:%M %p")
                
                if max_people_seen == 0:
                    announcement = f"0 persons present at the current given time, {current_clock_time}. Power Saving ON."
                elif max_people_seen == 1:
                    announcement = f"1 person present at the current given time, {current_clock_time}."
                else:
                    announcement = f"{max_people_seen} people present at the current given time, {current_clock_time}."
                    
                print(f"Status: {announcement}")
                speak(announcement)

        next_capture_time = time.time() + CYCLE_DELAY

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()