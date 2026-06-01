import cv2
import time
from ultralytics import YOLO

# Load the AI model
model = YOLO("yolov8n.pt") 

# 1. Connect to the laptop's default camera (Device 0)
# We keep this OPEN outside the loop so the camera stays warmed up
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the laptop camera. Check permissions in your OS.")
    exit()

# Track the last time we ran the AI
last_processed_time = time.time()
delay_seconds = 7

print("Camera active. Waiting for the first interval...")

while True:
    # Continuously read frames to keep the buffer empty and feed live
    ret, frame = cap.read()
    if not ret:
        break
        
    current_time = time.time()

    # 2. Only run the AI if 7 seconds have passed
    if (current_time - last_processed_time) >= delay_seconds:
        print("\n--- Running AI Check ---")
        
        # Pass the current frame to YOLO
        results = model(frame)
        
        person_count = 0
        for box in results[0].boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.6:
                person_count += 1
                
        if person_count == 0:
            print("Status: Empty room detected.")
        else:
            print(f"Status: {person_count} person(s) detected.")
            
        # Reset the timer
        last_processed_time = current_time

    # 3. Show the live camera feed on your screen so you can see what's happening
    # (Optional, but highly recommended for debugging)
    cv2.imshow("Laptop Camera Prototype", frame)
    
    # Press 'q' on your keyboard to gracefully shut everything down
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up hardware resources
cap.release()
cv2.destroyAllWindows()