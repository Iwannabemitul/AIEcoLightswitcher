import cv2
import time
import pyttsx3
from ultralytics import YOLO

# 1. Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# Optional: Adjust speech rate (default is usually around 200)
engine.setProperty('rate', 175) 

# Load the AI model
model = YOLO("yolov8n.pt") 

# Connect to the laptop's default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the laptop camera.")
    exit()

last_processed_time = time.time()
delay_seconds = 7

print("Camera and Audio system active. Waiting for the first interval...")

while True:
    # Continuously read frames to keep the buffer empty
    ret, frame = cap.read()
    if not ret:
        break
        
    current_time = time.time()

    # Only run the AI and announcements if 7 seconds have passed
    if (current_time - last_processed_time) >= delay_seconds:
        print("\n--- Running AI Check ---")
        
        results = model(frame)
        
        person_count = 0
        for box in results[0].boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.6:
                person_count += 1
                
        # 2. Execute conditional speech logic
        if person_count == 0:
            announcement = "Power Saving ON"
            print(f"Status: {announcement}")
            engine.say(announcement)
            engine.runAndWait()  # Blocks execution until the laptop finishes speaking
            
        else:
            if person_count == 1:
                announcement = "One person detected"
            else:
                announcement = f"{person_count} people detected"
                
            print(f"Status: {announcement}")
            engine.say(announcement)
            engine.runAndWait()
            
        # Reset the timer
        last_processed_time = current_time

    # Display the live window
    cv2.imshow("Laptop Camera Prototype", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()