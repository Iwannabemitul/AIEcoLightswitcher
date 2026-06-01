import cv2
import time
from ultralytics import YOLO

# Load the AI model (the 'n' version is lightweight and fast)
model = YOLO("yolov8n.pt") 

# Replace with your camera's RTSP or snapshot URL
camera_url = "rtsp://admin:password@192.168.1.100:554/stream"

while True:
    print("Grabbing snapshot...")
    
    # 1. Open connection, grab ONE frame, and immediately close it
    # This prevents the buffer from filling up in the background
    cap = cv2.VideoCapture(camera_url)
    ret, frame = cap.read()
    cap.release() 

    if ret:
        # 2. Run the AI on that single frame
        results = model(frame)
        
        # Tally up the detections
        person_count = 0
        for box in results[0].boxes:
             # Class 0 in the standard COCO dataset is 'person'
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.6:
                person_count += 1
        
        # 3. Execute your conditional logic
        if person_count == 0:
            print("Status: Classroom is empty. Triggering power-saving mode.")
            # e.g., turn_off_smart_lights()
            
        elif person_count > 0 and person_count <= 5:
            print(f"Status: {person_count} people detected. Likely after-hours study.")
            
        elif person_count > 40:
            print(f"Status: {person_count} people detected. Capacity warning.")
            # e.g., send_slack_alert()
            
    else:
        print("Failed to pull frame from camera.")

    # 4. The Delay
    print("Sleeping for 7 seconds...\n")
    time.sleep(7)