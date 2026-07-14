import cv2
from ultralytics import YOLO

# Load the nano model from your directory
model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(0)

print("Hold your dolls in front of the camera.")
print("Look at the bounding box to see what the AI thinks it is.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO inference
    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the class name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Draw the box and label on the frame
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Misclassification Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()