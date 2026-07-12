import cv2
import numpy as np

def nothing(x):
    pass

# Create a window with trackbars to adjust HSV values live
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cap = cv2.VideoCapture(0)

print("Place your toy in the camera view.")
print("Adjust the trackbars until ONLY the toy is white in the 'Mask' window.")
print("Press 'q' to quit and get your values.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read current trackbar positions
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Apply the mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original Feed", frame)
    cv2.imshow("Mask (Aim for solid white toy, black background)", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n--- COPY THESE VALUES INTO main.py ---")
        print(f"LOWER_COLOR = np.array([{l_h}, {l_s}, {l_v}])")
        print(f"UPPER_COLOR = np.array([{u_h}, {u_s}, {u_v}])")
        print("--------------------------------------\n")
        break

cap.release()
cv2.destroyAllWindows()