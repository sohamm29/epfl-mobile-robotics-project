import cv2
from ultralytics import YOLO


def global_vision():
    model = YOLO("mob_rob_yolo.pt") 
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from the camera.")
            break

        # 3. Run YOLO inference on the current frame
        results = model(frame)

        # 4. Draw the custom bounding boxes onto the frame
        annotated_frame = results[0].plot()

        # 5. Display the video feed
        cv2.imshow("Custom YOLO Live", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    global_vision()