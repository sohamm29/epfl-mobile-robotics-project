import cv2
from ultralytics import YOLO


def global_vision():

    # define model and camera IO
    model = YOLO("src/yolo11s_bmr_weights.pt") 
    cap = cv2.VideoCapture(0)

    # ensure camera available
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from the camera.")
            break

        # run YOLO inference on the current frame
        results = model(frame)

        # draw the custom bounding boxes onto the frame in real time
        annotated_frame = results[0].plot()

        # display the video feed
        cv2.imshow("Custom YOLO Live", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # close program
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    global_vision()