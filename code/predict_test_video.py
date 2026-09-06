from ultralytics import YOLO
import cv2
from llm_tools import UseLLM
# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("./runs/detect/train/weights/best.pt")  # load a custom model


# Open the video file
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)


# Get the frame width, height, and FPS from the original video for saving output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the annotated video
output_video_path = "annotated_output.mp4"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


# Initialize the list to store the trajectory points
trajectory_points = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)
        detections = results[0].boxes
        for detection in detections:
            # Get the center of the bounding box (x, y)
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # Add the center point to the trajectory
            trajectory_points.append((center_x, center_y))

            # Draw a circle at the center of the ball (for visualization)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Draw the trajectory (lines connecting all points)
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, 0, 255), 2)

        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        out.write(frame)
        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()