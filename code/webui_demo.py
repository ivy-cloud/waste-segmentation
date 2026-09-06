from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io
import tempfile
import shutil
# import magic  # pip install python-magic
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from llm_tools import UseLLM  # Ensure this is correctly implemented
import os
import uuid
import base64
from dotenv import load_dotenv  # pip install python-dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found in environment variables.")

# Initialize LLM tool
chatgpt = UseLLM(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Load YOLO models
# Uncomment the model you intend to use
# model = YOLO("yolo11n-seg.pt")  # Official model
model = YOLO("./runs/segment/train5/weights/best.pt")  # Custom model

# Define class names
classnames = ['bottle', 'carton', 'dixie_cup', 'paper']

# Define color mapping for classes
common_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0)   # Olive
]

num_classes = len(classnames)
colors = {i: common_colors[i % len(common_colors)] for i in range(num_classes)}

def get_file_type(file: UploadFile) -> str:
    """
    Get the file type (MIME type) of the uploaded file
    :param file:
    :return:
    """
    return file.content_type

# def get_file_type(file: UploadFile):
#     """
#     Determine the MIME type of the uploaded file using the magic library.
#     """
#     try:
#         mime = magic.Magic(mime=True)
        # file_content = file.file.read(1024)  # Read first 1024 bytes
        # file.file.seek(0)  # Reset file pointer after reading
        # file_type = mime.from_buffer(file_content)
        # return file_type
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"Error determining file type: {str(e)}")

def process_image(image_path: str, prompt: str):
    """
    Process an image using the YOLO model and generate a report.
    """
    #try:
    # Read image using OpenCV
    image = cv2.imread(image_path)
    new_width, new_height = 640, 640
    image = cv2.resize(image, (new_width, new_height))
    if image is None:
        raise ValueError("Failed to read the image.")

    results = model(image)

    class_counts = {}
    report_input = []

    for result in results:
        masks = result.masks.data.cpu().numpy()  # Shape: (num_objects, height, width)
        classes = result.boxes.cls.cpu().numpy()  # Class indices
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores

        for i, mask in enumerate(masks):
            class_idx = int(classes[i])
            if class_idx >= num_classes:
                continue  # Skip unknown classes

            mask_color = colors[class_idx]
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

            # Convert mask to binary
            mask = (mask > 0.5).astype(np.uint8)
            mask_colored = np.stack([
                mask * mask_color[0],
                mask * mask_color[1],
                mask * mask_color[2]
            ], axis=-1)

            # Overlay mask on image
            image = cv2.addWeighted(image, 1.0, mask_colored, 0.5, 0)

            # Draw label
            label = f"{classnames[class_idx]} {scores[i]:.2f}"
            bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)

    # Generate report based on class counts
    recyclables_count = 0
    unrecyclable_count = 0

    for class_idx, count in class_counts.items():
        message = f'{classnames[class_idx]}Is it recyclable waste? If yes, return True; if no, return False.'
        is_recyclable = chatgpt.use(message=message)
        if is_recyclable:
            recyclables_count += count
        else:
            unrecyclable_count += count
        report_input.append(f"{classnames[class_idx]}: {count}")

    # Add counts to the image
    start_x, start_y = 10, 30
    line_height = 25
    for idx, text in enumerate(report_input):
        color = colors[idx % num_classes]
        cv2.putText(image, text, (start_x, start_y + idx * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add recyclables and non-recyclables count
    cv2.putText(image, f"Recyclable: {recyclables_count}", (start_x, start_y + len(report_input) * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Non-recyclable: {unrecyclable_count}", (start_x, start_y + (len(report_input)+1) * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("segmented_output.jpg", image)

    # Convert to bytes
    pil_image = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Generate final report
    report_message = ','.join(report_input) + ('Please generate a report based on the above waste, '
                                               'and include waste disposal methods in the content.')
    report_text = chatgpt.use(report_message)

    response = {
        "recyclables_count": recyclables_count,
        "unrecyclable_count": unrecyclable_count,
        "report_input": report_input,
        "report_text": report_text
    }

    return img_byte_arr, response, report_text

    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

def process_video(video_path: str, prompt: str):
    """
    Process a video using the YOLO model and generate a report.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file.")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create a temporary file to save the processed video
        temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_video_fd)  # Close the file descriptor
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        class_counts = {}
        report_input = []

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for result in results:
                if not result.masks:
                    continue  # Skip if no masks

                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for i, mask in enumerate(masks):
                    class_idx = int(classes[i])
                    if class_idx >= num_classes:
                        continue  # Skip unknown classes

                    mask_color = colors[class_idx]
                    class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

                    # Convert mask to binary
                    mask = (mask > 0.5).astype(np.uint8)
                    mask_colored = np.stack([
                        mask * mask_color[0],
                        mask * mask_color[1],
                        mask * mask_color[2]
                    ], axis=-1)

                    # Ensure mask_colored matches frame size
                    if mask_colored.shape[:2] != frame.shape[:2]:
                        mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))

                    # Ensure mask_colored has 3 channels
                    if mask_colored.ndim == 2:
                        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)
                    elif mask_colored.shape[2] == 1:
                        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)

                    # Ensure matching dtypes
                    mask_colored = mask_colored.astype(frame.dtype)

                    # Add mask overlay to the frame
                    frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.5, 0)

                    # Draw label
                    label = f"{classnames[class_idx]} {scores[i]:.2f}"
                    bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)

            # Write the processed frame to the output video
            out.write(frame)
            frame_number += 1

        # Release resources
        cap.release()
        out.release()

        # Generate report based on class counts
        recyclables_count = 0
        unrecyclable_count = 0

        for class_idx, count in class_counts.items():
            message = f'{classnames[class_idx]}Is it recyclable waste? If yes, return True; if no, return False.'
            is_recyclable = chatgpt.use(message=message)
            if is_recyclable:
                recyclables_count += count
            else:
                unrecyclable_count += count
            report_input.append(f"{classnames[class_idx]}: {count}")

        # Generate final report
        report_message = ', '.join(report_input) + ('Please generate a report based on the above waste, '
                                                   'and include waste disposal methods in the content.')
        report_text = chatgpt.use(report_message)

        # Read the processed video
        with open(temp_video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Cleanup temporary video file
        os.remove(temp_video_path)

        response = {
            "recyclables_count": recyclables_count,
            "unrecyclable_count": unrecyclable_count,
            "report_input": report_input,
            "report_text": report_text
        }

        return video_bytes, response, report_text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),  # Image or video file
    text: str = Form(...)  # Input text
):
    """
    Endpoint to handle file uploads (images or videos), process them, and return results.
    """
    file_type = get_file_type(file)

    if "image" in file_type:
        media_type = "image"
    elif "video" in file_type:
        media_type = "video"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        if media_type == "image":
            processed_image_bytes, response_data, report_text = process_image(temp_file_path, text)

            # Encode image to base64
            encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')

            return JSONResponse(content={
                "message": "Image processed successfully",
                "text": text,
                "report_text": report_text,
                "processed_image": encoded_image  # Base64 encoded image
            })

        elif media_type == "video":
            processed_video_bytes, response_data, report_text = process_video(temp_file_path, text)

            # Encode video to base64 (Note: This can be very large)
            encoded_video = base64.b64encode(processed_video_bytes).decode('utf-8')

            return JSONResponse(content={
                "message": "Video processed successfully",
                "text": text,
                "report_text": report_text,
                "processed_video": encoded_video  # Base64 encoded video
            })

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    finally:
        # Delete the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)