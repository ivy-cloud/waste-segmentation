import random
import tempfile
import time
import os
import cv2
import numpy as np
import streamlit as st
from QtFusion.path import abs_path
from QtFusion.utils import drawRectBox

from log import ResultLogger, LogTable
from model import Web_Detector
from chinese_name_list import Label_list
from ui_style import def_css_hitml
from utils import save_uploaded_file, concat_results, load_default_image, get_camera_names
import tempfile
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

import numpy as np
import cv2
from hashlib import md5

def calculate_polygon_area(points):
    # Compute polygon area.
    return cv2.contourArea(points.astype(np.float32))

def draw_with_chinese(img, text, position, font_size):
    # Draw text on an image.
    # Implementation can be adjusted as needed.
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(img, text, position, font, font_size, color, thickness, cv2.LINE_AA)
    return img

def generate_color_based_on_name(name):
    # Generate a stable color from a name hash.
    hash_object = md5(name.encode())
    hex_color = hash_object.hexdigest()[:6]  # First 6 hex digits
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR

def draw_with_chinese(image, text, position, font_size=20, color=(255, 0, 0)):
    """
    Draw text on an OpenCV image using a TrueType font.
    """
    # Convert the image from OpenCV BGR to PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    # Use the specified font
    font = ImageFont.truetype("simsun.ttc", font_size, encoding="unic")
    draw.text(position, text, font=font, fill=color)
    # Convert the image from PIL RGB back to OpenCV BGR
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def adjust_parameter(image_size, base_size=1000):
    # Compute a scale factor from the image's longest side
    max_size = max(image_size)
    return max_size / base_size

def adjust_parameter(image_size, base_size=1000):
    max_size = max(image_size)
    return max_size / base_size


def draw_detections(image, info, alpha=0.2):
    name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']
    adjust_param = adjust_parameter(image.shape[:2])
    spacing = int(20 * adjust_param)

    if mask is None:
        x1, y1, x2, y2 = bbox
        aim_frame_area = (x2 - x1) * (y2 - y1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=int(3 * adjust_param))
        image = draw_with_chinese(image, name, (x1, y1 - int(30 * adjust_param)), font_size=int(35 * adjust_param))
        y_offset = int(50 * adjust_param)  # Draw the class name above; leave space below
    else:
        mask_points = np.concatenate(mask)
        aim_frame_area = calculate_polygon_area(mask_points)
        mask_color = generate_color_based_on_name(name)
        try:
            overlay = image.copy()
            cv2.fillPoly(overlay, [mask_points.astype(np.int32)], mask_color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.drawContours(image, [mask_points.astype(np.int32)], -1, (0, 0, 255), thickness=int(8 * adjust_param))

            # Compute area, perimeter, and circularity
            area = cv2.contourArea(mask_points.astype(np.int32))
            perimeter = cv2.arcLength(mask_points.astype(np.int32), True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Sample color
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [mask_points.astype(np.int32)], -1, 255, -1)
            color_points = cv2.findNonZero(mask)
            selected_points = color_points[np.random.choice(color_points.shape[0], 5, replace=False)]
            colors = np.mean([image[y, x] for x, y in selected_points[:, 0]], axis=0)
            color_str = f"({colors[0]:.1f}, {colors[1]:.1f}, {colors[2]:.1f})"

            # Draw the class name
            x, y = np.min(mask_points, axis=0).astype(int)
            image = draw_with_chinese(image, name, (x, y - int(30 * adjust_param)), font_size=int(35 * adjust_param))
            y_offset = int(50 * adjust_param)  # Draw the class name above; leave space below

            # Draw area, perimeter, circularity, and color
            # metrics = [("Area", area), ("Perimeter", perimeter), ("Circularity", circularity), ("Color", color_str)]
            # for idx, (metric_name, metric_value) in enumerate(metrics):
            #     text = f"{metric_name}: {metric_value}"
            #     image = draw_with_chinese(image, text, (x, y - y_offset - spacing * (idx + 1)),
            #                               font_size=int(35 * adjust_param))

        except Exception as e:
            print(f"An error occurred: {e}")

    return image, aim_frame_area

def calculate_polygon_area(points):
    """
    Compute polygon area. Input should be an Nx2 numpy array of vertex coordinates.
    """
    if len(points) < 3:  # A polygon needs at least 3 vertices
        return 0
    return cv2.contourArea(points)

def format_time(seconds):
    # Compute hours, minutes, and seconds
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    # Format as a string
    return "{:02}:{:02}:{:02}".format(int(hrs), int(mins), int(secs))



def save_chinese_image(file_path, image_array):
    """
    Save an image that may contain Unicode characters in the path.

    Args:
    file_path (str): Image save path, which may include Unicode characters
    image_array (numpy.ndarray): OpenCV image to save (numpy array)
    """
    try:
        # Convert the OpenCV image to a Pillow Image
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

        # Save the image with Pillow
        image.save(file_path)

        print(f"Saved image to: {file_path}")
    except Exception as e:
        print(f"Failed to save image: {str(e)}")

class Detection_UI:
    """
    Detection system class.

    Attributes:
        model_type (str): Model type.
        conf_threshold (float): Confidence threshold.
        iou_threshold (float): IoU threshold.
        selected_camera (str): Selected camera.
        file_type (str): File type.
        uploaded_file (FileUploader): Uploaded file.
        detection_result (str): Detection result.
        detection_location (str): Detection location.
        detection_confidence (str): Detection confidence.
        detection_time (str): Detection time.
    """

    def __init__(self):
        """
        Initialize detection system parameters.
        """
        # Initialize class labels and random colors per class
        self.cls_name = Label_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.cls_name))]

        # Set the page title
        self.title = "Waste Segmentation"
        self.setup_page()  # Initialize page layout
        def_css_hitml()  # Apply CSS styles

        # Initialize detection parameters
        self.model_type = None
        self.conf_threshold = 0.15  # Default confidence threshold
        self.iou_threshold = 0.5  # Default IoU threshold

        # Initialize camera and file variables
        self.selected_camera = None
        self.file_type = None
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file = None  # Custom model file

        # Initialize detection-result variables
        self.detection_result = None
        self.detection_location = None
        self.detection_confidence = None
        self.detection_time = None

        # Initialize UI display variables
        self.display_mode = None  # Display mode
        self.close_flag = None  # Flag to stop image display
        self.close_placeholder = None  # Stop-button area
        self.image_placeholder = None  # Image display area
        self.image_placeholder_res = None  # Image display area
        self.table_placeholder = None  # Table display area
        self.log_table_placeholder = None  # Full results table area
        self.selectbox_placeholder = None  # Selectbox area
        self.selectbox_target = None  # Selected selectbox item
        self.progress_bar = None  # Progress bar

        # Initialize FPS and video time
        self.FPS = 30
        self.timenow = 0

        # Initialize log save path
        self.saved_log_data = abs_path("tempDir/log_table_data.csv", path_type="current")

        # Create a LogTable in session state if missing
        if 'logTable' not in st.session_state:
            st.session_state['logTable'] = LogTable(self.saved_log_data)

        # Get or update the available camera list
        if 'available_cameras' not in st.session_state:
            st.session_state['available_cameras'] = get_camera_names()
        self.available_cameras = st.session_state['available_cameras']

        # Get or create the results table
        self.logTable = st.session_state['logTable']

        # Load or create the model instance
        if 'model' not in st.session_state:
            st.session_state['model'] = Web_Detector()  # Create a Detector instance

        self.model = st.session_state['model']
        # Load trained model weights
        self.model.load_model(model_path=abs_path("weights/yolov8s.pt", path_type="current"))
        # Reassign colors for model classes
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.model.names))]
        self.setup_sidebar()  # Initialize sidebar layout

    def setup_page(self):
        # Set page layout
        # st.set_page_config(
        #     page_title=self.title,
        #     page_icon="REC",
        #     initial_sidebar_state="expanded"
        # )

        # Center the title
        st.markdown(
            f'<h1 style="text-align: center;">{self.title}</h1>',
            unsafe_allow_html=True
        )

    def setup_sidebar(self):
        """
        Set up the Streamlit sidebar.

        Configure model settings, camera selection, and file recognition options in the sidebar.
        """
        # Confidence threshold slider
        self.conf_threshold = float(st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.15))
        # IoU threshold slider
        self.iou_threshold = float(st.sidebar.slider("IOU", min_value=0.0, max_value=1.0, value=0.25))
        # Sidebar model-settings section
        st.sidebar.header("Task Settings")
        # Model-type dropdown
        self.model_type = st.sidebar.selectbox("Select task type", ["Default task"])
        if self.model_type == "Default task":
            self.model.load_model(model_path=abs_path("weights/yolov8n.pt", path_type="current"))
        elif self.model_type == "Task 1":
            self.model.load_model(model_path=abs_path("weights/yolov8n.pt", path_type="current"))
        elif self.model_type == "Task 2":
            self.model.load_model(model_path=abs_path("weights/yolov8n.pt", path_type="current"))

        # Choose default or custom model weights
        model_file_option = st.sidebar.radio("Model Settings", ["Default", "Custom weights file"])
        if model_file_option == "Custom weights file":
            # Show a file uploader when custom weights are selected
            model_file = st.sidebar.file_uploader("Select .pt file", type="pt")

            # Save and load the model if a weights file was uploaded
            if model_file is not None:
                self.custom_model_file = save_uploaded_file(model_file)
                self.model.load_model(model_path=self.custom_model_file)
                self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                               range(len(self.model.names))]
        elif model_file_option == "Default":
            self.model.load_model(model_path=abs_path("weights/yolov8s-seg.pt", path_type="current"))
            # Reassign colors for model classes
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                           range(len(self.model.names))]



        # Sidebar camera-settings section
        st.sidebar.header("Live Camera Inspection")
        # Camera dropdown
        self.selected_camera = st.sidebar.selectbox("Select camera", self.available_cameras)

        # Sidebar file-recognition section
        st.sidebar.header("File Recognition Settings")
        # File-type dropdown
        self.file_type = st.sidebar.selectbox("Select file type", ["Image file", "Video file"])
        # Show an uploader matching the selected file type
        if self.file_type == "Image file":
            self.uploaded_file = st.sidebar.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        elif self.file_type == "Video file":
            self.uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4"])

        # Show hints based on camera and file-type selection
        if self.selected_camera == "Camera off":
            if self.file_type == "Image file":
                st.sidebar.write("Select an image and click Start Detection.")
            if self.file_type == "Video file":
                st.sidebar.write("Select a video and click Start Detection.")
        else:
            st.sidebar.write("Click Start Detection to run the camera.")

    def load_model_file(self):
        if self.custom_model_file:
            self.model.load_model(self.custom_model_file)
        else:
            pass  # Load

    def process_camera_or_file(self):
        """
        Process camera or file input.

        Process and display detection results from the selected source (camera, image, or video).
        """
        # If a camera is selected
        if self.selected_camera != "Camera off":
            self.logTable.clear_frames()  # Clear previous frames
            # Create a stop button
            self.close_flag = self.close_placeholder.button(label="Stop")

            # Capture frames with OpenCV
            if str(self.selected_camera) == '0':
                camera_id = 0
            else:
                camera_id = self.selected_camera

            cap = cv2.VideoCapture(camera_id)

            self.uploaded_video = None

            fps = cap.get(cv2.CAP_PROP_FPS)

            self.FPS = fps

            # Cap total frames at 1000
            total_frames = 1000
            current_frame = 0
            self.progress_bar.progress(0)  # Initialize the progress bar

            try:
                if len(self.selected_camera) < 8:
                    camera_id = int(self.selected_camera)
                else:
                    camera_id = self.selected_camera

                cap = cv2.VideoCapture(camera_id)

                # Get the frame rate
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.FPS = fps

                # Create the progress bar
                self.progress_bar.progress(0)

                # Prepare save paths
                camera_savepath = './tempDir/camera'
                if not os.path.exists(camera_savepath):
                    os.makedirs(camera_savepath)
                # ret, frame = cap.read()
                # height, width, layers = frame.shape
                # size = (width, height)
                #
                # file_name = abs_path('tempDir/camera.avi', path_type="current")
                # out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

                while cap.isOpened() and not self.close_flag:
                    ret, frame = cap.read()
                    if ret:
                        # Adjust camera resolution
                        # Set the new size
                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        # Resize the image
                        frame = cv2.resize(frame, (new_width, new_height))


                        framecopy = frame.copy()
                        image, detInfo, _ = self.frame_process(frame, 'camera')

                        # Save detection result images
                        if detInfo:
                            file_name = abs_path(camera_savepath + '/' + str(current_frame + 1) + '.jpg', path_type="current")
                            save_chinese_image(file_name, image)
                        #
                        # # Save detection result video
                        # out.write(image)

                        # Set the new size
                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        # Resize the image
                        resized_image = cv2.resize(image, (new_width, new_height))
                        resized_frame = cv2.resize(framecopy, (new_width, new_height))
                        if self.display_mode == "Overlay":
                            self.image_placeholder.image(resized_image, channels="BGR", caption="Video")
                        else:
                            self.image_placeholder.image(resized_frame, channels="BGR", caption="Original")
                            self.image_placeholder_res.image(resized_image, channels="BGR", caption="Detection")

                        self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                        # Update the progress bar
                        progress_percentage = int((current_frame / total_frames) * 100)
                        self.progress_bar.progress(progress_percentage)
                        current_frame = (current_frame + 1) % total_frames  # Reset the progress bar
                    else:
                        break
                if self.close_flag:
                    self.logTable.save_to_csv()
                    self.logTable.update_table(self.log_table_placeholder)
                    cap.release()
                    # out.release()

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()
                # out.release()


            finally:

                if self.uploaded_video is None:
                    name_in = None
                else:
                    name_in = self.uploaded_video.name

                res = self.logTable.save_frames_file(fps=self.FPS, video_name=name_in)
                st.write("Recognition results saved to: " + self.saved_log_data)
                if res:
                    st.write(f"Target result file saved: {res}")


        else:
            # If an image was uploaded
            if self.uploaded_file is not None:
                self.logTable.clear_frames()
                self.progress_bar.progress(0)
                # Display the uploaded image
                source_img = self.uploaded_file.read()
                file_bytes = np.asarray(bytearray(source_img), dtype=np.uint8)
                image_ini = cv2.imdecode(file_bytes, 1)
                framecopy = image_ini.copy()
                image, detInfo, select_info = self.frame_process(image_ini, self.uploaded_file.name)
                save_chinese_image('./tempDir/' + self.uploaded_file.name, image)
                # self.selectbox_placeholder = st.empty()
                # self.selectbox_target = self.selectbox_placeholder.selectbox("Target filter", select_info, key="22113")

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)  # Update the full results table

                # Set the new size
                new_width = 1080
                new_height = int(new_width * (9 / 16))
                # Resize the image
                resized_image = cv2.resize(image, (new_width, new_height))
                resized_frame = cv2.resize(framecopy, (new_width, new_height))
                if self.display_mode == "Overlay":
                    self.image_placeholder.image(resized_image, channels="BGR", caption="Image")
                else:
                    self.image_placeholder.image(resized_frame, channels="BGR", caption="Original")
                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="Detection")

                self.logTable.add_frames(image, detInfo, cv2.resize(image_ini, (640, 640)))
                self.progress_bar.progress(100)

            # If a video was uploaded
            elif self.uploaded_video is not None:
                # Process the uploaded video
                self.logTable.clear_frames()
                self.close_flag = self.close_placeholder.button(label="Stop")

                video_file = self.uploaded_video
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                try:
                    tfile.write(video_file.read())
                    tfile.flush()

                    tfile.seek(0)  # Rewind the file pointer

                    cap = cv2.VideoCapture(tfile.name)

                    # Get total frame count and FPS
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    self.FPS = fps
                    # Compute video duration in seconds
                    total_length = total_frames / fps if fps > 0 else 0
                    print('Video duration: ' + str(total_length)[:4] + 's')
                    # Create the progress bar
                    self.progress_bar.progress(0)

                    current_frame = 0

                    # Prepare save paths
                    video_savepath = './tempDir/' + self.uploaded_video.name
                    if not os.path.exists(video_savepath):
                        os.makedirs(video_savepath)
                    # ret, frame = cap.read()
                    # height, width, layers = frame.shape
                    # size = (width, height)
                    # file_name = abs_path('tempDir/' + self.uploaded_video.name + '.avi', path_type="current")
                    # out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

                    while cap.isOpened() and not self.close_flag:
                        ret, frame = cap.read()
                        if ret:
                            framecopy = frame.copy()
                            # Compute the current frame time in seconds
                            current_time = current_frame / fps
                            if current_time < total_length:
                                current_frame += 1
                                current_time_str = format_time(current_time)
                                image, detInfo, _ = self.frame_process(frame, self.uploaded_video.name,video_time=current_time_str)
                                # Save detection result images
                                if detInfo:
                                    # Parse the time string into a datetime
                                    time_obj = datetime.strptime(current_time_str, "%H:%M:%S")

                                    # Format the datetime as a string
                                    formatted_time = time_obj.strftime("%H_%M_%S")
                                    file_name = abs_path(video_savepath + '/' + formatted_time  + '_' + str(current_frame) + '.jpg',
                                                         path_type="current")
                                    save_chinese_image(file_name, image)

                                # # Save detection result video
                                # out.write(image)

                                # Set the new size
                                new_width = 1080
                                new_height = int(new_width * (9 / 16))
                                # Resize the image
                                resized_image = cv2.resize(image, (new_width, new_height))
                                resized_frame = cv2.resize(framecopy, (new_width, new_height))
                                if self.display_mode == "Overlay":
                                    self.image_placeholder.image(resized_image, channels="BGR", caption="Video")
                                else:
                                    self.image_placeholder.image(resized_frame, channels="BGR", caption="Original")
                                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="Detection")

                                self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                                # Update the progress bar
                                if total_length > 0:
                                    progress_percentage = int(((current_frame + 1) / total_frames) * 100)
                                    try:
                                        self.progress_bar.progress(progress_percentage)
                                    except:
                                        pass

                                current_frame += 1
                        else:
                            break
                    if self.close_flag:
                        self.logTable.save_to_csv()
                        self.logTable.update_table(self.log_table_placeholder)
                        cap.release()
                        # out.release()

                    self.logTable.save_to_csv()
                    self.logTable.update_table(self.log_table_placeholder)
                    cap.release()
                    # out.release()

                finally:

                    if self.uploaded_video is None:
                        name_in = None
                    else:
                        name_in = self.uploaded_video.name

                    res = self.logTable.save_frames_file(fps=self.FPS, video_name=name_in)
                    st.write("Recognition results saved to: " + self.saved_log_data)
                    if res:
                        st.write(f"Target result file saved: {res}")

                    tfile.close()
                    # Delete the temp file after processing if it is no longer needed
                    print(tfile.name + ' temporary file can be deleted')
                    # os.remove(tfile.name)

            else:
                st.warning("Please select a camera or upload a file.")

    def toggle_comboBox(self, frame_id):
        """
        Process and display detection results for a given frame.

        Args:
            frame_id (int): Frame ID whose detection results should be shown.

        Show detection results and the image for the selected frame ID.
        """
        # Make sure detection results have been saved
        if len(self.logTable.saved_results) > 0:
            frame = self.logTable.saved_images_ini[-1]  # Get the most recent frame
            image = frame  # Use it as the current image

            # Iterate saved detection results
            for i, detInfo in enumerate(self.logTable.saved_results):
                if frame_id != -1:
                    # If a frame ID is set, only process that frame
                    if frame_id != i:
                        continue

                if len(detInfo) > 0:
                    name, bbox, conf, use_time, cls_id = detInfo  # Get detection info
                    label = '%s %.0f%%' % (name, conf * 100)  # Build the label text

                    disp_res = ResultLogger()  # Create a result logger
                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(use_time))  # Merge results
                    self.table_placeholder.table(res)  # Show results in the table

                    # If a source image was saved
                    if len(self.logTable.saved_images_ini) > 0:
                        if len(self.colors) < cls_id:
                            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                                           range(cls_id+1)]
                        image = drawRectBox(image, bbox, alpha=0.2, addText=label,
                                            color=self.colors[cls_id])  # Draw the box and label

            # Resize to the display size
            new_width = 1080
            new_height = int(new_width * (9 / 16))
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Show overlay or side-by-side views
            if self.display_mode == "Overlay":
                self.image_placeholder.image(resized_image, channels="BGR", caption="Detection")
            else:
                self.image_placeholder.image(resized_frame, channels="BGR", caption="Original")
                self.image_placeholder_res.image(resized_image, channels="BGR", caption="Detection")

    def frame_process(self, image, file_name,video_time = None):
        """
        Process and predict a single image frame.

        Args:
            image (numpy.ndarray): Input image.
            file_name (str): Processed file name.

        Returns:
            tuple: Processed image, detection info, and selection list.

        Preprocess the input image, run the model, and handle predictions.
        """
        # image = cv2.resize(image, (640, 640))  # Resize the image for the model
        pre_img = self.model.preprocess(image)  # Preprocess the image

        # Update model parameters
        params = {'conf': self.conf_threshold, 'iou': self.iou_threshold}
        self.model.set_param(params)

        t1 = time.time()
        pred = self.model.predict(pre_img)  # Run model prediction

        t2 = time.time()
        use_time = t2 - t1  # Measure per-image inference time

        aim_area = 0 #Compute target area

        det = pred[0]  # Get prediction results

        # Initialize detection info and selection lists
        detInfo = []
        select_info = ["All targets"]

        # If there are valid detections
        if det is not None and len(det):
            det_info = self.model.postprocess(pred)  # Postprocess predictions
            if len(det_info):
                disp_res = ResultLogger()
                res = None
                cnt = 0

                # Iterate detected objects
                for info in det_info:
                    name, bbox, conf, cls_id, mask = info['class_name'], info['bbox'], info['score'], info['class_id'], info['mask']

                    # Draw boxes, labels, and area
                    image,aim_frame_area = draw_detections(image, info, alpha=0.5)
                    # image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=self.colors[cls_id])

                    res = disp_res.concat_results(name, bbox, str(int(aim_frame_area)),
                                                  video_time if video_time is not None else str(round(use_time, 2)))

                    # Add a log entry
                    self.logTable.add_log_entry(file_name, name, bbox, int(aim_frame_area), video_time if video_time is not None else str(round(use_time, 2)))
                    # Record detection info
                    detInfo.append([name, bbox, int(aim_frame_area), video_time if video_time is not None else str(round(use_time, 2)), cls_id])
                    # Append to the selection list
                    select_info.append(name + "-" + str(cnt))
                    cnt += 1

                # Show detections in the table
                self.table_placeholder.table(res)

        return image, detInfo, select_info

    def frame_table_process(self, frame, caption):
        # Show the frame and update results
        self.image_placeholder.image(frame, channels="BGR", caption=caption)

        # Update detection results
        detection_result = "None"
        detection_location = "[0, 0, 0, 0]"
        detection_confidence = str(random.random())
        detection_time = "0.00s"

        # Display results with concat_results
        res = concat_results(detection_result, detection_location, detection_confidence, detection_time)
        self.table_placeholder.table(res)
        # Add a short delay
        cv2.waitKey(1)

    def setupMainWindow(self):
        """ Run the detection system. """
        # st.title(self.title) # Show the system title
        st.write("--------")
        st.write("Image Segmentation Studio")
        st.write("--------")
        # Insert a divider

        # Create a column layout with the table on the right
        col1, col2, col3 = st.columns([4, 1, 2])

        # Display-mode selector in the first column
        with col1:
            self.display_mode = st.radio("Single / dual view", ["Overlay", "Side by side"])
            # Create empty containers for video based on display mode
            if self.display_mode == "Overlay":
                self.image_placeholder = st.empty()
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="Original")
            else:
                # "Dual view"
                self.image_placeholder = st.empty()
                self.image_placeholder_res = st.empty()
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="Original")
                    self.image_placeholder_res.image(load_default_image(), caption="Detection")
            # Display progress bar
            self.progress_bar = st.progress(0)

        # Create an empty results table
        res = concat_results("None", "[0, 0, 0, 0]", "0.00", "0.00s")

        # Show the results table in the right column
        with col3:
            self.table_placeholder = st.empty()  # Place on the far right
            self.table_placeholder.table(res)

            # Create an export-results button
            st.write("---------------------")
            if st.button("Export results"):
                self.logTable.save_to_csv()
                if self.uploaded_video is None:
                    name_in = None
                else:
                    name_in = self.uploaded_video.name
                res = self.logTable.save_frames_file(fps=self.FPS, video_name=name_in)
                st.write("Recognition results saved to: " + self.saved_log_data)
                if res:
                    st.write(f"Target result file saved: {res}")
                self.logTable.clear_data()
            # Show the full results table
            self.log_table_placeholder = st.empty()
            self.logTable.update_table(self.log_table_placeholder)

        # Empty stop-button placeholder in this column
        with col2:
            st.write("")
            self.close_placeholder = st.empty()

        # Target filter in the second column
        # with col2:
        # self.selectbox_placeholder = st.empty()
        # detected_targets = ["All targets"] # Initialize the target list
        #
        # Iterate and show detections
        # for i, info in enumerate(self.logTable.saved_results):
        # name, bbox, conf, use_time, cls_id = info
        # detected_targets.append(name + "-" + str(i))
        # self.selectbox_target = self.selectbox_placeholder.selectbox("Target filter", detected_targets)
        #
        # Handle target-filter selection
        # for i, info in enumerate(self.logTable.saved_results):
        # name, bbox, conf, use_time, cls_id = info
        # if self.selectbox_target == name + "-" + str(i):
        # self.toggle_comboBox(i)
        # elif self.selectbox_target == "All targets":
        # self.toggle_comboBox(-1)

        with col2:
            st.write("")
            run_button = st.button("Start Detection")
            if run_button:
                self.process_camera_or_file()  # Run camera or file processing
            else:
                # Show the default image if none has been saved
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="Original")
                    if self.display_mode == "Side by side":
                        self.image_placeholder_res.image(load_default_image(), caption="Detection")


# Instantiate and run the app
if __name__ == "__main__":
    app = Detection_UI()
    app.setupMainWindow()
