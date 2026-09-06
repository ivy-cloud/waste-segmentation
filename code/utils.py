import os

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from QtFusion.path import abs_path


def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file on the server.

    Args:
        uploaded_file (UploadedFile): File uploaded through Streamlit.

    Returns:
        str: Full path of the saved file, or None if nothing was uploaded.

    When a user uploads a file, save it under a fixed server directory.
    """
    # Check whether a file was uploaded
    if uploaded_file is not None:
        base_path = "tempDir"  # Base directory for saved uploads

        # Create the directory if it does not exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # Build the full file path
        file_path = os.path.join(base_path, uploaded_file.name)

        # Write the file in binary mode
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write file bytes

        return file_path  # Return the saved path

    return None  # No file uploaded


def concat_results(result, location, confidence, time):
    """
    Build a DataFrame of detection results.

    Args:
        result (str): Detection result / class name.
        location (str): Detection location / bbox.
        confidence (str): Confidence score.
        time (str): Detection time or elapsed time.
    """
    # Create a DataFrame with these fields
    result_data = {
        "Result": [result],
        "Location": [location],
        "Confidence": [confidence],
        "Time": [time],
    }

    results_df = pd.DataFrame(result_data)
    return results_df


def load_default_image():
    """
    Load the default placeholder image.

    Returns:
        Image: Default image object.
    """
    ini_image = abs_path("icon/ini-image.png")
    return Image.open(ini_image)


def get_camera_names():
    """
    Get a list of available camera names.

    Returns:
        list: Includes "Camera off" plus available camera index strings.
    """
    camera_names = ["Camera off", "0"]
    max_test_cameras = 10  # Max camera indexes to probe

    for i in range(max_test_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened() and str(i) not in camera_names:
            camera_names.append(str(i))
            cap.release()
    if len(camera_names) == 1:
        st.write("No available camera found")
    return camera_names
