import os
import time
import cv2
import pandas as pd
from QtFusion.path import abs_path
from PIL import Image
import numpy as np
from datetime import datetime

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

class ResultLogger:
    def __init__(self):
        """
        Initialize the ResultLogger class.
        """
        self.results_df = pd.DataFrame(columns=["Recognition result", "Location", "Area", "Time"])

    def concat_results(self, result, location, confidence, time):
        """
        Show detection results and append them to the results DataFrame.

        Args:
            result (str): Detection result.
            location (str): Detection location.
            confidence (str): Confidence。
            time (str): Time of the detected target.

        Returns:
            pd.DataFrame: Updated DataFrame.
        """
        # Build a dict of these fields
        result_data = {
            "Recognition result": [result],
            "Location": [location],
            "Area": [confidence],
            "Time": [time]
        }

        # Create a new DataFrame row and append it
        new_row = pd.DataFrame(result_data)
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

        return self.results_df


class LogTable:
    def __init__(self, csv_file_path=None):
        """
        Initialize the class instance.

        Args:
            csv_file_path (str): CSV path for persisted data.
        """
        self.csv_file_path = csv_file_path
        self.saved_images = []
        self.saved_target_images = []
        self.saved_images_ini = []
        self.saved_results = []

        columns = ['File path', 'Recognition result', 'Location', 'Area', 'Time']

        # Try loading CSV data; fall back to an empty DataFrame
        try:
            # Check whether the CSV file exists
            if not os.path.exists(csv_file_path):
                # If missing, create an empty DataFrame with headers and save it
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(csv_file_path, index=False, header=True)

            self.data = pd.DataFrame(columns=columns)
            # self.data = pd.read_csv(csv_file_path, encoding='utf-8')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            columns = ['File path', 'Recognition result', 'Location', 'Area', 'Time']
            self.data = pd.DataFrame(columns=columns)

    def add_frames(self, image, detInfo, img_ini):
        self.saved_images.append(image)
        self.saved_images_ini.append(img_ini)
        self.saved_results = detInfo
        if detInfo:
            self.saved_target_images.append(image)
        # print('____')
        # print(detInfo)
        # print('____')

    def clear_frames(self):
        self.saved_images = []
        self.saved_images_ini = []
        self.saved_results = []
        self.saved_target_images = []

    def save_frames_file(self,fps = 30,video_name = 'save', video_time = None):

        if self.saved_images:  # Check that the list is not empty
            # Perform the save
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            if len(self.saved_images) == 1:
                # Save as an image when there is only one frame
                file_name = abs_path('tempDir/pic_' + str(now_time) + '.png', path_type="current")
                cv2.imwrite(file_name, self.saved_images[0])
                return file_name
            else:
                # Save as a video when there is a sequence of frames
                height, width, layers = self.saved_images[0].shape
                size = (width, height)
                if video_name is None:
                    save_name = 'camera'
                else:
                    save_name = video_name

                file_name = abs_path('tempDir/' + str(save_name) + '.avi', path_type="current")

                out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                for img in self.saved_images:
                    out.write(img)
                out.release()

                # # Save video and camera target screenshots
                # if video_name is None:
                #     camera_savepath = './tempDir/camera'
                #     if not os.path.exists(camera_savepath):
                #         os.makedirs(camera_savepath)
                #
                #     try:
                #         for imgid in range(len(self.saved_target_images)):
                #             file_name = abs_path(camera_savepath + '/' + str(imgid + 1) + '.jpg', path_type="current")
                #             # cv2.imwrite(file_name, self.saved_target_images[imgid])
                #             save_chinese_image(file_name, self.saved_target_images[imgid])
                #     except:
                #         pass
                # else:
                #     video_savepath = './tempDir/' + video_name
                #     if not os.path.exists(video_savepath):
                #         os.makedirs(video_savepath)
                #
                #     try:
                #         for imgid in range(len(self.saved_target_images)):
                #             # Parse the time string into a datetime
                #             time_obj = datetime.strptime(video_time, "%H:%M:%S")
                #
                #             # Format the datetime as a string
                #             formatted_time = time_obj.strftime("%H_%M_%S")
                #             file_name = abs_path(video_savepath + '/' + formatted_time + '_' + str(imgid + 1) + '.jpg', path_type="current")
                #             # cv2.imwrite(file_name, self.saved_target_images[imgid])
                #             save_chinese_image(file_name, self.saved_target_images[imgid])
                #     except:
                #         pass

                return file_name
        return False

    def add_log_entry(self, file_path, recognition_result, position, confidence, time_spent):
        """
        Add a new record to the log.

        Args:
            file_path (str): File path
            recognition_result (str): Recognition result
            position (str): Location
            confidence (float): Confidence
            time_spent (float): Elapsed time (seconds or milliseconds)

        Returns:
            None
        """
        # Create a new data row
        position_str = str(position)
        file_path = str(file_path)
        new_entry = pd.DataFrame([[file_path, recognition_result, position_str, confidence, time_spent]],
                                 columns=['File path', 'Recognition result', 'Location', 'Area', 'Time'])

        # Append the new row to the DataFrame
        self.data = pd.concat([new_entry, self.data]).reset_index(drop=True)

        return self.data

    def clear_data(self):
        columns = ['File path', 'Recognition result', 'Location', 'Area', 'Time']
        self.data = pd.DataFrame(columns=columns)

    def save_to_csv(self):
        # Save the updated DataFrame to CSV
        self.data.to_csv(self.csv_file_path, index=False, encoding='utf-8', mode='a', header=False)

    def update_table(self, log_table_placeholder):
        """
        Update the table to show the latest 500 records.

        Args:
            log_table_placeholder: Streamlit table placeholder

        Returns:
            None
        """
        # Check whether the DataFrame has more than 500 rows
        if len(self.data) > 500:
            # If over 500, show only the latest 500 records
            display_data = self.data.head(500)
        else:
            # Otherwise show all data
            display_data = self.data

        log_table_placeholder.table(display_data)