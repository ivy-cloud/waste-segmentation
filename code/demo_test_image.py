import random
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from hashlib import md5
from model import Web_Detector
from class_name_list import Label_list

def generate_color_based_on_name(name):
    # Generate a stable color from a name hash
    hash_object = md5(name.encode())
    hex_color = hash_object.hexdigest()[:6]  # First 6 hex digits
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR

def calculate_polygon_area(points):
    return cv2.contourArea(points.astype(np.float32))


def draw_label(image, text, position, font_size=20, color=(255, 0, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("simsun.ttc", font_size, encoding="unic")
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


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
        image = draw_label(image, name, (x1, y1 - int(30 * adjust_param)), font_size=int(35 * adjust_param))
        y_offset = int(50 * adjust_param)  # Leave space below the class label
    else:
        mask_points = np.concatenate(mask)
        aim_frame_area = calculate_polygon_area(mask_points)
        mask_color = generate_color_based_on_name(name)
        try:
            overlay = image.copy()
            cv2.fillPoly(overlay, [mask_points.astype(np.int32)], mask_color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.drawContours(image, [mask_points.astype(np.int32)], -1, (0, 0, 255), thickness=int(8 * adjust_param))

            # Compute area, perimeter, circularity
            area = cv2.contourArea(mask_points.astype(np.int32))
            perimeter = cv2.arcLength(mask_points.astype(np.int32), True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Sample average color inside the mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [mask_points.astype(np.int32)], -1, 255, -1)
            color_points = cv2.findNonZero(mask)
            selected_points = color_points[np.random.choice(color_points.shape[0], 5, replace=False)]
            colors = np.mean([image[y, x] for x, y in selected_points[:, 0]], axis=0)
            color_str = f"({colors[0]:.1f}, {colors[1]:.1f}, {colors[2]:.1f})"

            # Draw class name
            x, y = np.min(mask_points, axis=0).astype(int)
            image = draw_label(image, name, (x, y - int(30 * adjust_param)), font_size=int(35 * adjust_param))
            y_offset = int(50 * adjust_param)  # Leave space below the class label

            # Draw area, perimeter, circularity, and color
            metrics = [("Area", area), ("Perimeter", perimeter), ("Circularity", circularity), ("Color", color_str)]
            for idx, (metric_name, metric_value) in enumerate(metrics):
                text = f"{metric_name}: {metric_value}"
                image = draw_label(image, text, (x, y - y_offset - spacing * (idx + 1)),
                                          font_size=int(35 * adjust_param))

        except Exception as e:
            print(f"An error occurred: {e}")

    return image, aim_frame_area


def process_frame(model, image):
    pre_img = model.preprocess(image)
    pred = model.predict(pre_img)
    det = pred[0]

    if det is not None and len(det):
        det_info = model.postprocess(pred)
        for info in det_info:
            image, _ = draw_detections(image, info)
    return image




if __name__ == "__main__":
    cls_name = Label_list
    model = Web_Detector()
    model.load_model("./runs/segment/train5/weights/best.pt")

    # Image processing
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/test/images/277_jpg.rf.31bf9a138ad3e89fa4d3ece2e83c6fbb.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/valid/images/17_jpg.rf.508463a5a465c406550e026b1a70925a.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/105-1-_jpg.rf.6ae158179a353918cb2ff549c48c2ceb.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/163_jpg.rf.4398caa525818d6ea2572b9a2b4a7c32.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/154_jpg.rf.b24fee4801924d67451689be2bb989f4.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/182_jpg.rf.0fad2e32cff428348d8d37d575a3f236.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/gettyimages-1161069493-612x612_jpg.rf.d9ee540104eb4f57b9d083b9ef8aee00.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/gettyimages-1231046643-612x612_jpg.rf.623f2fc82b1e43c7dbc88bf80b42d371.jpg'
    # 'E:/project/waste/WasteInstanceSegment_2400_Images/train/images/gettyimages-1250813719-612x612_jpg.rf.27e697a383c01eb94167ab05d32ac543.jpg'
    image_path = 'E:/project/waste/codeseg/code/datasets/new_wastes/test/images/a96d67693601b30b2bf84672a1248501_mp4-0012_jpg.rf.fcf163fdcba5394b48f7c9530c3aa7c3.jpg'
    image = cv2.imread(image_path)
    if image is not None:
        processed_image = process_frame(model, image)
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Image not found.')

