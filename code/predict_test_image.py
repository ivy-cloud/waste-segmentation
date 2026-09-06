import os
from ultralytics import YOLO
import cv2
import numpy as np
from llm_tools import UseLLM

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY is not set.")
chatgpt = UseLLM(api_key=api_key)

image_path ='./datasets/new_wastes/test/images/317376fc47dafdee5d71de5ee8c9a083_mp4-0003_jpg.rf.04425e56a6e604dd69c9fa24451cb976.jpg' 
#"E:/project/waste/WasteInstanceSegment_2400_Images/valid/images/18-4-_jpg.rf.a23ec759bc9a480703566fbad61130c9.jpg"
# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("./runs/segment/train5/weights/best.pt")  # load a custom model

classnames = ['bottle', 'carton', 'dixie_cup', 'paper']


# Read image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO

# Run model prediction
results = model(image_path)

# Fixed color map for classes
common_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Dark red
    (0, 128, 0),    # Dark green
    (0, 0, 128),    # Dark blue
    (128, 128, 0)   # Olive
]

# Limit colors to the number of classes
num_classes = 4
colors = {i: common_colors[i % len(common_colors)] for i in range(num_classes)}


# Count detections per class
class_counts = {}


# Visualize segmentation results
for result in results:
    masks = result.masks.data.cpu().numpy()  # Masks shape: (num_objects, height, width)
    classes = result.boxes.cls.cpu().numpy()  # Class index per object
    scores = result.boxes.conf.cpu().numpy()  # Confidence per object
    
    for i, mask in enumerate(masks):
        # Class color
        class_idx = int(classes[i])
        mask_color = colors[class_idx]

        # Update class counts
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1



        # Convert mask to a colored overlay
        mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
        mask_colored = np.stack([mask * mask_color[0], mask * mask_color[1], mask * mask_color[2]], axis=-1)
        
        # Overlay mask onto image
        image = cv2.addWeighted(image, 1.0, mask_colored, 0.5, 0)

        # Draw class and score on the object
        # bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)  # Bounding box (xmin, ymin, xmax, ymax)
        label = f"{int(classes[i])} {scores[i]:.2f}"
        print(label)

        # # Use mask centroid as label position
        # mask_coords = np.argwhere(mask == 1)
        # if mask_coords.size > 0:
        #     centroid = np.mean(mask_coords, axis=0).astype(int)
        #     cv2.putText(image, label, (centroid[1], centroid[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)

        # cv2.putText(image, label, (bbox[0] + 10, bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)
        # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), mask_color, 2)


# Draw class counts in the top-left corner
line_height = 20  # Line height
start_x, start_y = 10, 10  # Starting position
recyclables_count = 0
unrecyclable_count = 0
report_input = []
for class_idx, count in class_counts.items():
    message = f'Is {classnames[class_idx]} recyclable waste? Return True if yes, False if no.'
    result = chatgpt.use(message=message)
    if result:
        recyclables_count +=count
    else:
        unrecyclable_count += count
    text = f"{classnames[class_idx]}: {count}"
    report_input.append(text)
    color = colors[class_idx]
    cv2.putText(image, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    start_y += line_height

cv2.putText(image, f"Recyclable: {recyclables_count}", (start_x, start_y+line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(image, f"Non-recyclable: {unrecyclable_count}", (start_x, start_y + 2*line_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# Convert to BGR for OpenCV display
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Show result
cv2.imshow("Segmented Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result if needed
cv2.imwrite("segmented_output.jpg", image)

report_message = ','.join(report_input) + ' Please generate a report based on the waste above, and include disposal methods.'
text = chatgpt.use(report_message)
print(text)