import cv2
import os
import sqlite3
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Step 1: Data Collection
# Create a folder to save collected images
data_dir = "collected_data"
os.makedirs(data_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture image, 'q' to quit.")

image_count = 1  # To name images sequentially
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the captured frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # Save image with sequential name
        img_name = os.path.join(data_dir, f"object_{image_count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Image saved: {img_name}")
        image_count += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Step 2: Object Detection using YOLO
# Load YOLO model
weights_path = "yolov3.weights"  # Path to YOLO weights
config_path = "yolov3.cfg"       # Path to YOLO config file
names_path = "coco.names"        # Path to COCO names file

# Check for YOLO model files
if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(names_path):
    raise FileNotFoundError("YOLO model files (weights, cfg, or names) not found in the directory.")

net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, 'r') as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects_yolo(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = len(indices)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_objects, image

# Step 3: Store Data in SQLite Database
db_file = "object_detection.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create table to store object detection data
cursor.execute('''
CREATE TABLE IF NOT EXISTS detection_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    detected_objects INTEGER,
    timestamp TEXT
)
''')
conn.commit()

detected_data = []
print("Running Object Detection with YOLO...")
for image_file in os.listdir(data_dir):
    image_path = os.path.join(data_dir, image_file)
    num_objects, result_image = detect_objects_yolo(image_path)

    # Save processed image for visualization
    processed_path = os.path.join(data_dir, f"processed_{image_file}")
    cv2.imwrite(processed_path, result_image)

    # Insert detection results into database
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO detection_results (image_name, detected_objects, timestamp) VALUES (?, ?, ?)",
                   (image_file, num_objects, timestamp))
    conn.commit()
    detected_data.append((image_file, num_objects))
    print(f"Processed {image_file}: Detected {num_objects} object(s)")

conn.close()

# Step 4: Visualize Stored Data
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

cursor.execute("SELECT image_name, detected_objects FROM detection_results")
rows = cursor.fetchall()
image_names = [row[0] for row in rows]
detected_objects = [row[1] for row in rows]

# plt.figure(figsize=(10, 6))
# plt.bar(image_names, detected_objects, color='skyblue')
# plt.xticks(rotation=45, ha="right")
# plt.xlabel("Image Name")
# plt.ylabel("Number of Objects Detected")
# plt.title("Object Detection Results with YOLO")
# plt.tight_layout()
# plt.show()
# conn.close()
