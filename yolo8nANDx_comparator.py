from ultralytics import YOLO
import os
import csv
from tqdm import tqdm

# Load a model
model_n = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
# Load a model
model_x = YOLO('yolov8x.pt')  # pretrained YOLOv8x model


classes_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]
names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
         9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
         16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
         25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
         33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
         49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
         58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
         67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
         76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

folder = "Frames/"
images = []
confidence = 0.5
classes_list = [0, 1, 2, 3, 5, 7]
csv_filename = "YoloNandX_comparison.csv"
with open(csv_filename, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image Path", "person_N", "person_X", "bicycle_N", "bicycle_X", "car_N", "car_X", "motorcycle_N", "motorcycle_X", 
                     "bus_N", "bus_X", "truck_N", "truck_X"])

for img in tqdm(os.listdir(folder), desc="Folder completion"):
    img_src = os.path.join(folder, img)

    filerow = [img_src]
    # Run Model YoloV8n on a list of images
    results_n = model_n(img_src, conf=confidence, classes=classes_list) # return a list of Results objects
    count_n = [0] * len(names)
    for result in results_n:
        boxes = result.boxes  # Boxes object for bounding box outputs
        box_cls = result.boxes.cls.numpy().tolist()
        for c in box_cls:
            count_n[int(c)] += 1


    # Run Model YoloV8x on a list of images
    results_x = model_x(img_src, conf=confidence, classes=classes_list) # return a list of Results objects
    count_x = [0] * len(names)
    for result in results_x:
        boxes = result.boxes  # Boxes object for bounding box outputs
        box_cls = result.boxes.cls.numpy().tolist()
        for c in box_cls:
            count_x[int(c)] += 1



    #### Append count of predicted number for a particular class by respective models to "filerow" variable to write in csv file.
    for i in classes_list:
        filerow.append(count_n[i])
        filerow.append(count_x[i])
    
    #### Open and write the "filerow" variable in csv file
    with open(csv_filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(filerow)


