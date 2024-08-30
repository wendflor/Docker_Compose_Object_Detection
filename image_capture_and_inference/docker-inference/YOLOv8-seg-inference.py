import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import yaml
import json
import time

IMAGE_DIR = "/shared"
FLAG_FILE = "/shared/capture_complete.flag"

def draw_boxes_and_masks(image, predictions, class_colors, alpha=0.5):
    for pred in predictions:
        box = pred['box']
        mask = pred['mask']
        label = pred['label']
        confidence = pred['confidence']
        color = class_colors[label]

        # Draw the bounding box (denormalize)
        h, w = image.shape[:2]
        x_center, y_center, width, height = box
        x_center, y_center, width, height = x_center * w, y_center * h, width * w, height * h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Convert the mask to a NumPy array and resize (denormalize)
        mask = np.array(mask)
        mask[:, 0] = mask[:, 0] * w  # x coordinates
        mask[:, 1] = mask[:, 1] * h  # y coordinates
        mask = mask.astype(np.int32)

        # Create a blank image with the same dimensions as the original image
        overlay = image.copy()
        
        # Draw the mask on the overlay image
        cv2.fillPoly(overlay, [mask], color)

        # Blend the overlay with the original image
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Add the label and confidence
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def save_predictions(predictions, output_path):
    """ Save predictions as JSON for post-processing. """
    with open(output_path, 'w') as f:
        json.dump(predictions, f)

def main(img_path):
    model_path = 'best.pt'
    
    output_dir = IMAGE_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load the data configuration
    with open('data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
    
    class_names = data_config['names']
    num_classes = data_config['nc']
    class_colors = data_config['colors']

    # Load the model
    model = YOLO(model_path)

    # Ensure the model only uses the correct number of classes
    model.model.yaml['nc'] = num_classes
    model.model.names = class_names

    img_name = os.path.basename(img_path)
    image = cv2.imread(img_path)
    results = model(image, device=0, conf=0.7, retina_masks=True) # 

        # Extract predictions
    predictions = []
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xywhn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
        else:
            boxes = []
            classes = []
            confs = []

        if result.masks is not None:
            masks = result.masks.xyn
        else:
            masks = []

            # Ensure that all lists have the same length
        for box, mask, cls, conf in zip(boxes, masks, classes, confs):
            predictions.append({
                'box': box.tolist(),
                'mask': mask.tolist(),
                'label': class_names[int(cls)],
                'confidence': float(conf)
            })
        
        # Draw boxes and masks on the image
        output_image = draw_boxes_and_masks(image.copy(), predictions, class_colors)

        # Save the output image
        output_image_path = os.path.join(output_dir, f"output_{img_name}")
        cv2.imwrite(output_image_path, output_image)

        # Save the predictions for post-processing
        output_json_path = os.path.join(output_dir, f"predictions_{os.path.splitext(img_name)[0]}.json")
        save_predictions(predictions, output_json_path)

if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    while not os.path.exists(FLAG_FILE):
        time.sleep(1)  # Warten auf die Flag-Datei

    with open(FLAG_FILE, 'r') as f:
        latest_image_path = f.read().strip()
    
    if latest_image_path:
        main(latest_image_path)
        # Lösche die Flag-Datei nach der Inferenz
        os.remove(FLAG_FILE)


