import tensorflow as tf
import cv2
import numpy as np
from extract import extract_details

# Load EfficientDet model from TensorFlow Hub
EFFICIENTDET_MODEL = "https://tfhub.dev/tensorflow/efficientdet/d1/1"
model = tf.saved_model.load(EFFICIENTDET_MODEL)

def detect_objects(image_path):
    """
    Detects objects using EfficientDet and extracts details.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Image with detected objects and bounding boxes.
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run detection
    detections = model(input_tensor)

    # Extract bounding boxes and labels
    num_detections = int(detections.pop("num_detections"))
    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)

    # Draw bounding boxes for high-confidence detections
    for i in range(num_detections):
        if scores[i] > 0.5:  # Confidence threshold
            y1, x1, y2, x2 = boxes[i]
            h, w, _ = image.shape
            (x1, y1, x2, y2) = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {classes[i]}: {scores[i]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Example Usage
if __name__ == "__main__":
    image_path = "test.jpg"
    
    # Run EfficientDet for object detection
    detected_image = detect_objects(image_path)

    # Extract text from the image
    extracted_text = extract_details(image_path)

    # Show results
    print("Extracted Text:", extracted_text)
    cv2.imshow("Detected Objects", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
