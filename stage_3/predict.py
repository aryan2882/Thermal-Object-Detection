import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and class labels
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

def categorize_and_predict(image):
    """
    Detects and classifies objects in an image using MobileNet SSD.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with bounding boxes and labels.
    """
    # Get image dimensions
    h, w = image.shape[:2]

    # Convert image to blob format (needed for deep learning models)
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

    # Set input to the pre-trained deep learning model
    net.setInput(blob)
    
    # Get detections
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence score

        if confidence > 0.5:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])  # Get class ID
            label = CLASSES[class_id]  # Get object label

            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return image
