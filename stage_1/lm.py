import cv2
import numpy as np

class LightweightFilter:
    def __init__(self, model_cfg="yolov4-tiny.cfg", model_weights="yolov4-tiny.weights", confidence_threshold=0.5):
        """
        Initializes the YOLO model and sets confidence threshold.
        
        Parameters:
            model_cfg (str): Path to the YOLO model configuration file.
            model_weights (str): Path to the YOLO model weights file.
            confidence_threshold (float): Minimum confidence required to keep a detection.
        """
        self.conf_threshold = confidence_threshold
        
        # Load YOLO model
        self.net = cv2.dnn.readNet(model_weights, model_cfg)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load class labels
        self.classes = self.load_classes("coco.names")
    
    def load_classes(self, class_file):
        """
        Loads class names from a file.
        
        Parameters:
            class_file (str): Path to the class labels file.

        Returns:
            list: List of class names.
        """
        with open(class_file, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    def filter_low_confidence(self, image_path):
        """
        Detects objects in an image and filters out low-confidence detections.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            numpy.ndarray: Processed image with high-confidence detections.
        """
        # Load image
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Preprocessing: Convert image to blob
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Forward pass through YOLO
        layer_outputs = self.net.forward(self.output_layers)

        # Initialize lists for detections
        boxes = []
        confidences = []
        class_ids = []

        # Process each output layer
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter detections based on confidence threshold
                if confidence > self.conf_threshold:
                    # Scale bounding box to image dimensions
                    center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.4)

        # Draw high-confidence detections
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display class label
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

# Example Usage
if __name__ == "__main__":
    model = LightweightFilter()

    input_image = "test.jpg"
    output_image = model.filter_low_confidence(input_image)

    # Show final filtered image
    cv2.imshow("Filtered Detections", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
