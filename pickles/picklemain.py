import cv2
import numpy as np

class ConveyorBeltDetector:
    def __init__(self, lower_hsv, upper_hsv):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)  # Create mask based on HSV range
        return mask


from tensorflow.keras.models import load_model

class CucumberDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def is_cucumber(self, image):
        resized_image = cv2.resize(image, (64, 64))  # Resize image to model's input size
        resized_image = resized_image / 255.0  # Normalize image
        resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
        prediction = self.model.predict(resized_image)
        return prediction[0] > 0.5  # Assuming sigmoid activation for binary classification


class VideoProcessor:
    def __init__(self, conveyor_detector, cucumber_detector):
        self.conveyor_detector = conveyor_detector
        self.cucumber_detector = cucumber_detector

    def process_frame(self, frame):
        conveyor_mask = self.conveyor_detector.detect(frame)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(conveyor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            
            if self.cucumber_detector.is_cucumber(roi):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return frame


def main():
    # Define the color range for the conveyor belt in HSV
    lower_hsv = np.array([100, 150, 0])
    upper_hsv = np.array([140, 255, 255])
    
    # Initialize the detectors
    conveyor_detector = ConveyorBeltDetector(lower_hsv, upper_hsv)
    cucumber_detector = CucumberDetector('cucumber_model.h5')
    video_processor = VideoProcessor(conveyor_detector, cucumber_detector)
    
    # Open video capture
    cap = cv2.VideoCapture(0)  # Use your video source, 0 for webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = video_processor.process_frame(frame)
        
        cv2.imshow('Processed Frame', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
