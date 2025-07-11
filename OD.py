import cv2
from ultralytics import YOLO

# Load better model for accuracy (yolov8s.pt or higher)
model = YOLO('yolov8s.pt')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define colors for different classes (BGR format)
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Keep original orientation (no flip)
    # Run prediction directly on BGR frame
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # Get detection results
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
        
        # Draw colorful boxes and labels
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class name
            class_name = model.model.names[class_id] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(class_id)
            
            # Choose color based on class_id
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    
    # Show result
    cv2.imshow('YOLOv8 Real-Time Detection', frame)
    
    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
