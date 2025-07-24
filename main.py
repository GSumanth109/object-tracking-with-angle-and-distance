import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from PIL import Image
from io import BytesIO
import os
from depth_anything_v2.dpt import DepthAnythingV2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Depth Anything V2 Configuration
model_configs = {
    'vitl': {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    }
}

# Initialize Depth Anything V2 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnythingV2(**model_configs['vitl'])
checkpoint = torch.load("checkpoints/depth_anything_v2_vitl.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device).eval()

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize([518, 518]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Camera setup
ipcam_url = "http://192.168.1.4:8080/shot.jpg"
os.makedirs("depth_output", exist_ok=True)
frame_id = 0

# Calibration variables
calibration_points = []
current_scaling_factor = 1.0  # Initial scaling factor

def smoothed_depth_to_cm(raw_depth):
    """
    Maps raw depth value to actual distance in cm using a smoothed exponential scale.
    Based on calibration: 240 -> 86cm, 360 -> 15cm
    """
    a = 6000  # scale factor
    b = 0.017  # exponential rate
    offset = 5  # smooth offset to avoid jitter
    
    # Exponential mapping
    distance = a * np.exp(-b * raw_depth) + offset
    return min(max(distance, 15), 86)  # clamp to known range

def calculate_angle(face_center, frame_width, frame_height, is_horizontal=True):
    """Calculate the angle of the face from the center of the frame"""
    # Center of the frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # Calculate horizontal or vertical angle
    if is_horizontal:
        delta_x = face_center[0] - frame_center_x
        angle = np.degrees(np.arctan(delta_x / (frame_width / 2)))
    else:
        delta_y = face_center[1] - frame_center_y
        angle = np.degrees(np.arctan(delta_y / (frame_height / 2)))
    
    return angle

def calculate_scaling_factor(known_distance, raw_depth):
    """Calculate and update scaling factor based on calibration data"""
    global current_scaling_factor
    calibration_points.append((raw_depth, known_distance))
    
    if len(calibration_points) >= 2:
        # Use linear regression for better accuracy
        raw_depths = np.array([p[0] for p in calibration_points])
        distances = np.array([p[1] for p in calibration_points])
        coeffs = np.polyfit(raw_depths, distances, 1)
        current_scaling_factor = coeffs[0]
        print(f"Updated scaling factor: {current_scaling_factor:.4f}")

def process_frame(frame):
    """Main processing pipeline for face detection and depth estimation"""
    # Convert to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection
    results = face_detection.process(rgb_frame)
    face_detected = False
    face_center = None
    raw_depth = None
    
    if results.detections:
        # Get largest face
        detection = max(results.detections,
                       key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
        bbox = detection.location_data.relative_bounding_box
        
        # Calculate face coordinates
        h, w = frame.shape[:2]
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        x_max = int((bbox.xmin + bbox.width) * w)
        y_max = int((bbox.ymin + bbox.height) * h)
        
        # Get face center
        face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        
        # Prepare image for depth estimation
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Depth estimation
        with torch.no_grad():
            depth_map = model(input_tensor).cpu().squeeze().numpy()
        
        # Convert depth map coordinates
        depth_h, depth_w = depth_map.shape
        scale_x = depth_w / w
        scale_y = depth_h / h
        
        # Get depth at face center
        depth_x = min(int(face_center[0] * scale_x), depth_w - 1)
        depth_y = min(int(face_center[1] * scale_y), depth_h - 1)
        raw_depth = depth_map[depth_y, depth_x]
        
        face_detected = True
        
        # Draw face bounding box and center
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.circle(frame, face_center, 5, (0, 0, 255), -1)
        
        # Calculate and display estimated distance
        estimated_distance = smoothed_depth_to_cm(raw_depth)
        cv2.putText(frame, f"Distance: {estimated_distance:.2f}cm", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate and display horizontal angle
        horizontal_angle = calculate_angle(face_center, frame.shape[1], frame.shape[0], is_horizontal=True)
        cv2.putText(frame, f"Horizontal Angle: {horizontal_angle:.2f}°", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate and display vertical angle
        vertical_angle = calculate_angle(face_center, frame.shape[1], frame.shape[0], is_horizontal=False)
        cv2.putText(frame, f"Vertical Angle: {vertical_angle:.2f}°", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, face_detected, raw_depth, face_center

# Main loop
while True:
    try:
        # Get frame from IP camera
        response = requests.get(ipcam_url, timeout=2)
        frame = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        
        # Process frame
        processed_frame, face_detected, raw_depth, _ = process_frame(frame)
        
        # Display results
        cv2.imshow("Face Distance and Angle Estimation", processed_frame)
        
        # Save frame occasionally
        if frame_id % 10 == 0:
            cv2.imwrite(f"depth_output/frame_{frame_id:04d}.png", processed_frame)
        
        frame_id += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and face_detected:
            try:
                actual_distance = float(input("Enter actual distance in cm: "))
                calculate_scaling_factor(actual_distance, raw_depth)
            except:
                print("Invalid input. Enter a numerical value (e.g., 1.5)")
        
    except Exception as e:
        print(f"Error: {e}")
        continue

cv2.destroyAllWindows()

# Save final calibration data
if calibration_points:
    with open("calibration_data.txt", "w") as f:
        f.write("Calibration Data:\n")
        f.write(f"Final Scaling Factor: {current_scaling_factor:.6f}\n\n")
        f.write("Raw Depth, Actual Distance\n")
        for raw, actual in calibration_points:
            f.write(f"{raw:.6f}, {actual:.2f}\n")
