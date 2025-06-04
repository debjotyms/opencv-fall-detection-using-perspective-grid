import cv2
import mediapipe as mp
import numpy as np
import os

# Constants
checkerboard_size = (6, 8)  # Checkerboard size
grid_spacing = 6  # 6 inches per square
height_above_floor = 12  # 1 foot = 12 inches
real_world_person_height = 68  # Average adult height in inches
velocity_threshold = 200  # Threshold for velocity-based fall detection
history_length = 10  # Number of frames to keep for history analysis

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def load_calibration_image(image_path):
    """Load and process calibration image"""
    if not os.path.exists(image_path):
        print(f"Error: Could not load calibration image. Make sure '{image_path}' exists.")
        return None, None, None
        
    calibration_image = cv2.imread(image_path)
    if calibration_image is None:
        print(f"Error: Failed to read image from '{image_path}'.")
        return None, None, None
        
    gray_calibration = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_calibration, checkerboard_size, None)
    
    if not ret:
        print("No corners found in calibration image.")
        return None, None, None
        
    return calibration_image, gray_calibration, corners

def calibrate_camera(gray_image, corners):
    """Perform camera calibration"""
    # Prepare object points for checkerboard
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * grid_spacing
    
    object_points = [objp]
    image_points = [corners]
    
    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray_image.shape[::-1], None, None
    )
    
    if not ret:
        print("Camera calibration failed.")
        return None, None, None, None
        
    return mtx, dist, rvecs, tvecs

def draw_grid(frame, grid_points, rvecs, tvecs, mtx, dist, color):
    """Draw a grid on the frame"""
    image_points, _ = cv2.projectPoints(grid_points, rvecs[0], tvecs[0], mtx, dist)
    image_points_2d = image_points.reshape(20, 20, 2)
    
    for i in range(20):
        for j in range(20):
            if j < 19:
                pt1 = (int(image_points_2d[i, j, 0]), int(image_points_2d[i, j, 1]))
                pt2 = (int(image_points_2d[i, j+1, 0]), int(image_points_2d[i, j+1, 1]))
                cv2.line(frame, pt1, pt2, color, 1)
            if i < 19:
                pt3 = (int(image_points_2d[i, j, 0]), int(image_points_2d[i, j, 1]))
                pt4 = (int(image_points_2d[i+1, j, 0]), int(image_points_2d[i+1, j, 1]))
                cv2.line(frame, pt3, pt4, color, 1)
                
    return image_points_2d

def get_min_y_coordinate(image_points_2d):
    """Get minimum y coordinate from grid points"""
    min_y = float('inf')
    for i in range(image_points_2d.shape[0]):
        for j in range(image_points_2d.shape[1]):
            current_y = int(image_points_2d[i, j, 1])
            min_y = min(min_y, current_y)
    return min_y

def main():
    # Initialize variables
    pose_history = []
    fall_confidence = 0
    
    # Load calibration image
    calibration_image, gray_calibration, corners = load_calibration_image('./source_2.jpg')
    if calibration_image is None:
        return
    
    # Calibrate camera
    mtx, dist, rvecs, tvecs = calibrate_camera(gray_calibration, corners)
    if mtx is None:
        return
    
    # Load video
    cap = cv2.VideoCapture('./coffe_video(1).avi')
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Create grid points
    grid_points_floor = np.array([[x * grid_spacing, y * grid_spacing, 0] 
                                for x in range(-10, 10) 
                                for y in range(-10, 10)], dtype=np.float32)
                                
    grid_points_upper = np.array([[x * grid_spacing, y * grid_spacing, height_above_floor] 
                                for x in range(-10, 10) 
                                for y in range(-10, 10)], dtype=np.float32)
    
    fall_detected_flag = False
    is_paused = False
    playback_speed = 1.0
    frame_delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))
    frame_buffer = []
    buffer_size = 50
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('fall_detection_output.avi', fourcc, output_fps, (output_width, output_height))
    
    # Setup pose detector
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    break
                
                # Add frame to buffer
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)
            
            # Process with MediaPipe Pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Draw floor grid (Green)
            image_points_2d_floor = draw_grid(frame, grid_points_floor, rvecs, tvecs, mtx, dist, (0, 255, 0))
            
            # Draw upper grid (Red)
            image_points_2d_upper = draw_grid(frame, grid_points_upper, rvecs, tvecs, mtx, dist, (0, 0, 255))
            min_y_upper_grid = get_min_y_coordinate(image_points_2d_upper)
            
            # Draw vertical lines between grids
            for i in range(0, 20, 5):
                for j in range(0, 20, 5):
                    pt_floor = (int(image_points_2d_floor[i, j, 0]), int(image_points_2d_floor[i, j, 1]))
                    pt_upper = (int(image_points_2d_upper[i, j, 0]), int(image_points_2d_upper[i, j, 1]))
                    cv2.line(frame, pt_floor, pt_upper, (255, 165, 0), 1)  # Orange lines
            
            # Draw skeleton and check for fall
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                
                # Get key landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Get nose and draw as colored point
                nose_landmark = landmarks[mp_pose.PoseLandmark.NOSE]
                nose_x = int(nose_landmark.x * frame.shape[1])
                nose_y = int(nose_landmark.y * frame.shape[0])
                cv2.circle(frame, (nose_x, nose_y), 10, (255, 255, 0), -1)
                
                # Get hip points for body orientation
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                mid_hip_x = int(((left_hip.x + right_hip.x) / 2) * frame.shape[1])
                mid_hip_y = int(((left_hip.y + right_hip.y) / 2) * frame.shape[0])
                
                # Calculate body orientation (vertical = 0°, horizontal = 90°)
                dx = mid_hip_x - nose_x
                dy = mid_hip_y - nose_y
                body_angle = abs(np.degrees(np.arctan2(dx, dy)))
                
                # Store current pose data for velocity calculation
                current_pose = {
                    'nose': (nose_x, nose_y),
                    'mid_hip': (mid_hip_x, mid_hip_y),
                    'body_angle': body_angle,
                    'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
                }
                
                pose_history.append(current_pose)
                if len(pose_history) > history_length:
                    pose_history.pop(0)
                
                # Draw bounding box and distance
                h, w, _ = frame.shape
                x_coords = [int(lm.x * w) for lm in landmarks if lm.visibility > 0.5]
                y_coords = [int(lm.y * h) for lm in landmarks if lm.visibility > 0.5]
                
                if x_coords and y_coords:  # Check if lists are not empty
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Draw bounding box with padding
                    padding = 20
                    cv2.rectangle(frame, 
                              (max(0, x_min - padding), max(0, y_min - padding)),
                              (min(w, x_max + padding), min(h, y_max + padding)),
                              (255, 0, 0), 2)
                    
                    # Estimate distance from camera
                    bbox_height_pixels = y_max - y_min
                    if bbox_height_pixels > 0:  # Avoid division by zero
                        focal_length = mtx[0][0]
                        distance_inches = (real_world_person_height * focal_length) / bbox_height_pixels
                        cv2.putText(frame, f"Distance: {distance_inches:.2f} inches", 
                                  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Enhanced fall detection logic
                fall_detected = False
                
                # 1. Check position threshold (existing logic)
                position_fall = nose_y > min_y_upper_grid
                
                # 2. Check body orientation (horizontal orientation suggests fall)
                orientation_fall = body_angle > 60  # Body angle > 60° suggests horizontal position
                
                # 3. Check velocity (sudden movements)
                velocity_fall = False
                if len(pose_history) >= 2:
                    prev_pose = pose_history[-2]
                    time_diff = current_pose['timestamp'] - prev_pose['timestamp']
                    if time_diff > 0:  # Avoid division by zero
                        # Calculate velocity of nose movement
                        nose_velocity = np.sqrt(
                            (current_pose['nose'][0] - prev_pose['nose'][0])**2 +
                            (current_pose['nose'][1] - prev_pose['nose'][1])**2
                        ) / time_diff
                        velocity_fall = nose_velocity > velocity_threshold
                
                # Show debug info
                cv2.putText(frame, f"Body angle: {body_angle:.1f}°", (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Combine all fall indicators with a confidence-based approach
                # Base confidence on multiple factors
                if position_fall:
                    fall_confidence += 2  # Position is a strong indicator
                if orientation_fall:
                    fall_confidence += 1.5  # Orientation is a good indicator
                if velocity_fall:
                    fall_confidence += 1  # Sudden movement adds to confidence
                
                # Apply decay to confidence (gradually reduce if no indicators)
                fall_confidence = max(0, fall_confidence - 0.2)
                
                # Set threshold for fall detection
                if fall_confidence > 3:
                    fall_detected = True
                else:
                    fall_detected = False
                    
                # Show fall confidence meter
                cv2.putText(frame, f"Fall confidence: {fall_confidence:.1f}", (10, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Handle fall detection state and display
                if fall_detected:
                    if not fall_detected_flag:
                        print("Fall Detected!")
                        fall_detected_flag = True
                    cv2.putText(frame, "FALL DETECTED", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # Draw a transparent red overlay to highlight fall
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                else:
                    fall_detected_flag = False
            
            # Add FPS and playback speed info
            cv2.putText(frame, f"Speed: {playback_speed:.1f}x", (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
            # Write frame to output video
            out.write(frame)
                
            # Display the frame with info
            cv2.imshow("Fall Detection System", frame)
            
            # Handle user input
            key = cv2.waitKey(int(frame_delay / playback_speed))
            if key == ord('q'):  # Quit
                break
            elif key == ord(' '):  # Toggle play/pause
                is_paused = not is_paused
                status = "Paused" if is_paused else "Playing"
                print(f"Video {status}")
            elif key == ord('+') or key == ord('='):  # Increase speed
                playback_speed = min(4.0, playback_speed + 0.1)  # Cap at 4x speed
                print(f"Playback speed: {playback_speed:.1f}x")
            elif key == ord('-'):  # Decrease speed
                playback_speed = max(0.1, playback_speed - 0.1)  # Minimum 0.1x speed
                print(f"Playback speed: {playback_speed:.1f}x")
            elif key == ord('n'):  # Next frame when paused
                if is_paused:
                    ret, frame = cap.read()
                    if ret:
                        frame_buffer.append(frame.copy())
                        if len(frame_buffer) > buffer_size:
                            frame_buffer.pop(0)
                    else:
                        print("End of video reached.")
            elif key == ord('p'):  # Previous frame when paused
                if is_paused and len(frame_buffer) > 1:
                    frame_buffer.pop()  # Remove current frame
                    frame = frame_buffer[-1].copy()  # Get previous frame
            elif key == ord('s'):  # Save current frame
                cv2.imwrite("fall_detection_frame.jpg", frame)
                print("Frame saved as fall_detection_frame.jpg")
                
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

if __name__ == "__main__":
    main()