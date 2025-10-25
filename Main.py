import cv2 as cv
import numpy as np
import face_recognition
import pandas
import time

cap = cv.VideoCapture(0)

process_this_frame = True
last_processed_time = time.time()
min_processing_interval = .05  # .1 means that = Process at most 10 frames per second

def process_frame(frame, scale_factor=0.25):
    # Resize frame for faster face detection
    small_frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find face locations in the small frame
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    
    # Scale back the face locations to match original frame size
    scaled_face_locations = []
    for top, right, bottom, left in face_locations:
        scaled_face_locations.append(
            (
                int(top / scale_factor),
                int(right / scale_factor),
                int(bottom / scale_factor),
                int(left / scale_factor)
            )
        )
    
    # Draw boxes around faces
    for top, right, bottom, left in scaled_face_locations:
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Add counter in bottom left
    num_faces = len(scaled_face_locations)
    # Create dark background for text
    cv.rectangle(frame, (10, frame.shape[0]-60), (200, frame.shape[0]-20), (0, 0, 0), -1)
    # Add text showing number of people
    cv.putText(frame, f'FACES DETECTED: {num_faces}', 
              (20, frame.shape[0]-35),  # position
              cv.FONT_HERSHEY_SIMPLEX,  # font
              0.7,  # font scale
              (255, 255, 255),  # color (white)
              2)  # thickness
    
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # Only process every few frames to reduce lag
    if process_this_frame and (current_time - last_processed_time) >= min_processing_interval:
        processed_frame = process_frame(frame)
        last_processed_time = current_time
    else:
        processed_frame = frame
    
    # Toggle frame processing flag
    process_this_frame = not process_this_frame
    
    # Display the frame
    cv.imshow('Face Detection', processed_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()