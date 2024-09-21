import cv2
import dlib
import numpy as np
import json
import os

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Load the database from the JSON file
with open('database/face_database.json', 'r') as f:
    face_database = json.load(f)

# Create a new dictionary where keys are the IDs and values are the loaded embeddings
database = {unique_id: np.load(embedding_path) for unique_id, embedding_path in face_database.items()}

def get_face_embedding(image, face_rect):
    # Detect facial landmarks
    landmarks = predictor(image, face_rect)
    # Compute face embedding
    embedding = np.array(face_rec.compute_face_descriptor(image, landmarks))
    return embedding

def recognize_face(embedding, database, threshold=0.5):
    best_match = "Unknown"
    best_similarity = 0

    # Compare the embedding with all known embeddings in the database
    for person, db_embedding in database.items():
        similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
        if similarity > threshold and similarity > best_similarity:
            best_match = person
            best_similarity = similarity

    return best_match, best_similarity

def process_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get the video frame dimensions and FPS to create an output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Convert the frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get face embedding for each detected face
            embedding = get_face_embedding(frame, face)
            
            # Identify the person
            if embedding is not None:
                person, confidence = recognize_face(embedding, database)
                label = f"{person} ({confidence:.2f})"
                
                # Display the label on the frame
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    print(f"Processing complete. Output video saved to {output_video_path}")
