import os
import cv2
import dlib
import numpy as np

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)

# Directories
faces_folder = "faces"  # Folder containing face images
embeddings_folder = "face_embeddings"  # Folder to save face embeddings

# Ensure the embeddings folder exists
os.makedirs(embeddings_folder, exist_ok=True)


def get_face_embedding(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the face in the image
    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces found in {image_path}")
        return None

    # Take the first detected face (assuming there's one face in the image)
    face = faces[0]

    # Detect landmarks
    landmarks = predictor(image, face)

    # Get the embedding
    embedding = np.array(face_rec.compute_face_descriptor(image, landmarks))

    return embedding


def generate_embeddings():
    # Iterate over all jpg/jpeg files in the faces folder
    for filename in os.listdir(faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Extract the unique identifier (6-letter string) by removing the extension
            unique_id = filename.split(".")[0]

            # Check if the embedding for this unique_id is already generated
            embedding_file = os.path.join(
                embeddings_folder, f"{unique_id}_embedding.npy"
            )
            if os.path.exists(embedding_file):
                print(f"Embedding for ID {unique_id} already exists.")
                continue

            # Get the full path of the image
            image_path = os.path.join(faces_folder, filename)

            # Generate the face embedding
            embedding = get_face_embedding(image_path)

            if embedding is not None:
                # Save the embedding as {unique_id}_embedding.npy
                np.save(embedding_file, embedding)
                print(f"Generated and saved embedding for ID {unique_id}")
            else:
                print(f"Failed to generate embedding for ID {unique_id}")


# Run the function to generate embeddings
generate_embeddings()
