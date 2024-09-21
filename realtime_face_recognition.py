import cv2
import json
import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)

# Load the database from the JSON file
with open("database/face_database.json", "r") as f:
    face_database = json.load(f)

# Create a new dictionary where keys are the IDs and values are the loaded embeddings
database = {
    unique_id: np.load(embedding_path)
    for unique_id, embedding_path in face_database.items()
}


def get_face_embedding(image, face_rect):
    # Detect facial landmarks
    landmarks = predictor(image, face_rect)
    # Compute face embedding
    embedding = np.array(face_rec.compute_face_descriptor(image, landmarks))
    return embedding


def recognize_face(embedding, database, threshold=0.5):
    best_match = "Unknown"
    best_similarity = 0  # Initialize best similarity as zero

    for person, db_embedding in database.items():
        similarity = cosine_similarity([embedding], [db_embedding])[0][0]
        print(
            f"Comparing with {person}: {similarity}"
        )  # Debugging line to print similarities

        # Only consider matches above the threshold and prioritize the best match
        if similarity > threshold and similarity > best_similarity:
            best_match = person
            best_similarity = similarity

    print(f"Identified {best_match} with similarity {best_similarity}")

    return best_match, best_similarity


# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get face embedding
        embedding = get_face_embedding(frame, face)

        # Identify the person
        person, confidence = recognize_face(embedding, database)
        label = f"{person} ({confidence:.2f})"

        # Display the label on the frame
        cv2.putText(
            frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
