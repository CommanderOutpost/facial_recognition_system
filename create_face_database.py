import os
import json

# Directories
embeddings_folder = "face_embeddings"  # Folder containing face embeddings
database_file = "database/face_database.json"  # JSON file to store the database

# Load existing database (if it exists)
if os.path.exists(database_file):
    with open(database_file, "r") as f:
        face_database = json.load(f)
    print("Loaded existing database.")
else:
    face_database = {}  # Initialize an empty dictionary for the database
    print("No existing database found. Starting fresh.")

def add_to_database():
    # Iterate over all .npy files in the embeddings folder
    for filename in os.listdir(embeddings_folder):
        if filename.endswith("_embedding.npy"):
            # Extract the unique ID from the filename (remove _embedding.npy)
            unique_id = filename.replace("_embedding.npy", "")
            
            # Check if the ID is already in the database
            if unique_id in face_database:
                print(f"ID {unique_id} is already in the database.")
                continue
            
            # Store the file path instead of the actual embedding
            embedding_path = os.path.join(embeddings_folder, filename)
            
            # Add the ID and the file path to the database
            face_database[unique_id] = embedding_path
            print(f"Added ID {unique_id} with embedding path to the database.")
    
    # Save the updated database to the JSON file
    with open(database_file, "w") as f:
        json.dump(face_database, f, indent=4)
    print("Database updated and saved.")

# Run the function to update the database
add_to_database()
