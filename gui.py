import tkinter as tk
from tkinter import filedialog
import threading
import os
from non_realtime_face_recognition import process_video

def select_and_process_video():
    # Open file dialog to select video file
    input_video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
    )
    if input_video_path:
        # Update status label to show processing has started
        status_var.set("Processing...")
        # Generate output video path by adding '_output' before the file extension
        base, ext = os.path.splitext(input_video_path)
        output_video_path = f"{base}_output{ext}"

        # Start a new thread to process the video
        threading.Thread(
            target=process_video_thread,
            args=(input_video_path, output_video_path),
            daemon=True
        ).start()
    else:
        status_var.set("No file selected.")

def process_video_thread(input_video_path, output_video_path):
    # Call the process_video function from face_recognition_module
    process_video(input_video_path, output_video_path)
    # Update the status label to indicate processing is done
    status_var.set("Done processing.")

# Main application code
if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("Video Face Recognition")

    # Create a StringVar to hold status messages
    status_var = tk.StringVar()
    status_var.set("Select a video to process.")

    # Create GUI elements
    select_button = tk.Button(root, text="Select Video", command=select_and_process_video)
    status_label = tk.Label(root, textvariable=status_var)

    # Layout the GUI elements
    select_button.pack(pady=10)
    status_label.pack(pady=10)

    # Start the Tkinter event loop
    root.mainloop()
