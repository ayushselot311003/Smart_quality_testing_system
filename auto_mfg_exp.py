import cv2
import tkinter as tk
from PIL import Image, ImageTk
import ocr_detection

# Initialize OCR reader
reader = ocr_detection.Reader(['en'])  # 'en' for English

# Initialize dictionaries to store brand counts and dynamically generated labels
brand_counts = {}
brand_labels = {}  # To store label references

# Function to start webcam and process video feed
def start_webcam():
    global panel, video_stream
    # Start video capture from webcam
    video_stream = cv2.VideoCapture(0)  # 0 is for the default camera

    def capture_frame():
        global panel  # Declare panel as global to modify it

        # Read a frame from the webcam
        ret, frame = video_stream.read()
        if not ret:
            return

        # Perform OCR on the frame
        result = reader.readtext(frame)

        current_counts = {}

        # Initialize a variable to store the largest detected text (by area)
        largest_text = None
        largest_area = 0

        for (bbox, text, prob) in result:
            # Calculate the bounding box area (width * height)
            (top_left, top_right, bottom_right, bottom_left) = bbox
            width = abs(top_right[0] - top_left[0])
            height = abs(bottom_left[1] - top_left[1])
            area = width * height

            # Check if this is the largest detected text
            if area > largest_area:
                largest_area = area
                largest_text = text

        if largest_text:
            # If the detected brand starts with "MFG" or "Exp", just display the full text without incrementing the count
            if largest_text.startswith("MFG") or largest_text.startswith("EXP"):
                if largest_text not in brand_labels:
                    label = tk.Label(root, text=f"Detected Text: {largest_text}", font=("Arial", 16, "bold"))
                    label.pack()
                    brand_labels[largest_text] = label
                else:
                    brand_labels[largest_text].config(text=f"Detected Text: {largest_text}")
            else:
                # If the detected brand is new, create a label for it
                if largest_text not in brand_counts:
                    brand_counts[largest_text] = 0
                    # Dynamically create a label for the new brand
                    label = tk.Label(root, text=f"{largest_text}: 0", font=("Arial", 16, "bold"))
                    label.pack()
                    brand_labels[largest_text] = label

                # Increment count for the detected brand
                if largest_text not in current_counts:
                    current_counts[largest_text] = 0
                current_counts[largest_text] += 1

        # Update brand counts and corresponding labels for brands that don't start with "MFG" or "Exp"
        for brand, count in current_counts.items():
            brand_counts[brand] += count
            update_labels(brand)

        # Convert frame to a format Tkinter can use and display it
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((250, 250))  # Resize image for display
        imgtk = ImageTk.PhotoImage(image=img)

        if panel is None:
            panel = tk.Label(image=imgtk)
            panel.image = imgtk
            panel.pack(side="left", padx=10, pady=10)
        else:
            panel.configure(image=imgtk)
            panel.image = imgtk

        # Continue capturing frames
        root.after(10, capture_frame)  # Capture next frame after 10ms

    # Start capturing frames
    capture_frame()

# Function to update labels with brand counts
def update_labels(brand):
    brand_labels[brand].config(text=f"{brand}: {brand_counts[brand]}")

# Initialize Tkinter window
root = tk.Tk()
root.title("Product Brand Detection")

# Button to start the webcam feed
btn = tk.Button(root, text="Start Webcam", command=start_webcam, font=("Arial", 14, "bold"))
btn.pack(side="top", padx=10, pady=10)

# Panel for displaying the webcam feed
panel = None
video_stream = None

# Start the GUI loop
root.mainloop()

# Release video stream on exit
if video_stream is not None:
    video_stream.release()
