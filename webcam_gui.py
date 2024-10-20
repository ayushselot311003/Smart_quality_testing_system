import cv2
import tkinter as tk
from PIL import Image, ImageTk
import ocr_detection

# Initialize OCR reader
reader = ocr_detection.Reader(['en'])  # 'en' for English

# Initialize brand counts
brand_counts = {"Saffola": 0, "Maggi": 0, "Dettol": 0, "Colgate": 0}
previous_counts = {"Saffola": 0, "Maggi": 0, "Dettol": 0, "Colgate": 0}  # For comparison

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

        current_counts = {"Saffola": 0, "Maggi": 0, "Dettol": 0, "Colgate": 0}

        for (bbox, text, prob) in result:
            # Check if detected text is one of the known brands
            if text in current_counts:
                current_counts[text] += 1

        # Compare with previous frame's counts to prevent continuous increments for the same image
        for brand in current_counts:
            if current_counts[brand] > previous_counts[brand]:
                brand_counts[brand] += current_counts[brand] - previous_counts[brand]
                update_labels()

        # Update previous counts
        previous_counts.update(current_counts)

        # Convert frame to a format Tkinter can use and display it
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((1024, 720))  # Resize image for display
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
def update_labels():
    label_saffola.config(text=f"Saffola: {brand_counts['Saffola']}")
    label_branda.config(text=f"Maggi: {brand_counts['Maggi']}")
    label_brandb.config(text=f"Dettol: {brand_counts['Dettol']}")
    label_brandc.config(text=f"Colgate: {brand_counts['Colgate']}")

# Initialize Tkinter window
root = tk.Tk()
root.title("Product Brand Detection")

# Button to start the webcam feed
btn = tk.Button(root, text="Start Webcam", command=start_webcam, font=("Arial", 14, "bold"))
btn.pack(side="top", padx=10, pady=10)

# Labels to display brand counts with larger, bold font
label_saffola = tk.Label(root, text="Saffola: 0", font=("Arial", 16, "bold"))
label_saffola.pack()

label_branda = tk.Label(root, text="Maggi: 0", font=("Arial", 16, "bold"))
label_branda.pack()

label_brandb = tk.Label(root, text="Dettol: 0", font=("Arial", 16, "bold"))
label_brandb.pack()

label_brandc = tk.Label(root, text="Colgate: 0", font=("Arial", 16, "bold"))
label_brandc.pack()

# Panel for displaying the webcam feed
panel = None
video_stream = None

# Start the GUI loop
root.mainloop()

# Release video stream on exit
if video_stream is not None:
    video_stream.release()
