import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Load the trained model
class_labels = ['Fresh Apple', 'Fresh Banana', 'Fresh Apple', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
model_path = r"model.h5"
model = load_model(model_path)

# Function to capture and predict the image from the webcam
def capture_and_predict():
    global label_result, frame

    if frame is not None:
        # Resize to match the input size of the model (64x64 here)
        img = cv2.resize(frame, (64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        print(f"Image array shape: {img_array.shape}")

        # Prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        # Update the label with the prediction
        label_result.config(text=f'Predicted: {predicted_class_label}')
    else:
        label_result.config(text="No frame to predict")

# Function to continuously show the webcam stream
def show_frame():
    global frame
    ret, frame = cap.read()

    if ret:
        # Convert the image to RGB (Tkinter can’t display BGR directly)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)

        # Update the panel with the new frame
        panel.imgtk = imgtk
        panel.config(image=imgtk)

    # Repeat this function after 10 ms to keep the stream alive
    panel.after(10, show_frame)

# Create the Tkinter window
root = tk.Tk()
root.title("Fruit Freshness Detector")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Create a label to display the result
label_result = Label(root, text="Prediction will appear here", font=("Arial", 16))
label_result.pack(pady=20)

# Panel to display webcam feed
panel = Label(root)
panel.pack()

# Button to capture the image and make a prediction
btn_predict = tk.Button(root, text="Predict", command=capture_and_predict)
btn_predict.pack(pady=20)

# Start showing the webcam feed
show_frame()

# Run the Tkinter event loop
root.mainloop()

# Release the webcam when the window is closed
cap.release()
cv2.destroyAllWindows()
