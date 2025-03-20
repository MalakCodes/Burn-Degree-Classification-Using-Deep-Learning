# Project Idea: This project classifies burn degrees based on images uploaded by the user. 
# The user selects an image, and the model predicts the degree of burn (minor, moderate, or severe) 
# using a trained neural network. The prediction is displayed on the GUI along with the uploaded image.

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# 1. Define the path to the dataset containing folders for each class (burn degrees).
dataset_path = "C:/Users/malak/Desktop/train"  # Make sure to provide the correct path

# 2. Set up the image data generator to normalize images and split the data into training and validation sets
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize the image (scale pixel values to 0-1)
    validation_split=0.2  # Reserve 20% of the data for validation
)

# 3. Load the training data
train_generator = datagen.flow_from_directory(
    dataset_path,  # Path to the dataset
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,  # Number of images to process in each batch
    class_mode="categorical",  # Multi-class classification
    subset="training"  # Use this subset for training
)

# 4. Load the validation data
val_generator = datagen.flow_from_directory(
    dataset_path,  # Same dataset path
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode="categorical",  # Multi-class classification
    subset="validation"  # Use this subset for validation
)

# 5. Build the model using a Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes (burn degrees)
])

# 6. Compile the model with an optimizer and loss function suitable for categorical classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# 8. Create the GUI for user interaction
def upload_image():
    # Open file dialog to let the user upload an image
    filename = filedialog.askopenfilename()
    img = Image.open(filename)
    img = img.resize((224, 224))  # Resize the image to match the model input size
    img_array = np.array(img) / 255.0  # Normalize image

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension for prediction

    prediction = model.predict(img_array)  # Predict the class of the image
    predicted_class = np.argmax(prediction, axis=1)  # Get the index of the predicted class
    
    class_names = list(train_generator.class_indices.keys())  # Get class names from the training data
    predicted_class_name = class_names[predicted_class[0]]  # Get the predicted class name

    # Convert image to a format suitable for Tkinter
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

    # Display the predicted class (burn degree)
    label.config(text=f"Predicted: {predicted_class_name}")

# 9. Create the main window for the GUI
window = tk.Tk()
window.title("Burn Classification")
window.config(bg="#000000")  # Set the background color to black

# Add a title label at the top
title_label = tk.Label(window, text="Burn Degree Classification", font=("Arial", 20, "bold"), fg="white", bg="#000000")
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Add an upload button for the user to select an image
upload_button = tk.Button(window, text="Upload Image", font=("Arial", 14), bg="#4CAF50", fg="white", command=upload_image)
upload_button.grid(row=1, column=0, columnspan=2, pady=10)

# Add a panel to display the uploaded image
panel = tk.Label(window, bg="#000000")  # Set background color of image display area to black
panel.grid(row=2, column=0, columnspan=2)

# Add a label to display the prediction result (burn degree)
label = tk.Label(window, text="Predicted: ", font=("Arial", 14), fg="white", bg="#000000")  # White text on black background
label.grid(row=3, column=0, columnspan=2, pady=20)

# Start the GUI application
window.mainloop()