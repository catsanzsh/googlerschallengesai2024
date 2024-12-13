import cv2
import numpy as np
import tensorflow as tf

def identify_bird(image_path, model_path):
    """
    Identifies the bird species in an image using a trained TensorFlow model.

    Args:
        image_path: The path to the image file.
        model_path: The path to the trained TensorFlow model.

    Returns:
        The predicted bird species as a string,
        or "Unknown" if no bird is detected.
    """
    try:
        # Read and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img = cv2.resize(img, (224, 224)) / 255.0

        # Load the TensorFlow model
        model = tf.keras.models.load_model(model_path)

        # Make a prediction
        # pred = np.argmax(model.predict(np.expand_dims(img, 0)))
        # Make a prediction (M1 compatible)
        prediction = model.predict(np.expand_dims(img, axis=0))
        pred = np.argmax(prediction, axis=1)[0]

        # Define the bird species list
        bird_species = ["American Crow", "Bald Eagle", "Blue Jay", "Canada Goose", "Cardinal"]

        # Map the prediction index to the species name
        bird_species_name = bird_species[pred]

    except Exception as e:
        print(f"An error occurred: {e}")
        bird_species_name = "Unknown"

    return bird_species_name
