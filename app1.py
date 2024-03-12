from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uuid
import os
from PIL import Image

app = Flask(__name__)

# Function to preprocess the input image
def preprocess_image(img_path):
     # Load and resize the image
    img = Image.open(img_path)
    img = img.resize((128,128))
    
    # Convert image to NumPy array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the CNN model for each request
    cnn_model_filename = 'your_cnn_model.h5'
    cnn_model = load_model(cnn_model_filename)

    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        if file:
            # Generate a unique filename using uuid
            unique_filename = str(uuid.uuid4()) + '.jpg'
            file_path = 'uploads/' + unique_filename  # Save the file in the 'uploads' folder

            # Create the 'uploads' directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            # Save the file temporarily
            file.save(file_path)

            # Preprocess the image
            input_image = preprocess_image(file_path)

            # Make predictions using the CNN model
            prediction = cnn_model.predict(input_image)
            print(prediction)

            # Decide which result page to redirect to based on the prediction
            if prediction[0, 1] > 0.5:  # Assuming index 1 corresponds to pneumonia
                return redirect(url_for('result'))
            else:
                return redirect(url_for('result1'))

    return render_template('index.html')

# ... (rest of the code)


@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/result1')
def result1():
    return render_template('result1.html')

if __name__ == '__main__':
    app.run(debug=True)
