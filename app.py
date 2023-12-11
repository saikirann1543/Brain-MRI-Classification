from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import re

def remove_special_characters(filename):
    # Remove special characters using regex
    cleaned_filename = re.sub(r'[^\w\s.-]', '', filename)
    return cleaned_filename


app = Flask(__name__)

# Define class names and image size
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
class_namess = ["MRI", "Random Image"]

img_size = 224
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='/opt/render/project/src/mobilenet_v2.tflite')
interpreter.allocate_tensors()

# Load the TFLite model
mri_interpreter = tf.lite.Interpreter(model_path=r'mobilenet_mri.tflite')
mri_interpreter.allocate_tensors()

# Function to predict and return the class label

def predict_class(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_details = interpreter.get_output_details()
    result = interpreter.get_tensor(output_details[0]['index'])

     # Get class probabilities
    probabilities = tf.nn.softmax(result).numpy()

    # Get the predicted class index
    predicted_class_index = np.argmax(result)
    predicted_class = class_names[predicted_class_index]

    # Adjust the probability if the predicted class is 'no_tumor'
    if predicted_class == 'no_tumor':
        tumor_probability = 1 - probabilities[0][class_names.index('no_tumor')]
    else:
        tumor_probability = probabilities[0][class_names.index('no_tumor')]

    predicted_class_index = np.argmax(result)
    predicted_class = class_names[predicted_class_index]
    accuracy = 100 * np.max(result)  # Compute accuracy as a percentage
    print("Accuracy: ", accuracy)

    return predicted_class, accuracy, tumor_probability


# Function to determine whether an image is an MRI scan or not
def is_mri_scan(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Correct color channel ordering, resize, and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Set input tensor
    input_details = mri_interpreter.get_input_details()
    mri_interpreter.set_tensor(input_details[0]['index'], image)

    # Perform inference
    mri_interpreter.invoke()

    # Get output tensor
    output_details = mri_interpreter.get_output_details()
    result = mri_interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(result)
    predicted_class = class_namess[predicted_class_index]

    # Check if the predicted class is "MRI"
    return predicted_class == "MRI"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save image to the static folder
        filename = secure_filename(file.filename)
        filename = remove_special_characters(filename)
        file_path = os.path.join('/opt/render/project/src/static', filename)
        file.save(file_path)

        # Read the image and make predictions
        image = cv2.imread(file_path)
        is_mri = is_mri_scan(file_path)
        if is_mri:
            predicted_class, accuracy, tumor_probability = predict_class(image)
            # make accuracy and tumor_probability to 2 decimal places
            accuracy = format(accuracy, ".2f")
            tumor_probability = format(tumor_probability, ".2f")
            accuracy = str(accuracy) + "%"
            tumor_probability = str(tumor_probability) + "%"
        else :
            return error()
        # Pass the result to the 'result' route
        return redirect(url_for('result', result=predicted_class, image=filename, accuracy=accuracy, tumor_probability=tumor_probability))

    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    # Get result data from the URL parameters
    result = request.args.get('result')
    image = request.args.get('image')
    accuracy = request.args.get('accuracy')
    tumor_probability = request.args.get('tumor_probability')

    return render_template('result.html', result=result, image=image, accuracy=accuracy, tumor_probability=tumor_probability)

@app.route('/error', methods=['GET'])
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
