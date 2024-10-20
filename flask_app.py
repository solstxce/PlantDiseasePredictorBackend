from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('Pesticide_Sprayer.h5')

# Load disease information
disease_info = pd.read_csv('Pesto_Data.csv')

# Preprocess image
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Predict function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        prediction = predict(image)
        predicted_class = np.argmax(prediction[0])
        
        class_labels = disease_info['Disease'].tolist()
        predicted_label = class_labels[predicted_class]
        
        disease_details = disease_info[disease_info['Disease'] == predicted_label].iloc[0]
        
        return jsonify({
            'predicted_disease': predicted_label,
            'confidence': float(prediction[0][predicted_class]),
            'best_pesticides': disease_details['Best_pesticides'],
            'worst_pesticides': disease_details['Worst_pesticides']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
