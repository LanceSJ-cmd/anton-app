from flask import Flask, request, jsonify, render_template
from keras._tf_keras.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the trained model
model = load_model('models_new/skin_disease_best.keras')

# Class labels and skincare recommendations
class_labels = ['dry', 'normal', 'oily']
skincare_recommendations = {
    'dry': [
        "Prefer ointments or creams over lotions for fewer irritants.",
        "Use fragrance-free products to reduce harshness.",
        "Combat dryness with a humidifier or frequent moisturizing."
    ],
    'oily': [
        "Use non-comedogenic products to prevent clogged pores.",
        "Cleanse twice daily and after sweating.",
        "Choose a gentle foaming cleanser to avoid excess oil production."
    ],
    'normal': [
        "Opt for hyaluronic acid for hydration without excess oil.",
        "Maintain your skin barrier for overall health.",
        "Keep your routine simple and avoid unnecessary chemicals."
    ]
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((50, 50))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Initialize Flask app
app = Flask(__name__)

# Route for the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read and preprocess the image
    image_file = request.files['image']
    image = Image.open(image_file)
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Get recommendations if available
    recommendations = skincare_recommendations.get(predicted_class_label, [])

    # Prepare and return the response
    response = {
        'predicted_class': predicted_class_label,
        'recommendations': recommendations
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)