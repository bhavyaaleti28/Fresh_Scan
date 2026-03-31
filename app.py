from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('freshscan_cnn.h5')

# Must match training order exactly
CLASS_NAMES = [
    'Apple_Ripe',    'Apple_Rotten',   'Apple_Unripe',
    'Banana_Ripe',   'Banana_Rotten',  'Banana_Unripe',
    'Orange_Ripe',   'Orange_Rotten',  'Orange_Unripe'
]

@app.route('/analyse', methods=['POST'])
def analyse():
    if not request.data:
        return jsonify({'error': 'No image received'}), 400

    img = Image.open(io.BytesIO(request.data)).convert('RGB')
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr)
    label = CLASS_NAMES[np.argmax(predictions)]

    parts    = label.split('_')
    fruit    = parts[0]
    ripeness = parts[1]

    return jsonify({'fruit': fruit, 'ripeness': ripeness})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
