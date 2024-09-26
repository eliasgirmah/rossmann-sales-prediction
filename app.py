from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your model
model = joblib.load('models/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.form.getlist('input_features')
    input_features = [float(i) for i in input_features]  # Convert input to float

    if len(input_features) != 21:
        return jsonify({'error': f'Expected 21 features, but got {len(input_features)}'}), 400

    prediction = model.predict([input_features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
