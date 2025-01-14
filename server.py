from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Flask API Setup
app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get("input")
    if not input_text:
        return jsonify({"error": "No input provided"}), 400

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug="True")