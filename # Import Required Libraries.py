# Import Required Libraries
# Human Anatomy Chatbot with AI - Jupyter Notebook

# Import Required Libraries
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from flask import Flask, request, jsonify

# Dataset Loading and Preprocessing
def load_and_prepare_dataset(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jsonl'):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                all_data.extend([json.loads(line) for line in file])
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['contents'] = df['title'] + " " + df['content']
    return df

folder_path = 'C:\\Users\\alandavis\\Downloads\\chunk'  # Update with the actual folder containing JSONL files
dataset = load_and_prepare_dataset(folder_path)

# Display Sample Data
print("Dataset Sample:")
print(dataset.head())

# Split Dataset
def split_dataset(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    return train_data, test_data

train_data, test_data = split_dataset(dataset)

# Tokenizer and Model Initialization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['contents'].tolist(), padding="max_length", truncation=True, return_tensors="pt")

tokenized_train = Dataset.from_pandas(train_data).map(tokenize_function, batched=True)
tokenized_test = Dataset.from_pandas(test_data).map(tokenize_function, batched=True)

# Trainer Setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return {"accuracy": acc, "confusion_matrix": cm}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Training
trainer.train()

# Evaluation
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

# Plot Accuracy and Loss
training_logs = trainer.state.log_history
accuracy = [log['eval_accuracy'] for log in training_logs if 'eval_accuracy' in log]
loss = [log['eval_loss'] for log in training_logs if 'eval_loss' in log]

def plot_metrics(accuracy, loss):
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(accuracy, loss)

# Confusion Matrix
cm = metrics['confusion_matrix']
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.show()

# Flask API Setup
app = Flask(__name__)

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
    app.run(host="0.0.0.0", port=5000)

# Unity Integration Tab
print("\nInstructions for Unity Integration:")
print("1. Set up an HTTP client in Unity (e.g., using UnityWebRequest).")
print("2. Use POST requests to send user input to http://<host>:5000/predict.")
print("3. Parse the JSON response to get predictions.")

# Additional Table Creation
def create_additional_table(df):
    df['query_log'] = None  # Placeholder for logging user queries
    df['response_time'] = None  # Placeholder for tracking response times
    return df

dataset_with_additional_table = create_additional_table(dataset)
print("Updated Dataset Structure:")
print(dataset_with_additional_table.head())
