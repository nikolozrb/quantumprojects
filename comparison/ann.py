
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# TensorFlow for ANN
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load Iris dataset
def load_data():
    data = load_iris()
    X, y = data.data, data.target
    return X, y

# Split and standardize the dataset
def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Create ANN model
def create_ann_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train ANN
def train_ann(X_train, y_train):
    ann_model = create_ann_model(X_train.shape[1])
    start_time = time.time()
    ann_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    ann_train_time = time.time() - start_time
    return ann_model, ann_train_time

# Placeholder for QNN training and evaluation
def train_qnn_placeholder(X_train, y_train):
    time.sleep(2)  # Simulate training time
    qnn_accuracy = 0.9  # Simulated accuracy
    return qnn_accuracy

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to compare ANN and QNN
def compare_ann_qnn():
    # Load and prepare data
    X, y = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train and evaluate ANN
    ann_model, ann_train_time = train_ann(X_train, y_train)
    ann_accuracy = evaluate_model(ann_model, X_test, y_test)

    # Train and evaluate QNN (Placeholder)
    start_time = time.time()
    qnn_accuracy = train_qnn_placeholder(X_train, y_train)
    qnn_train_time = time.time() - start_time

    # Compare results
    print(f"ANN Training Time: {ann_train_time:.2f} seconds")
    print(f"ANN Accuracy: {ann_accuracy:.2f}")
    print(f"QNN Training Time (Placeholder): {qnn_train_time:.2f} seconds")
    print(f"QNN Accuracy (Placeholder): {qnn_accuracy:.2f}")

# Run comparison
if __name__ == "__main__":
    compare_ann_qnn()
