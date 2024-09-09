#pip install pennylane pennylane-qiskit torch
import pennylane as qml
import time
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
from torch.optim import Adam

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

# QNN layer using PennyLane
def qnn_layer(weights):
    for i in range(len(weights)):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RZ(weights[i, 2], wires=i)
    qml.broadcast(qml.CNOT, wires=range(len(weights)), pattern="ring")
    return [qml.expval(qml.PauliZ(i)) for i in range(len(weights))]

# Device and quantum node setup
n_qubits = 40
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnn_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    return qnn_layer(weights)

# QNN model class
class QNN(torch.nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.weight_shapes = {"weights": (n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnn_circuit, self.weight_shapes)
        self.fc = torch.nn.Linear(n_qubits, 3)

    def forward(self, x):
        x = self.qlayer(x)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)

# Train QNN
def train_qnn(X_train, y_train, n_qubits=4, n_layers=2, epochs=50, batch_size=8):
    # Convert data to torch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)

    # Instantiate the QNN model
    qnn_model = QNN(n_qubits, n_layers)
    optimizer = Adam(qnn_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_torch.size()[0])
        for i in range(0, X_train_torch.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_torch[indices], y_train_torch[indices]

            optimizer.zero_grad()
            output = qnn_model(batch_x)
            loss = torch.nn.functional.nll_loss(output, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return qnn_model

# Evaluate QNN
def evaluate_qnn(model, X_test, y_test):
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        output = model(X_test_torch)
    y_pred = output.argmax(dim=1).numpy()
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to compare ANN and QNN
def compare_ann_qnn():
    # Load and prepare data
    X, y = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train and evaluate QNN
    print("Training QNN...")
    start_time = time.time()
    qnn_model = train_qnn(X_train, y_train)
    qnn_train_time = time.time() - start_time
    qnn_accuracy = evaluate_qnn(qnn_model, X_test, y_test)

    # Display QNN results
    print(f"QNN Training Time: {qnn_train_time:.2f} seconds")
    print(f"QNN Accuracy: {qnn_accuracy:.2f}")

# Run comparison
if __name__ == "__main__":
    compare_ann_qnn()
