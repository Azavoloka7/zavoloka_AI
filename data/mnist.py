import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist():
    # Load the MNIST dataset from TensorFlow/Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def preprocess_mnist(x_train, y_train, x_test, y_test, val_split=0.1, random_state=None):
    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=random_state)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def save_npz(x_train, y_train, x_val, y_val, x_test, y_test, file_path):
    # Save the preprocessed data as .npz file
    np.savez(file_path, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    print("Data saved successfully as", file_path)

def load_npz(file_path):
    # Load the preprocessed data from .npz file
    data = np.load(file_path)
    x_train, y_train, x_val, y_val, x_test, y_test = data['x_train'], data['y_train'], data['x_val'], data['y_val'], data['x_test'], data['y_test']
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == "__main__":
    # Example usage
    (x_train, y_train), (x_test, y_test) = load_mnist()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_mnist(x_train, y_train, x_test, y_test, val_split=0.1, random_state=42)
    save_npz(x_train, y_train, x_val, y_val, x_test, y_test, file_path="mnist_data.npz")
