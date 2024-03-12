# Import necessary libraries
import numpy as np
from model import build_model
from train import train_model
from dataset_utils import load_npz
import matplotlib.pyplot as plt

def main():
    # Load preprocessed data
    file_path = r'C:\Users\zavol\ZavolokaRepository\zavoloka_AI\data\mnist_data.npz'
    (x_train, y_train), (x_val, y_val), (_, _) = load_npz(file_path)

    # Define model
    input_shape = (28, 28)
    num_classes = 10
    model = build_model(input_shape, num_classes)

    # Train model
    history = train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=10)

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
