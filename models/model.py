import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_model(input_shape, num_classes):
    """
    Build a neural network model for MNIST classification.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (28, 28) for MNIST images).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: Compiled neural network model.
    """
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Example usage to print model summary
    input_shape = (28, 28)
    num_classes = 10
    model = build_model(input_shape, num_classes)
    model.summary()
