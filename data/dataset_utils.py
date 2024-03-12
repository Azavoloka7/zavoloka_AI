import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_mnist():
    """
    Load the MNIST dataset from TensorFlow/Keras.

    Returns:
        Tuple of numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_mnist(x_train, y_train, x_test, y_test, val_split=0.1, random_state=None):
    """
    Preprocess the MNIST dataset by normalizing pixel values and splitting into training/validation sets.

    Args:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
        val_split (float): Fraction of training data to use for validation.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        Tuple of numpy arrays: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    # Normalize pixel values to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=random_state)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def augment_data(x_train, y_train, batch_size=32, seed=None):
    """
    Apply data augmentation to the training data using TensorFlow/Keras ImageDataGenerator.

    Args:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        batch_size (int): Batch size for data augmentation.
        seed (int or None): Random seed for reproducibility.

    Returns:
        Tuple of numpy arrays: (x_train_aug, y_train_aug)
    """
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    datagen.fit(x_train)
    augment_data_generator = datagen.flow(x_train, y_train, batch_size=batch_size, seed=seed)
    return next(augment_data_generator)

def save_npz(x_train, y_train, x_val, y_val, x_test, y_test, file_path):
    """
    Save the preprocessed MNIST dataset as a .npz file.

    Args:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        x_val (numpy.ndarray): Validation images.
        y_val (numpy.ndarray): Validation labels.
        x_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
        file_path (str): Path to save the .npz file.
    """
    np.savez(file_path, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    print("Data saved successfully as", file_path)

def load_npz(file_path):
    data = np.load(file_path)
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
