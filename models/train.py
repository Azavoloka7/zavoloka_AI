from zavoloka_AI.data import mnist
from zavoloka_AI.models import model

def train_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_mnist()

    # Build the model
    nn_model = model.build_model(input_shape=(28, 28), num_classes=10)

    # Train the model
    nn_model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # Evaluate the model
    loss, accuracy = nn_model.evaluate(x_test, y_test)
    print("Test accuracy:", accuracy)

    # Save the model
    nn_model.save("mnist_model.h5")

if __name__ == "__main__":
    train_model()
