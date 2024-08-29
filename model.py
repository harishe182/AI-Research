import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data from CSV file
data = pd.read_csv('/Users/harisheswarathas/AI Projects/Recongization/train.csv')

# Convert the data to a NumPy array
data = np.array(data)
m, n = data.shape  # Get number of samples (m) and features (n)
np.random.shuffle(data)  # Shuffle data before splitting

# Split the data into development (dev) and training sets
data_dev = data[0:1000].T  # Transpose the first 1000 samples for development set
Y_dev = data_dev[0]  # Labels for development set
X_dev = data_dev[1:n]  # Features for development set
X_dev = X_dev / 255.  # Normalize feature values to [0, 1]

data_train = data[1000:m].T  # Transpose the remaining samples for training set
Y_train = data_train[0]  # Labels for training set
X_train = data_train[1:n]  # Features for training set
X_train = X_train / 255.  # Normalize feature values to [0, 1]
_, m_train = X_train.shape  # Get the number of training samples

# Initialize parameters with more neurons in hidden layers
def init_params():
    W1 = np.random.rand(20, 784) - 0.5  # Increased number of neurons in layer 1
    b1 = np.random.rand(20, 1) - 0.5
    W2 = np.random.rand(10, 20) - 0.5  # Adjust layer 2 accordingly
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# ReLU activation function
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax activation function
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # Subtract max for numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Forward propagation through the network
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  # Linear transformation for layer 1
    A1 = ReLU(Z1)  # Apply ReLU activation
    Z2 = W2.dot(A1) + b2  # Linear transformation for layer 2
    A2 = softmax(Z2)  # Apply softmax activation
    return Z1, A1, Z2, A2

# Derivative of ReLU activation function
def ReLU_deriv(Z):
    return Z > 0  # Gradient of ReLU

# Convert labels to one-hot encoded format
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Create matrix of zeros
    one_hot_Y[np.arange(Y.size), Y] = 1  # Set the corresponding index to 1
    one_hot_Y = one_hot_Y.T  # Transpose for correct shape
    return one_hot_Y

# Backward propagation to compute gradients
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)  # One-hot encode the labels
    dZ2 = A2 - one_hot_Y  # Compute error for output layer
    dW2 = 1 / m * dZ2.dot(A1.T)  # Gradient for weights of layer 2
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)  # Gradient for biases of layer 2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  # Compute error for hidden layer
    dW1 = 1 / m * dZ1.dot(X.T)  # Gradient for weights of layer 1
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)  # Gradient for biases of layer 1
    return dW1, db1, dW2, db2

def get_predictions(A2):
    """
    Convert the output probabilities into class predictions.
    
    Parameters:
    A2 (numpy.ndarray): The output probabilities from the softmax layer.
    
    Returns:
    numpy.ndarray: The predicted class labels.
    """
    return np.argmax(A2, axis=0)  # Get the index of the max probability for each sample

def get_accuracy(predictions, Y):
    """
    Compute the accuracy of the predictions.
    
    Parameters:
    predictions (numpy.ndarray): The predicted class labels.
    Y (numpy.ndarray): The true class labels.
    
    Returns:
    float: The accuracy as a fraction of correct predictions.
    """
    return np.mean(predictions == Y)  # Compute the fraction of correct predictions


# Initialize Adam optimizer parameters
def init_optimizer_params(W1, b1, W2, b2):
    vW1, vb1, vW2, vb2 = [np.zeros_like(param) for param in [W1, b1, W2, b2]]
    sW1, sb1, sW2, sb2 = [np.zeros_like(param) for param in [W1, b1, W2, b2]]
    return vW1, vb1, vW2, vb2, sW1, sb1, sW2, sb2

# Update parameters using Adam optimizer
def update_params_adam(W1, b1, W2, b2, dW1, db1, dW2, db2, vW1, vb1, vW2, vb2, sW1, sb1, sW2, sb2, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    vW1 = beta1 * vW1 + (1 - beta1) * dW1
    vb1 = beta1 * vb1 + (1 - beta1) * db1
    vW2 = beta1 * vW2 + (1 - beta1) * dW2
    vb2 = beta1 * vb2 + (1 - beta1) * db2
    
    sW1 = beta2 * sW1 + (1 - beta2) * (dW1 ** 2)
    sb1 = beta2 * sb1 + (1 - beta2) * (db1 ** 2)
    sW2 = beta2 * sW2 + (1 - beta2) * (dW2 ** 2)
    sb2 = beta2 * sb2 + (1 - beta2) * (db2 ** 2)
    
    vW1_corr = vW1 / (1 - beta1 ** t)
    vb1_corr = vb1 / (1 - beta1 ** t)
    vW2_corr = vW2 / (1 - beta1 ** t)
    vb2_corr = vb2 / (1 - beta1 ** t)
    
    sW1_corr = sW1 / (1 - beta2 ** t)
    sb1_corr = sb1 / (1 - beta2 ** t)
    sW2_corr = sW2 / (1 - beta2 ** t)
    sb2_corr = sb2 / (1 - beta2 ** t)
    
    W1 -= alpha * vW1_corr / (np.sqrt(sW1_corr) + epsilon)
    b1 -= alpha * vb1_corr / (np.sqrt(sb1_corr) + epsilon)
    W2 -= alpha * vW2_corr / (np.sqrt(sW2_corr) + epsilon)
    b2 -= alpha * vb2_corr / (np.sqrt(sb2_corr) + epsilon)
    
    return W1, b1, W2, b2, vW1, vb1, vW2, vb2, sW1, sb1, sW2, sb2

# Perform gradient descent with Adam optimizer
def gradient_descent_adam(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()  # Initialize parameters
    vW1, vb1, vW2, vb2, sW1, sb1, sW2, sb2 = init_optimizer_params(W1, b1, W2, b2)  # Initialize optimizer parameters
    
    for t in range(1, iterations + 1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)  # Forward propagation
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)  # Backward propagation
        W1, b1, W2, b2, vW1, vb1, vW2, vb2, sW1, sb1, sW2, sb2 = update_params_adam(
            W1, b1, W2, b2, dW1, db1, dW2, db2, vW1, vb1, vW2, vb2, sW1, sb1, sW2, sb2, t, alpha
        )  # Update parameters using Adam optimizer
        
        if t % 10 == 0:  # Print progress every 10 iterations
            print("Iteration: ", t)
            predictions = get_predictions(A2)
            print("Training Accuracy: ", get_accuracy(predictions, Y))
    
    return W1, b1, W2, b2

# Train the model using Adam optimizer
W1, b1, W2, b2 = gradient_descent_adam(X_train, Y_train, 0.001, 500)

# Make predictions using the trained model
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Test predictions and display an image
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]  # Get the image for the given index
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)  # Make prediction
    label = Y_train[index]  # Get true label
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    # Display the image
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Get an index from the user and test the prediction
index = int(input("Enter index: "))

# Check index validity
if index < 0 or index >= X_train.shape[1]:
    print("Invalid index. Please enter an index between 0 and", X_train.shape[1] - 1)
else:
    # Test prediction for the entered index
    test_prediction(index, W1, b1, W2, b2)

# Calculate and print development set accuracy
dev_predictions = get_predictions(forward_prop(W1, b1, W2, b2, X_dev)[-1])
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print("Dev set accuracy: ", dev_accuracy * 100, "%")

# Load and prepare data
def load_data():
    data = pd.read_csv('/Users/harisheswarathas/AI Projects/Recongization/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n] / 255.  # Normalize
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.  # Normalize
    
    return X_train, Y_train, X_dev, Y_dev

def run_demo():
    print("Loading data...")
    X_train, Y_train, X_dev, Y_dev = load_data()
    
    print("Training model...")
    W1, b1, W2, b2 = gradient_descent_adam(X_train, Y_train, 0.001, 500)
    
    # Test predictions on a sample
    index = np.random.randint(0, X_train.shape[1])
    test_prediction(index, W1, b1, W2, b2)
    
    # Print development set accuracy
    dev_predictions = get_predictions(forward_prop(W1, b1, W2, b2, X_dev)[-1])
    dev_accuracy = get_accuracy(dev_predictions, Y_dev)
    print("Development set accuracy: {:.2f}%".format(dev_accuracy * 100))

if __name__ == "__main__":
    run_demo()
