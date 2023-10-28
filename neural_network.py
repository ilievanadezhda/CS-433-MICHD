"""This file contains a neural network implementation with all supporting functions."""
import numpy as np
from cross_validation import f1_score


class Layer:

    """
    Each layer performs two things:
    1. Forward pass: Process input to get output: output = layer.forward(input)
    2. Backward pass: Back-propagate gradients through itself: grad_input = layer.backward(input, grad_output)

    The layers that contain learnable parameters also update their parameters during layer.backward.
    """

    def __init__(self):
        """
        Initialize layer parameters.
        """
        pass

    def forward(self, input):
        """
        Takes input data of shape [batch, input_dims], returns output data [batch, output_dims]
        """
        return input

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer.

        We need to apply the chain rule to compute the gradients of the input x:
        d loss / d x  = (d loss / d layer) * (d layer / d x)

        Grad_output provides us d loss / d layer, so we only need to multiply it by d layer / d x.

        Note that if the layer has trainable parameters, we also need to update them using d loss / d layer.
        """

        return grad_output


class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer applies elementwise rectified linear to the elements in the inputs.
        There is nothing to initialize in this simple implementation.
        """

        pass

    def forward(self, input):
        """
        Apply elementwise ReLU to the input
        """

        output = np.maximum(0, input)

        return output

    def backward(self, input, grad_output):
        """
        Compute the gradient of loss.
        """

        grad_input = grad_output * (input > 0)  # derivative od ReLU

        return grad_input


class Fully_connected(Layer):
    def __init__(self, input_dims, output_dims, learning_rate=0.1):
        """
        Fully_connected layer: f(x) = <x*W> + b
        We initialize W, b, and learning rate to update W and b.

        X is an object-feature matrix of shape [batch_size, num_features == input_dims],
        W is a weight matrix [num_features, num_outputs]
        """
        np.random.seed(0)
        self.weights = np.random.randn(input_dims, output_dims) * 0.05
        self.biases = np.zeros(output_dims)
        self.learning_rate = learning_rate

    def forward(self, input):
        """
        Perform f(x) = <x*W> + b .

        input shape: [batch, input_dims]
        output shape: [batch, output_dims]
        """

        output = np.dot(input, self.weights) + self.biases

        return output

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, np.transpose(self.weights))
        self.weights = self.weights - self.learning_rate * np.dot(
            np.transpose(input), grad_output
        )

        self.biases = self.biases - self.learning_rate * np.sum(grad_output, axis=0)

        return grad_input


def softmax_crossentropy_with_logits(logits, reference_answers, weights=[1, 4.5]):
    """
    Compute cross-entropy loss for each sample from output logits [batch,n_classes] and reference_answers [batch].
    Takes into account class weights.
    """

    softmax = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax /= np.sum(softmax, axis=1, keepdims=True)

    batch_size = logits.shape[0]
    predicted_probs = softmax[np.arange(batch_size), reference_answers]
    negative_log_probs = -np.log(predicted_probs) * np.array(
        [weights[ans] for ans in reference_answers]
    )
    xentropy = negative_log_probs.reshape(-1, 1)

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers, weights=[1, 4.5]):
    """
    Compute cross-entropy gradient from output logits [batch,n_classes] and reference_answers [batch].
    Takes into account class weights.
    """

    softmax = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax /= np.sum(softmax, axis=1, keepdims=True)

    one_hot_Y = np.zeros_like(logits)
    one_hot_Y[np.arange(len(logits)), reference_answers] = 1

    grad_input = (
        (-one_hot_Y + softmax)
        * np.array([weights[ans] for ans in reference_answers]).reshape(-1, 1)
        / logits.shape[0]
    )

    return grad_input


def forward(network, X):
    """
    Compute the output of all network layers by applying them sequentially.
    Note that we should return a list of outputs for each layer since we need them for the backward pass.
    """

    outputs = []
    input = X

    for layer in network:
        outputs.append(layer.forward(input))
        input = layer.forward(input)

    assert len(outputs) == len(network)
    return outputs


def predict(network, X):
    """
    Use network to predict the result for each sample. Since we are doing classification, the result should be the index of the most likely class.
    """

    prediction = forward(network, X)[-1]

    return prediction.argmax(axis=-1)


def train(network, X, y):
    """
    Train your network on a given batch of X and y only once.
    Here are the steps to train once:
    1. Run forward to get all layer outputs.
    2. Estimate loss and loss_grad.
    3. Run layer.backward going from last layer to first.

    Note that after you called backward for all layers, the layers with trainable parameters should have already updated.
    """

    layer_outputs = forward(network, X)

    layer_inputs = [
        X
    ] + layer_outputs  # layer_inputs = [X, Layer 1 Output, Layer 2 Output, Layer 3 Output, Layer 4 Output, Layer 5 Output]

    logits = layer_outputs[-1]

    loss = softmax_crossentropy_with_logits(logits, y, weights=[1, 4.5])
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y, weights=[1, 4.5])

    for i in range(len(network) - 1, -1, -1):
        layer = network[i]
        loss_grad = layer.backward(layer_inputs[i], loss_grad)

    return np.mean(loss)


# iterate over minibatches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    np.random.seed(0)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    else:
        indices = np.arange(len(inputs))

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            indexes = indices[start_idx : start_idx + batchsize]
        else:
            indexes = slice(start_idx, start_idx + batchsize)

        yield inputs[indexes], targets[indexes]


# cross validation for neural network
def cross_validation_nn(y, tx, k_indices):
    k = len(k_indices)
    f1s = []
    accuracies = []
    for idx in range(k):
        print(f"Cross validation fold {idx+1}/{k}")
        # get k'th subgroup in test, others in train
        test_idx, train_idx = k_indices[idx], k_indices[np.arange(k) != idx].reshape(-1)

        x_train = tx[train_idx]
        y_train = y[train_idx]
        x_test = tx[test_idx]
        y_test = y[test_idx]

        network = []
        network.append(Fully_connected(x_train.shape[1], 128))
        network.append(ReLU())
        network.append(Fully_connected(128, 2))

        # train network
        for epoch in range(5):
            for x_batch, y_batch in iterate_minibatches(
                x_train, y_train, batchsize=1000, shuffle=True
            ):
                train(network, x_batch, y_batch)
        # predict
        y_pred = predict(network, x_test)
        f1 = f1_score(y_test, y_pred)
        accuracy = np.mean(predict(network, x_test) == y_test)
        f1s.append(f1)
        accuracies.append(accuracy)
        print(f"F1 score: {f1}")
        print(f"Accuracy: {accuracy}")
    print("Average f1 score: ", np.mean(f1s))
    print(" +- ", np.std(f1s))
    print("Average accuracy: ", np.mean(accuracy))
    print(" +- ", np.std(accuracy))
    return np.mean(f1s), np.mean(accuracy)
