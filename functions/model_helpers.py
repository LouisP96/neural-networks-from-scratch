import numpy as np


def shuffle_data(X, Y):
    m = X.shape[1] if X.ndim == 2 else X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    if X.ndim == 2:
        return X[:, indices], Y[indices]
    else:
        return X[indices], Y[indices]


def batchify(array, n, ax, dims):
    """
    Split the dataset into batches of a specified size ready for mini-batch gradient descent
    """
    amount = dims // n
    return np.array_split(array, amount, axis=ax)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def logistic_prime(x):
    return np.multiply(logistic(x), (1 - logistic(x)))


def predict(z):
    if z >= 0.5:
        return 1
    else:
        return 0


def build_model(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # Intilisation using Glorot Initilisation
        parameters["W" + str(l)] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(6 / (layer_dims[l - 1] + layer_dims[l]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_pass(X, parameters):
    caches = []

    A = X
    L = len(parameters) // 2

    # Compute linear and sigmoid at each layer
    for l in range(1, L):
        A_prev = A

        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        # Linear calculation
        Z = np.dot(W, A_prev) + b
        # Store parameters that generate Z, for backprop
        linear_cache = (A_prev, W, b)

        # Sigmoid Activation Function
        A = logistic(Z)
        # Store Z that generates A, for backprop
        activation_cache = Z

        # Store both caches together and add to 'caches'
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    # Same as code in loop, just for the last layer which will be used to compute loss
    A_prev = A

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]

    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    AL = logistic(Z)
    activation_cache = Z

    cache = (linear_cache, activation_cache)
    caches.append(cache)

    return AL, caches


def loss_fun(AL, Y):
    m = Y.shape[0]
    loss = -(1 / m) * np.sum((Y * np.log(AL)) + (1 - Y) * np.log(1 - AL))
    loss = np.squeeze(loss)
    return loss


def model_accuracy(dataX, Y):
    m = Y.shape[0]
    correct = 0

    for i in range(dataX.shape[1]):
        item = np.array(dataX[:, i])
        itemY = Y[i]
        if itemY == predict(item):
            correct += 1

    accuracy = correct / m
    accuracy = np.squeeze(accuracy)
    return accuracy


def backprop(dA, cache):
    linear_cache, activation_cache = cache

    Z = activation_cache
    s = logistic(Z)
    dZ = dA * s * (1 - s)

    # Using this, calculate the other differentials
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, np.transpose(A_prev))
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    return dA_prev, dW, db


def backprop_loop(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # Initialising backprop
    dAL = AL - Y

    # Gradients from the last layer
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backprop(
        dAL, current_cache
    )

    # Gradients from rest of the layers
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backprop(
            grads["dA" + str(l + 2)], current_cache
        )
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(
    parameters, grads, learning_rate, momentum_coefficient, velocity_old
):
    # Number of layers in the neural network. Divide by 2 because parameters contains W & b
    L = len(parameters) // 2
    weight_increment = {}
    velocity_new = {}

    for l in range(L):
        weight_increment["W" + str(l + 1)] = learning_rate * grads["dW" + str(l + 1)]
        weight_increment["b" + str(l + 1)] = learning_rate * grads["db" + str(l + 1)]

        velocity_new["W" + str(l + 1)] = (
            momentum_coefficient * velocity_old["W" + str(l + 1)]
            + weight_increment["W" + str(l + 1)]
        )
        velocity_new["b" + str(l + 1)] = (
            momentum_coefficient * velocity_old["b" + str(l + 1)]
            + weight_increment["b" + str(l + 1)]
        )

        # With momentum
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - velocity_new["W" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - velocity_new["b" + str(l + 1)]
        )

    return parameters, velocity_new


def check_early_stopping(metric_history, window_size=15, threshold=0.0, mode="max"):
    """
    Check if early stopping criteria is met based on the metric history.

    Parameters:
      metric_history : List or numpy array of validation metrics (accuracy or loss).
      window_size    : The number of recent epochs to consider for comparing the metric.
      threshold      : The minimum required improvement between the two windows.
                       For accuracy (mode='max'), stop if recent improvement is less than this threshold.
                       For loss (mode='min'), stop if the decrease is less than this threshold.
      mode           : 'max' if the metric is to be maximized (e.g., accuracy),
                       'min' if the metric is to be minimized (e.g., loss).

    Returns:
      True if early stopping condition is met, False otherwise.
    """
    if len(metric_history) < 2 * window_size:
        return False

    recent_mean = np.mean(metric_history[-window_size:])
    previous_mean = np.mean(metric_history[-2 * window_size : -window_size])

    if mode == "max":
        # For metrics we want to maximize, expect recent_mean to be higher.
        return bool((recent_mean - previous_mean) < threshold)
    elif mode == "min":
        # For metrics we want to minimize, expect recent_mean to be lower.
        return bool((previous_mean - recent_mean) < threshold)
    else:
        raise ValueError("mode must be either 'max' or 'min'")


def three_layer_model(
    X,
    Y,
    devxs,
    devys,
    layers_dims,
    data_dims,
    learning_rate,
    momentum_coefficient,
    max_epochs,
    batch_size,
):
    # Keep track of loss and accuracy of each epoch
    loss = []
    accs = []
    converged = False
    i = 0

    dev_loss = []
    dev_accuracies = []

    best_dev_accuracy = 0

    parameters = build_model(layers_dims)

    # Empty the velocity
    velocity = {
        "W1": np.array([0.0]),
        "W2": np.array([0.0]),
        "W3": np.array([0.0]),
        "b1": np.array([0.0]),
        "b2": np.array([0.0]),
        "b3": np.array([0.0]),
    }

    while (converged == False) and (i <= max_epochs):
        newX, newY = shuffle_data(X, Y)

        batchesX = batchify(newX, batch_size, 1, data_dims)
        batchesY = batchify(newY, batch_size, 0, data_dims)
        acc_total = 0
        loss_total = 0
        for b in range(len(batchesX)):
            bx = batchesX[b]
            by = batchesY[b]

            AL, caches = forward_pass(bx, parameters)
            loss_single_batch = loss_fun(AL, by)
            loss_total += loss_single_batch
            acc_single_batch = model_accuracy(AL, by)
            acc_total += acc_single_batch
            grads = backprop_loop(AL, by, caches)
            parameters, velocity_new = update_parameters(
                parameters, grads, learning_rate, momentum_coefficient, velocity
            )
            velocity = velocity_new

        loss_fulldataset = loss_total / (len(batchesX))
        accuracy_fulldataset = acc_total / (len(batchesX))

        """
        # Full-Batch - uncomment this if you wish to run full batch
        
        
        AL, caches = forward_pass(newX, parameters)
        # prev = parameters
        loss_fulldataset = loss_fun(AL, newY)
        accuracy_fulldataset = model_accuracy(AL, newY)
        grads = backprop_loop(AL, newY, caches)
        parameters, velocity_new = update_parameters(parameters, grads, learning_rate, 
                                                     momentum_coefficient, velocity) 
        velocity = velocity_new
        """

        loss.append(loss_fulldataset)
        accs.append(accuracy_fulldataset)

        # Print out the training loss and accuracy every 100 epochs
        if i % 100 == 0:
            print("Loss after iteration %i: %f" % (i, loss_fulldataset))
            print("Accuracy after iteration %i: %f" % (i, accuracy_fulldataset))

        # Calculate loss and accuracy on validation dataset using current epoch's parameters
        out = forward_pass(devxs, parameters)
        dev_los = loss_fun(out[0], devys)
        dev_accuracy = model_accuracy(out[0], devys)
        if i == 0 or i % 1 == 0:
            dev_loss.append(dev_los)
            dev_accuracies.append(dev_accuracy)

        # Store the epochs results and parameters if it is the highest validation accuracy so far
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_train_accuracy = accuracy_fulldataset
            best_epoch = i
            best_dev_parameters = parameters

        if check_early_stopping(dev_loss, window_size=15, threshold=0.001, mode="min"):
            print("Early stopping: Validation loss has not decreased sufficiently.")
            break

        i += 1

    title = (
        "Best Validation Accuracy: "
        + str(np.round(best_dev_accuracy, 3))
        + " Train Accuracy: "
        + str(np.round(best_train_accuracy, 3))
        + " Type: Mini batch"
        + "\n Learning rate = "
        + str(learning_rate)
    )
    if momentum_coefficient != -1: 
        title += " Momentum = " + str(momentum_coefficient)
    if (
        batch_size != -1
    ):  
        title += " Batch Size = " + str(batch_size)

    best_model = (
        best_dev_accuracy,
        best_train_accuracy,
        best_epoch,
        best_dev_parameters,
    )

    return parameters, best_model, loss, dev_loss, accs, dev_accuracies, title
