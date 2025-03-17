import numpy as np
import pytest

from functions.model_helpers import (
    backprop,
    backprop_loop,
    batchify,
    build_model,
    check_early_stopping,
    forward_pass,
    logistic,
    logistic_prime,
    loss_fun,
    model_accuracy,
    predict,
    shuffle_data,
    three_layer_model,
    update_parameters,
)


def test_shuffle_data():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([0, 1, 0])
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    assert X_shuffled.shape == X.shape
    assert Y_shuffled.shape == Y.shape
    assert set(map(tuple, X.T)) == set(map(tuple, X_shuffled.T))


def test_batchify():
    array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    batches = batchify(array, 2, 1, 4)

    assert len(batches) == 2
    assert batches[0].shape == (2, 2)
    assert batches[1].shape == (2, 2)


def test_batchify_validation():
    with pytest.raises(ValueError):
        # Test invalid batch size
        array = np.array([[1, 2], [3, 4]])
        batchify(array, 3, 1, 2)


def test_logistic():
    x = np.array([0])
    assert logistic(x) == 0.5

    x = np.array([10])
    assert logistic(x) > 0.99

    x = np.array([-10])
    assert logistic(x) < 0.01


def test_logistic_prime():
    x = np.array([0])
    assert np.isclose(logistic_prime(x), 0.25)


def test_predict():
    assert predict(0.7) == 1
    assert predict(0.3) == 0
    assert predict(0.5) == 1


def test_build_model():
    layer_dims = [2, 3, 1]
    params = build_model(layer_dims)

    assert "W1" in params
    assert "b1" in params
    assert "W2" in params
    assert "b2" in params
    assert params["W1"].shape == (3, 2)
    assert params["b1"].shape == (3, 1)
    assert params["W2"].shape == (1, 3)
    assert params["b2"].shape == (1, 1)


def test_forward_pass():
    X = np.array([[1, 2], [3, 4]])
    layer_dims = [2, 3, 1]
    params = build_model(layer_dims)
    AL, caches = forward_pass(X, params)

    assert AL.shape == (1, 2)
    assert len(caches) == 2


def test_loss_fun():
    AL = np.array([[0.9, 0.8]])
    Y = np.array([1, 1])
    loss = loss_fun(AL, Y)
    expected = -np.mean([np.log(0.9), np.log(0.8)])
    np.testing.assert_almost_equal(loss, expected)


def test_model_accuracy():
    AL = np.array([[0.9, 0.2, 0.6, 0.4]])
    Y = np.array([1, 0, 1, 0])
    acc = model_accuracy(AL, Y)
    np.testing.assert_almost_equal(acc, 1.0)


def test_check_early_stopping():
    # For "max" mode: simulate a metric that doesn't improve
    metric_history_max = [0.6] * 30
    assert (
        check_early_stopping(
            metric_history_max, window_size=15, threshold=0.01, mode="max"
        )
        is True
    )
    # For "min" mode: simulate a metric that doesn't decrease
    metric_history_min = [0.5] * 30
    assert (
        check_early_stopping(
            metric_history_min, window_size=15, threshold=0.01, mode="min"
        )
        is True
    )


def test_backprop_and_backprop_loop():
    # Test backprop for a single layer.
    A_prev = np.array([[1, 2], [3, 4]])
    W = np.array([[0.5, -0.5]])
    b = np.array([[0]])
    Z = np.dot(W, A_prev) + b
    cache = ((A_prev, W, b), Z)
    dA = np.array([[0.1, -0.2]])
    dA_prev, dW, db = backprop(dA, cache)
    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    # Test backprop_loop with two layers.
    # Create dummy caches for a two-layer network.
    cache1 = (
        (np.array([[1, 1]]), np.array([[0.5]]), np.array([[0]])),
        np.array([[0.1, -0.1]]),
    )
    cache2 = (
        (np.array([[0.6, 0.4]]), np.array([[1]]), np.array([[0]])),
        np.array([[0.2, 0.3]]),
    )

    caches = [cache1, cache2]
    # Create dummy AL and Y for 2 examples.
    AL = np.array([[0.8, 0.3]])
    Y = np.array([1, 0])
    grads = backprop_loop(AL, Y, caches)
    # Check that gradients for each layer exist.
    assert "dA2" in grads and "dW2" in grads and "db2" in grads
    assert "dA1" in grads and "dW1" in grads and "db1" in grads


def test_update_parameters():
    parameters = {
        "W1": np.array([[1.0, -1.0], [0.5, 0.5]]),
        "b1": np.array([[0.0], [0.0]]),
    }
    grads = {
        "dW1": np.array([[0.1, -0.1], [0.05, 0.05]]),
        "db1": np.array([[0.0], [0.0]]),
    }
    learning_rate = 0.1
    momentum_coefficient = 0.9
    velocity_old = {
        "W1": np.zeros_like(parameters["W1"]),
        "b1": np.zeros_like(parameters["b1"]),
    }
    new_params, velocity_new = update_parameters(
        parameters, grads, learning_rate, momentum_coefficient, velocity_old
    )
    assert new_params["W1"].shape == parameters["W1"].shape
    assert new_params["b1"].shape == parameters["b1"].shape
    # Check that velocity_new is non-zero if gradients are non-zero.
    assert np.any(velocity_new["W1"] != 0)


def test_three_layer_model():
    np.random.seed(0)
    n_features = 4
    n_examples = 20
    X = np.random.randn(n_features, n_examples)
    Y = (np.sum(X, axis=0) > 0).astype(int)
    devxs = X[:, :5]
    devys = Y[:5]
    layers_dims = [n_features, 3, 1]
    data_dims = X.shape[1]
    learning_rate = 0.01
    momentum_coefficient = 0.9
    max_epochs = 50
    batch_size = 5
    result = three_layer_model(
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
    )
    parameters, best_model, loss_history, dev_loss_history, accs, dev_accs, title = (
        result
    )
    # Check that best_model is a tuple with 4 elements.
    assert isinstance(best_model, tuple) and len(best_model) == 4
    # Check that loss_history and dev_loss_history are non-empty lists.
    assert len(loss_history) > 0 and len(dev_loss_history) > 0
    # Check that title is a string.
    assert isinstance(title, str)
