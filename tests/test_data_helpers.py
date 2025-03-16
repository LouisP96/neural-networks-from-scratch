import matplotlib.pyplot as plt
import numpy as np

# Import functions from your modules
from functions.data_helpers import (
    display_examples,
    flatten_2d_images,
)


def test_flatten_2d_images():
    # Create a dummy 3D array with shape (height, width, num_examples)
    arr = np.arange(24).reshape(2, 3, 4)
    flattened = flatten_2d_images(arr)
    # Expected shape: (2*3, 4) = (6, 4)
    assert flattened.shape == (6, 4)
    np.testing.assert_array_equal(flattened, arr.reshape(6, 4))


def test_display_examples_runs_without_error(monkeypatch):
    # Create dummy grayscale images: shape (height, width, num_examples)
    xs = np.random.rand(5, 5, 2)
    ys = [0, 1]
    # We override plt.show to prevent blocking
    monkeypatch.setattr(plt, "show", lambda: None)
    # Call display_examples with example indices [0, 1]
    display_examples([0, 1], xs, ys)
