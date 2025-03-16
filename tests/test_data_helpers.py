import matplotlib.pyplot as plt
import numpy as np

from functions.data_helpers import (
    display_examples,
    flatten_2d_images,
)


def test_flatten_2d_images():
    arr = np.arange(24).reshape(2, 3, 4)
    flattened = flatten_2d_images(arr)
    assert flattened.shape == (6, 4)
    np.testing.assert_array_equal(flattened, arr.reshape(6, 4))


def test_display_examples_runs_without_error(monkeypatch):
    xs = np.random.rand(5, 5, 2)
    ys = [0, 1]
    monkeypatch.setattr(plt, "show", lambda: None)
    display_examples([0, 1], xs, ys)
