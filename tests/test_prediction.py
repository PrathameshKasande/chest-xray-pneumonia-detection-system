from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model import build_custom_cnn


def test_model_output_shape():
    model = build_custom_cnn()

    sample = np.random.rand(1, 224, 224, 3).astype("float32")
    prediction = model.predict(sample, verbose=0)

    assert prediction.shape == (1, 1)


def test_model_output_range():
    model = build_custom_cnn()

    sample = np.random.rand(1, 224, 224, 3).astype("float32")
    prediction = model.predict(sample, verbose=0)

    assert 0 <= prediction[0][0] <= 1