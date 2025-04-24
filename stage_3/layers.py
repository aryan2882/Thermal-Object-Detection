import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_regression_model(input_shape):
    """
    Builds a simple regression model with dense layers.

    Parameters:
        input_shape (tuple): Shape of the input features.

    Returns:
        keras.Model: Compiled regression model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")  # Output layer for regression
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def build_classification_model(input_shape, num_classes):
    """
    Builds a simple classification model with dense layers.

    Parameters:
        input_shape (tuple): Shape of the input features.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled classification model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")  # Output layer for classification
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
