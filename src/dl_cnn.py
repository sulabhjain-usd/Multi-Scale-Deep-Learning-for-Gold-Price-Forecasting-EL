import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(64, 64, 3), num_classes=2):
    """
    Build a simple CNN model.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input images (H, W, C).
    num_classes : int
        Number of output classes.
    
    Returns
    -------
    model : tf.keras.Model
        Compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_cnn(model, train_data, val_data, epochs=10):
    """
    Train the CNN model.
    
    Parameters
    ----------
    model : tf.keras.Model
        CNN model to train.
    train_data : tf.data.Dataset
        Training dataset.
    val_data : tf.data.Dataset
        Validation