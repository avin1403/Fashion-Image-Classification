import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_fashion_cnn(input_shape=(28, 28, 1), num_classes=10, l2_factor=1e-4):
    """
    Constructs a streamlined, high-accuracy CNN architecture with regularization:
      - Multi-scale convolutional feature extraction
      - Batch Normalization after each convolution and dense projection
      - Spatial and Dense Dropout layers to curb overfitting
      - L2 weight regularization
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Block 1 (Filters: 32)
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor), name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        layers.Dropout(0.20, name='dropout1'),

        # Block 2 (Filters: 64)
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor), name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),

        # Block 3 (Filters: 128)
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor), name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
        layers.Dropout(0.30, name='dropout3'),

        # Classification Head
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_factor), name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(0.40, name='dropout_dense1'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='Fashion_CNN_Classifier')

    return model

def compile_fashion_cnn(model, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
