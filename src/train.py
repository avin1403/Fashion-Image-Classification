import os
import json
import numpy as np
import tensorflow as tf
from src.dataset import load_and_preprocess_data, CLASS_NAMES
from src.model import create_fashion_cnn, compile_fashion_cnn
from src.evaluate import evaluate_model

def train():
    print("=" * 60)
    print("Starting Fashion-MNIST CNN Training Pipeline")
    print("=" * 60)

    os.makedirs("model", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 1. Load data
    print("Loading and preparing Fashion-MNIST dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(val_split=0.15)
    print(f"Training samples  : {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Testing samples   : {x_test.shape[0]}")
    print(f"Input shape       : {x_train.shape[1:]}")

    # 2. Save sample images from test set for quick Streamlit testing
    print("Caching sample test images for web application...")
    sample_images = []
    sample_labels = []
    samples_per_class = 6
    for class_idx in range(len(CLASS_NAMES)):
        indices = np.where(y_test == class_idx)[0][:samples_per_class]
        for idx in indices:
            sample_images.append(x_test[idx])
            sample_labels.append(class_idx)

    np.savez_compressed(
        "model/sample_test_images.npz",
        images=np.array(sample_images, dtype=np.float32),
        labels=np.array(sample_labels, dtype=np.int32),
        class_names=np.array(CLASS_NAMES)
    )
    print(f"Saved {len(sample_images)} curated test samples to model/sample_test_images.npz")

    # 3. Create and compile model
    print("Building CNN architecture with Regularization (BatchNorm, Dropout, L2)...")
    model = create_fashion_cnn()
    compile_fashion_cnn(model, learning_rate=0.001)

    # 4. Callbacks
    model_save_path = "model/fashion_cnn_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Model training (batch size 128 for efficient multi-core CPU matrix multiplications)
    epochs = 12
    batch_size = 128
    print(f"\nTraining for up to {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save history
    history_dict = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "lr": [float(x) for x in history.history.get("learning_rate", history.history.get("lr", []))]
    }
    with open("model/model_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)
    print("Saved training history to model/model_history.json")

    # Ensure best model is saved
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")

    # 7. Evaluate on test set
    print("\nRunning comprehensive model evaluation on test set...")
    evaluate_model(
        model_path=model_save_path,
        history_path="model/model_history.json",
        output_metrics_path="model/test_metrics.json",
        plots_dir="plots"
    )

if __name__ == '__main__':
    train()
