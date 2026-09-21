import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

CLASS_ICONS = {
    "T-shirt/top": "👕",
    "Trouser": "👖",
    "Pullover": "🧥",
    "Dress": "👗",
    "Coat": "🧥",
    "Sandal": "👡",
    "Shirt": "👔",
    "Sneaker": "👟",
    "Bag": "👜",
    "Ankle boot": "👢"
}

CLASS_DESCRIPTIONS = {
    "T-shirt/top": "Casual short-sleeved tops, tees, and crewnecks.",
    "Trouser": "Pants, trousers, slacks, and denim jeans.",
    "Pullover": "Sweaters, hoodies, jumpers, and knitwear.",
    "Dress": "One-piece dresses, gowns, and frocks.",
    "Coat": "Outerwear jackets, overcoats, parkas, and blazers.",
    "Sandal": "Open-toe sandals, flip-flops, and summer strappy footwear.",
    "Shirt": "Formal button-up or collared dress shirts.",
    "Sneaker": "Athletic sneakers, running shoes, and trainers.",
    "Bag": "Handbags, backpacks, clutches, and luggage bags.",
    "Ankle boot": "Chelsea boots, ankle-high lace-up boots, and boot wear."
}

def load_and_preprocess_data(val_split=0.15, random_seed=42):
    """
    Loads Fashion-MNIST, normalizes pixel intensities to [0, 1],
    expands shape to (N, 28, 28, 1), and creates stratified train/val splits.
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize to [0.0, 1.0]
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Reshape to (N, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Stratified validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_split,
        random_state=random_seed,
        stratify=y_train_full
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
