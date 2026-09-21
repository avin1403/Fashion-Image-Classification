import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from src.dataset import CLASS_NAMES, CLASS_ICONS, CLASS_DESCRIPTIONS

def preprocess_image(pil_image, invert=False, auto_invert=True):
    """
    Prepares a PIL Image for inference with the Fashion-MNIST CNN:
      1. Converts to grayscale.
      2. Auto-detects light background and inverts to dark background (Fashion-MNIST standard).
      3. Fits inside 24x24 maintaining aspect ratio, then pads to 28x28 (centered).
      4. Normalizes pixel values to [0.0, 1.0].
      5. Reshapes to (1, 28, 28, 1).

    Returns:
      tensor: np.ndarray of shape (1, 28, 28, 1) float32
      preview_img: PIL.Image of shape (28, 28) for display
      was_inverted: bool
    """
    gray_img = pil_image.convert("L")
    img_array = np.array(gray_img)

    h, w = img_array.shape
    corners = [
        img_array[0, 0], img_array[0, -1],
        img_array[-1, 0], img_array[-1, -1]
    ]
    avg_corner = np.mean(corners)

    should_invert = False
    if auto_invert and avg_corner > 130:
        should_invert = True
    elif invert:
        should_invert = True

    if should_invert:
        gray_img = ImageOps.invert(gray_img)

    # Resize preserving aspect ratio into 24x24, then pad to 28x28
    gray_img.thumbnail((24, 24), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    offset_x = (28 - gray_img.width) // 2
    offset_y = (28 - gray_img.height) // 2
    canvas.paste(gray_img, (offset_x, offset_y))

    norm_array = np.array(canvas, dtype=np.float32) / 255.0
    tensor = np.expand_dims(norm_array, axis=(0, -1))

    return tensor, canvas, should_invert

def predict_single(model, tensor):
    """
    Runs model inference on (1, 28, 28, 1) tensor and returns structured prediction results.
    """
    preds = model.predict(tensor, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(preds[top_idx])

    all_probabilities = [
        {
            "class_index": i,
            "class_name": CLASS_NAMES[i],
            "icon": CLASS_ICONS[CLASS_NAMES[i]],
            "probability": float(preds[i]),
            "percentage": f"{preds[i] * 100:.2f}%",
            "description": CLASS_DESCRIPTIONS[CLASS_NAMES[i]]
        }
        for i in range(len(CLASS_NAMES))
    ]
    all_probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "top_class": top_class,
        "top_index": top_idx,
        "confidence": top_prob,
        "icon": CLASS_ICONS[top_class],
        "description": CLASS_DESCRIPTIONS[top_class],
        "ranked_probabilities": all_probabilities,
        "raw_probabilities": preds.tolist()
    }

def generate_gradcam(model, tensor, last_conv_layer_name="conv3", pred_index=None, alpha=0.45):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
    which regions of the input garment the CNN focuses on for its classification.

    Returns:
      superimposed_pil: PIL.Image of superimposed heatmap on the 28x28 input image (upscaled for display)
      heatmap_28: np.ndarray (28, 28) normalized between 0 and 1
    """
    img_tensor = tf.cast(tensor, tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = conv_output[0] @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    cam_np = cam.numpy()

    # Resize heatmap to 28x28
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((28, 28), Image.Resampling.BICUBIC)
    heatmap_28 = np.array(cam_img, dtype=np.float32) / 255.0

    # Colorize heatmap with jet colormap
    jet = plt.get_cmap("jet")
    colored_heatmap = jet(heatmap_28)[:, :, :3]  # shape (28, 28, 3) float [0, 1]

    # Convert grayscale input (28, 28) to RGB
    base_gray = tensor[0, :, :, 0]
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)

    # Blend
    blended = colored_heatmap * alpha + base_rgb * (1.0 - alpha)
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    # Upscale for crisp visual display (140x140)
    superimposed_pil = Image.fromarray(blended).resize((140, 140), Image.Resampling.NEAREST)

    return superimposed_pil, heatmap_28
