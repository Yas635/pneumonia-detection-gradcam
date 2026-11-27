"""Chargement du modèle, prétraitement et Grad-CAM.

TensorFlow n'est importé qu'à la demande : les tests et les commandes Django
n'ont donc pas besoin de charger la bibliothèque.
"""
import threading
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

CLASSES = ["Normal", "Pneumonia"]
IMAGE_SIZE = (224, 224)

_model = None
_lock = threading.Lock()


class ModelNotAvailable(Exception):
    """Le fichier du modèle est absent ou ne peut pas être chargé."""


def get_model():
    """Retourne le modèle Keras (chargé une seule fois)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                path = Path(settings.MODEL_PATH)
                if not path.exists():
                    raise ModelNotAvailable(
                        f"Modèle introuvable : {path}. Voir models/README.md."
                    )
                try:
                    from tensorflow.keras.models import load_model

                    # compile=False : inutile pour l'inférence, et évite de dépendre
                    # de l'optimiseur personnalisé (AdamW) utilisé à l'entraînement.
                    _model = load_model(path, compile=False)
                except Exception as exc:  # noqa: BLE001
                    raise ModelNotAvailable(f"Impossible de charger le modèle : {exc}") from exc
    return _model


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """Radiographie (octets) -> tenseur (1, 224, 224, 3) normalisé entre 0 et 1."""
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Le fichier n'est pas une image lisible.")
    img = cv2.resize(img, IMAGE_SIZE) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return np.repeat(img, 3, axis=-1)


def predict(model, img: np.ndarray) -> str:
    probs = model.predict(img, verbose=0)
    return CLASSES[int(np.argmax(probs, axis=1)[0])]


def generate_grad_cam(model, img: np.ndarray, output_dir: Path):
    """Calcule la carte Grad-CAM et sa superposition ; retourne leurs chemins."""
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.output, model.get_layer(settings.GRAD_CAM_LAYER).output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img, tf.float32)
        predictions, conv_output = grad_model(inputs)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1).numpy()
    if heatmap.ndim == 3:
        heatmap = heatmap[0]

    # Normalisation (le maximum peut être nul : on évite la division par zéro)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    height, width = img.shape[1:3]
    heatmap_resized = np.uint8(255 * cv2.resize(heatmap, (width, height)))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    base = np.uint8(img[0] * 255)
    overlay = cv2.addWeighted(base, 0.7, heatmap_colored, 0.3, 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    grad_cam_path = output_dir / "grad_cam_image.png"
    overlay_path = output_dir / "overlay_image.png"
    cv2.imwrite(str(grad_cam_path), heatmap_colored)
    cv2.imwrite(str(overlay_path), overlay)
    return grad_cam_path, overlay_path
