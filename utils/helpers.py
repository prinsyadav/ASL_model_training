import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('models', 'training_history.png'))
    plt.show()


def save_model(model, path='models/best_model.h5'):
    """Save model to specified path."""
    model.save(path)
    print(f"Model saved to {path}")


def load_model(path='models/best_model.h5'):
    """Load model from specified path."""
    return tf.keras.models.load_model(path)


def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess image for prediction."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def visualize_predictions(images, true_labels, pred_labels, class_names):
    """Visualize sample predictions."""
    plt.figure(figsize=(15, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_class_labels(train_generator):
    """Get class labels from training generator."""
    return list(train_generator.class_indices.keys())