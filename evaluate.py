import tensorflow as tf
import numpy as np
from utils.helpers import (
    load_model,
    plot_training_history,
    visualize_predictions,
    get_class_labels,
    preprocess_image
)
from utils.data_loader import load_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model():
    # Load model and data
    model = load_model(path='models/best_model1.h5')
    _, val_generator = load_data()  # Use validation data
    class_labels = get_class_labels(val_generator)

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")

    # Predictions
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    # Confusion Matrix
    plt.figure(figsize=(20, 15))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join('models', 'confusion_matrix.png'))
    plt.show()

    # Per-Class Accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(15, 5))
    sns.barplot(x=class_labels, y=class_acc)
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join('models', 'per_class_accuracy.png'))
    plt.show()

    # Sample Predictions Visualization
    sample_images = []
    sample_true = []
    sample_pred = []

    for i in range(9):  # Get first 9 samples
        batch = next(val_generator)
        images, labels = batch
        preds = model.predict(images)
        sample_images.extend(images)
        sample_true.extend(np.argmax(labels, axis=1))
        sample_pred.extend(np.argmax(preds, axis=1))

    visualize_predictions(sample_images, sample_true, sample_pred, class_labels)


if __name__ == "__main__":
    evaluate_model()