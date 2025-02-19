import tensorflow as tf
import numpy as np
import cv2

class_labels = ['A', 'B', 'C', ..., 'del', 'nothing', 'space']  # 29 classes

def predict_image(image_path):
    model = tf.keras.models.load_model('models/best_model1.h5')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))
    return class_labels[np.argmax(pred)]

# Test with a sample image
print(predict_image('data/asl_alphabet_test/A_test.jpg'))