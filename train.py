# from keras.src.layers import BatchNormalization
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.python.keras.layers import Dropout
#
# from config import *
# from utils.data_loader import load_data
#
#
# def build_model():
#     base_model = MobileNetV2(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
#     )
#     base_model.trainable = False
#
#     # x = base_model.output
#     # x = GlobalAveragePooling2D()(x)
#     # x = Dense(512, activation='relu')(x)
#     # x = Dropout(0.5)(x)
#     # x = BatchNormalization()(x)
#     # predictions = Dense(NUM_CLASSES, activation='softmax')(x)
#     # return Model(inputs=base_model.input, outputs=predictions)
#
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     # x = Dense(512)(x)  # Remove activation here
#     x = BatchNormalization()(x)  # BatchNorm before activation
#     x = tf.keras.layers.ReLU()(x)  # Explicit activation layer
#     x = Dropout(0.5)(x)  # Add dropout after activation
#
#     predictions = Dense(NUM_CLASSES, activation='softmax')(x)
#     return Model(inputs=base_model.input, outputs=predictions)
#
#
# def train():
#     train_generator, val_generator = load_data()
#
#     model = build_model()
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#
#     history = model.fit(
#         train_generator,
#         epochs=EPOCHS,
#         validation_data=val_generator,
#         callbacks=[EarlyStopping(patience=2)],
#         # workers=4,
#         # use_multiprocessing=True
#     )
#
#     model.save('models/best_model1.h5')
#     print("Model saved successfully!")
#
#
# if __name__ == "__main__":
#     train()

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    ReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from config import *
from utils.data_loader import load_data


def build_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)  # No activation
    x = BatchNormalization()(x)  # BatchNorm first
    x = ReLU()(x)  # Then activation
    x = Dropout(0.5)(x)  # Dropout last

    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)


def train():
    train_generator, val_generator = load_data()

    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[EarlyStopping(patience=2)]
    )

    model.save('models/best_model1.h5')
    print("Model saved successfully!")


if __name__ == "__main__":
    train()