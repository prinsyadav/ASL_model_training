from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATASET_PATH, IMG_SIZE, BATCH_SIZE

def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )

    # Training data (no workers/multiprocessing here)
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )

    # Validation data
    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    return train_generator, val_generator