# Paths
DATASET_PATH = "./data/asl_alphabet_train/"
MODEL_SAVE_PATH = "./models/best_model.h5"

# Training Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 29  # 26 letters + space, del, nothing

# Model Parameters
BASE_MODEL = "MobileNetV2"
FREEZE_LAYERS = True
LEARNING_RATE = 1e-4