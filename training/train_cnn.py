import sys
try:
    import tensorflow as tf  # type: ignore[import]
    from tensorflow.keras import layers, models  # type: ignore[import]
except ImportError:
    print("Error: TensorFlow is not installed. Install it with 'pip install tensorflow' and rerun.")
    sys.exit(1)
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
BASE_DIR = Path(r"C:\Users\pendu\OneDrive\Documents\Desktop\GPR_CNN")
DATA_DIR = BASE_DIR / "data.set"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# LOAD DATA
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

print("Classes:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =========================
# CNN MODEL
# =========================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =========================
# TEST
# =========================
test_loss, test_acc = model.evaluate(test_ds)
print("TEST ACCURACY:", test_acc)

# =========================
# SAVE MODEL
# =========================
model.save(BASE_DIR / "gpr_cnn_metal_pvc_model.h5")
print("Model saved as gpr_cnn_metal_pvc_model.h5")

# =========================
# SAVE ACCURACY GRAPH
# =========================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(BASE_DIR / "accuracy_graph.png")

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(BASE_DIR / "loss_graph.png")

print("Graphs saved.")