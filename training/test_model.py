import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    classification_report = None
    confusion_matrix = None
    accuracy_score = None


def _get_keras_functions():
    for module_name in ("tensorflow", "keras"):
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        keras_mod = module.keras if module_name == "tensorflow" else module
        return (
            keras_mod.models.load_model,
            keras_mod.preprocessing.image.load_img,
            keras_mod.preprocessing.image.img_to_array,
        )
    print("Error: TensorFlow or Keras is required to run this script.")
    print("Install one with `pip install tensorflow` or `pip install keras`.")
    sys.exit(1)

load_model, load_img, img_to_array = _get_keras_functions()

from pathlib import Path

BASE = Path(r"C:\Users\pendu\OneDrive\Documents\Desktop\GPR_PROJECT.ML")
MODEL_PATH = BASE / "model" / "gpr_cnn_metal_pvc_model.h5"
TEST_DIR = Path(
    r"C:\Users\pendu\OneDrive\Documents\Desktop\GPR_CNN\data.set\test"
)

if not MODEL_PATH.exists():
    print(f"Error: Model file not found: {MODEL_PATH}")
    sys.exit(1)
if not TEST_DIR.exists():
    print(f"Error: Test directory not found: {TEST_DIR}")
    sys.exit(1)

IMG_SIZE = (128, 128)
classes = ["metal", "pvc"]

model = load_model(MODEL_PATH)

y_true = []
y_pred = []
confidences = []

for class_id, class_name in enumerate(classes):
    folder = TEST_DIR / class_name

    for img_path in folder.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            continue

        img = load_img(str(img_path), target_size=IMG_SIZE, color_mode="grayscale")
        arr = img_to_array(img)

        # Same as your old code: DON'T divide by 255
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr, verbose=0)[0]

        predicted_id = np.argmax(pred)
        confidence = pred[predicted_id] * 100

        y_true.append(class_id)
        y_pred.append(predicted_id)
        confidences.append(confidence)

print("\n================ TEST RESULTS ================")
if SKLEARN_AVAILABLE:
    print("Test Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
else:
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\n================ CONFUSION MATRIX ================")
if SKLEARN_AVAILABLE:
    cm = confusion_matrix(y_true, y_pred)
else:
    cm = [[0 for _ in classes] for _ in classes]
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
for row in cm:
    print(row)

print("\n================ CLASSIFICATION REPORT ================")
if SKLEARN_AVAILABLE:
    print(classification_report(y_true, y_pred, target_names=classes))
else:
    print("sklearn is not installed; classification report is unavailable.")

print("\n================ SENSITIVITY / RECALL ================")
for i, class_name in enumerate(classes):
    TP = cm[i][i]
    FN = sum(cm[i]) - TP
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    print(f"{class_name} Sensitivity:", round(sensitivity * 100, 2), "%")

print("\nAverage Confidence:", round(np.mean(confidences), 2), "%")
print("\nDONE: Sensitivity test completed on unseen test dataset.")