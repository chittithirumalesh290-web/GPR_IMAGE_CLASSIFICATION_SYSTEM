import importlib
import importlib.util
from pathlib import Path

# Project root: GPR_PROJECT.ML
BASE = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE / "model" / "gpr_cnn_metal_pvc_model.h5"
TEST_DIR = BASE / "data.set" / "test"


def _load_ml_framework():
    for module_name in ("tensorflow", "keras"):
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        return module_name, module
    raise ModuleNotFoundError(
        "TensorFlow or Keras is not installed. Install it with `pip install tensorflow`."
    )


framework_name, ml = _load_ml_framework()
keras = ml.keras if framework_name == "tensorflow" else ml

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

model = keras.models.load_model(MODEL_PATH)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(128, 128),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical",
    shuffle=False,
)

loss, acc = model.evaluate(test_ds)

print("TEST ACCURACY:", acc)
print("TEST ACCURACY PERCENT:", round(acc * 100, 2), "%")