import shutil
import random
from pathlib import Path

# ====================================================
# PATHS
# ====================================================

BASE = Path(r"C:\Users\pendu\OneDrive\Documents\Desktop\GPR_CNN")

# OUTPUT FOLDER (20,000 images)
SOURCE = BASE / "output"

# TRAIN / VAL / TEST
DEST = BASE / "data.set"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

random.seed(42)

# ====================================================
# CREATE FOLDERS
# ====================================================

for split in ["train", "val", "test"]:
    for cls in ["metal", "pvc"]:
        (DEST / split / cls).mkdir(parents=True, exist_ok=True)

# ====================================================
# FIND ALL IMAGES
# ====================================================

metal_files = []
pvc_files = []

for file in SOURCE.rglob("*"):

    if file.suffix.lower() not in IMG_EXTS:
        continue

    name = file.name.lower()

    if name.startswith("metal"):
        metal_files.append(file)

    elif name.startswith("pvc"):
        pvc_files.append(file)

print("\nFOUND:")
print("Metal =", len(metal_files))
print("PVC   =", len(pvc_files))

# ====================================================
# SPLIT FUNCTION
# ====================================================

def split_copy(files, class_name):

    random.shuffle(files)

    total = len(files)

    train_end = int(total * 0.80)
    val_end = int(total * 0.90)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, split_files in split_map.items():

        for src in split_files:

            dst = DEST / split_name / class_name / src.name

            # duplicate filename protection
            if dst.exists():
                dst = DEST / split_name / class_name / (
                    src.parent.name + "_" + src.name
                )

            shutil.copy2(src, dst)

    print(f"\n{class_name.upper()} SPLIT")
    print("Train =", len(train_files))
    print("Val   =", len(val_files))
    print("Test  =", len(test_files))

# ====================================================
# RUN
# ====================================================

split_copy(metal_files, "metal")
split_copy(pvc_files, "pvc")

print("\nDONE ")
print("Dataset created successfully.")