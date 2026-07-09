import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2  # type: ignore[reportMissingImports]
import tensorflow as tf  # type: ignore[reportMissingModuleSource]
from tensorflow.keras.utils import img_to_array  # type: ignore[reportMissingImports]
import numpy as np
from PIL import Image, ImageTk

# =========================================================
# GLOBALS
# =========================================================
apex_click_center = None
def set_apex_center(event):
    global apex_click_center

    coords = preview_click_to_original(event)
    if not coords:
        return

    apex_click_center = coords
    selected_label.config(text=f"Apex set: {coords}")
    print("APEX CENTER SET:", coords)
main_folder = ""
raw_folder = ""
boxed_folder = ""
label_folder = ""
output_folder = ""
raw_images = []          # only unboxed/original images
output_images = []
active_list = []
current_index = 0
current_raw_path = ""
current_boxed_path = ""
current_pil_image = None
image_canvas_photo = None
image_canvas_item = None
zoom_factor = 1.0
raw_preview_photo = None
raw_preview_display_size = (0, 0)
raw_preview_original_size = (0, 0)
current_raw_cv = None
apex_allowed_box = None
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "gpr_cnn_metal_pvc_model.h5"
)
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
# =========================================================
# AUGMENTATION FUNCTIONS
# =========================================================
def add_noise(img, p):
    v = max(int(p * 255 / 100), 1)
    noise = np.random.randint(0, v, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)

def blur_img(img, k): return cv2.GaussianBlur(img, (k, k), 0)
def flip_img(img): return cv2.flip(img, 1)
def brighten(img, p): return cv2.convertScaleAbs(img, alpha=1 + p / 100, beta=int(p / 2))
def rotate_img(img, angle):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
def contrast_img(img, p): return cv2.convertScaleAbs(img, alpha=1 + p / 100, beta=0)
def sharpen_img(img):
    return cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
def clahe_img(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
def motion_blur(img):
    kernel = np.zeros((15, 15)); kernel[7, :] = np.ones(15); kernel /= 15
    return cv2.filter2D(img, -1, kernel)
def gamma_img(img, gamma):
    table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)
def hsv_img(img, scale):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
def median_blur(img): return cv2.medianBlur(img, 5)
def jpeg_compress(img):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return cv2.imdecode(enc, 1)
def color_jitter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
def bilateral_img(img): return cv2.bilateralFilter(img, 9, 75, 75)
def histeq_img(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
def desat_img(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.8, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
def warm_img(img):
    out = img.astype(np.int16); out[:, :, 2] = np.clip(out[:, :, 2] + 15, 0, 255)
    return out.astype(np.uint8)
def cool_img(img):
    out = img.astype(np.int16); out[:, :, 0] = np.clip(out[:, :, 0] + 15, 0, 255)
    return out.astype(np.uint8)
def unsharp_img(img):
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

# =========================================================
# ROOT + STYLE
# =========================================================
root = tk.Tk()
root.title("GPR Image Augmentation Tool - RAW to BOXED Pair Viewer")
root.geometry("1600x900")
root.minsize(1200, 700)
style = ttk.Style(); style.theme_use("clam")
style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
style.map("Treeview", background=[("selected", "#1f6feb")], foreground=[("selected", "white")])

# Variables
noise_vars = {5: tk.IntVar(value=1), 10: tk.IntVar(value=1), 15: tk.IntVar(value=1), 20: tk.IntVar(value=1)}
blur_vars = {5: tk.IntVar(), 9: tk.IntVar(), 15: tk.IntVar()}
bright_vars = {5: tk.IntVar(), 10: tk.IntVar(), 15: tk.IntVar(), 20: tk.IntVar()}
rotate_vars = {1: tk.IntVar(), 2: tk.IntVar(), 3: tk.IntVar(), 4: tk.IntVar()}
contrast_vars = {10: tk.IntVar(), 20: tk.IntVar(), 30: tk.IntVar()}
sharpen_var = tk.IntVar(); clahe_var = tk.IntVar(); motion_var = tk.IntVar()
gamma1_var = tk.IntVar(); gamma2_var = tk.IntVar(); hsv1_var = tk.IntVar(); hsv2_var = tk.IntVar()
median_var = tk.IntVar(); jpeg_var = tk.IntVar(); color_var = tk.IntVar(); bilateral_var = tk.IntVar()
histeq_var = tk.IntVar(); desat_var = tk.IntVar(); warm_var = tk.IntVar(); cool_var = tk.IntVar()
unsharp_var = tk.IntVar(); vflip_var = tk.IntVar()
aug_mode = tk.StringVar(value="selected")

# =========================================================
# HELPERS (move update_count earlier so UI can reference it)
# =========================================================
def update_count():
    if aug_mode.get() == "selected":
        base = 1 if current_raw_path else 0
    else:
        base = len(raw_images)

    total = base
    total += base * sum(v.get() for v in noise_vars.values())
    total += base * sum(v.get() for v in blur_vars.values())
    total += base * sum(v.get() for v in bright_vars.values())
    total += base * sum(v.get() for v in rotate_vars.values()) * 2
    total += base * sum(v.get() for v in contrast_vars.values())

    singles = [
        sharpen_var, clahe_var, motion_var, gamma1_var, gamma2_var,
        hsv1_var, hsv2_var, median_var, jpeg_var, color_var,
        bilateral_var, histeq_var, desat_var, warm_var, cool_var,
        unsharp_var, vflip_var
    ]

    total += base * sum(v.get() for v in singles)
    count_label.config(text=f"TOTAL OUTPUT: {total}")

# =========================================================
# LAYOUT
# =========================================================
main = tk.Frame(root, bg="#f4f6f8"); main.pack(fill="both", expand=True)
main.grid_columnconfigure(0, minsize=360, weight=0)
main.grid_columnconfigure(1, weight=1)
main.grid_columnconfigure(2, minsize=330, weight=0)
main.grid_rowconfigure(0, weight=1)

left_frame = tk.Frame(main, width=360, bg="#eef2f5"); left_frame.grid(row=0, column=0, sticky="nsew"); left_frame.grid_propagate(False)
center = tk.Frame(main, bg="white"); center.grid(row=0, column=1, sticky="nsew"); center.grid_rowconfigure(1, weight=1); center.grid_columnconfigure(0, weight=1)
right_frame = tk.Frame(main, width=330, bg="white", highlightbackground="#d1d5db", highlightthickness=1); right_frame.grid(row=0, column=2, sticky="nsew"); right_frame.grid_propagate(False); right_frame.grid_rowconfigure(1, weight=1); right_frame.grid_columnconfigure(0, weight=1)

# LEFT
header = tk.Label(left_frame, text="GPR DATASET BROWSER", font=("Segoe UI", 12, "bold"), bg="#eef2f5", fg="#111827")
header.pack(fill="x", pady=(14, 4))
folder_label = tk.Label(left_frame, text="Select main dataset folder", font=("Segoe UI", 9), bg="#eef2f5", fg="#4b5563", wraplength=320)
folder_label.pack(fill="x", padx=12, pady=(0, 8))
tk.Button(left_frame, text="📁  UPLOAD DATASET FOLDER", command=lambda: load_folder(), bg="#1f6feb", fg="white", relief="flat", height=2, font=("Segoe UI", 10, "bold"), cursor="hand2").pack(fill="x", padx=12, pady=(0, 8))
image_count_label = tk.Label(left_frame, text="RAW Images: 0", font=("Segoe UI", 9, "bold"), bg="#eef2f5", fg="#0f172a")
image_count_label.pack(fill="x", padx=12, pady=(0, 6))

box = tk.Frame(left_frame, bg="#eef2f5"); box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
tree = ttk.Treeview(box, columns=("path", "kind"), show="tree", displaycolumns=())
ys = ttk.Scrollbar(box, orient="vertical", command=tree.yview); xs = ttk.Scrollbar(box, orient="horizontal", command=tree.xview)
tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
tree.grid(row=0, column=0, sticky="nsew"); ys.grid(row=0, column=1, sticky="ns"); xs.grid(row=1, column=0, sticky="ew")
box.grid_rowconfigure(0, weight=1); box.grid_columnconfigure(0, weight=1)

# CENTER
btn_frame = tk.Frame(center, bg="white"); btn_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
tk.Button(btn_frame, text="◀ Prev", command=lambda: show_previous(), width=10).pack(side="left", padx=4)
tk.Button(btn_frame, text="▶ Next", command=lambda: show_next(), width=10).pack(side="left", padx=4)
tk.Button(btn_frame, text="RUN AUG", command=lambda: start_aug(), bg="#008000", fg="white", width=12).pack(side="left", padx=4)
tk.Button(btn_frame, text="PREDICT", command=lambda: predict_current_crop()).pack(side="left", padx=4)
selected_label = tk.Label(btn_frame, text="No image selected", bg="white", fg="#374151", font=("Segoe UI", 9, "bold")); selected_label.pack(side="left", padx=15)

image_area = tk.Frame(center, bg="white"); image_area.grid(row=1, column=0, sticky="nsew"); image_area.grid_columnconfigure(1, weight=1); image_area.grid_rowconfigure(0, weight=1)
thumb_frame = tk.Frame(image_area, width=300, bg="#ffffff"); thumb_frame.grid(row=0, column=0, sticky="ns"); thumb_frame.grid_propagate(False)
tk.Label(thumb_frame, text="RAW PREVIEW", font=("Segoe UI", 9, "bold"), bg="white").pack(pady=(10, 0))
tk.Label(thumb_frame, text="click hyperbola bend curve → open boxed", font=("Segoe UI", 8), bg="white", fg="#6b7280").pack(pady=(0, 5))
raw_image_label = tk.Label(thumb_frame, text="No Image", bg="white", cursor="hand2", relief="solid", bd=0)
raw_image_label.pack(padx=10, pady=10)
image_name_label = tk.Label(thumb_frame, text="No image selected", font=("Segoe UI", 9, "bold"), bg="white", fg="#111827", wraplength=155)
image_name_label.pack(padx=8, pady=(4, 0))

view_frame = tk.Frame(image_area, bg="#ffffff", highlightbackground="#1f6feb", highlightthickness=1); view_frame.grid(row=0, column=1, sticky="nsew"); view_frame.grid_rowconfigure(0, weight=1); view_frame.grid_columnconfigure(0, weight=1)
image_canvas = tk.Canvas(view_frame, bg="#ffffff", highlightthickness=0, cursor="fleur"); image_canvas.grid(row=0, column=0, sticky="nsew")
image_canvas.create_text(400, 250, text="Upload folder → click RAW image → click RAW PREVIEW to show BOXED image", fill="#111827", font=("Segoe UI", 14), tags="placeholder")

# RIGHT
tr = tk.Frame(right_frame, bg="white"); tr.grid(row=0, column=0, sticky="ew")
tk.Button(tr, text="START AUGMENTATION", bg="#008000", fg="white", command=lambda: start_aug(), relief="flat", height=2, font=("Segoe UI", 9, "bold")).pack(fill="x")
tk.Radiobutton(tr, text="Selected Image Only", variable=aug_mode, value="selected", bg="white", command=update_count).pack(anchor="w", padx=10)
tk.Radiobutton(tr, text="All Images", variable=aug_mode, value="all", bg="white", command=update_count).pack(anchor="w", padx=10)
count_label = tk.Label(tr, text="TOTAL OUTPUT: 0", fg="blue", bg="white", font=("Segoe UI", 10, "bold")); count_label.pack(fill="x", pady=5)
sc = tk.Frame(right_frame, bg="white"); sc.grid(row=1, column=0, sticky="nsew"); sc.grid_rowconfigure(0, weight=1); sc.grid_columnconfigure(0, weight=1)
canvas = tk.Canvas(sc, bg="white", highlightthickness=0); scrollbar = ttk.Scrollbar(sc, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas, bg="white"); canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set); canvas.grid(row=0, column=0, sticky="nsew"); scrollbar.grid(row=0, column=1, sticky="ns")
scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

# =========================================================
# HELPERS
# =========================================================
def detect_folders(selected):
    """
    Auto-detect RAW and BOXED folders for this project.

    Supported structure:
        DATA.SET/
            raw.dataset/   -> unboxed/raw images
            box.dataset/   -> boxed images
            gui.py

    You can select:
        1) DATA.SET folder
        2) raw.dataset folder
        3) box.dataset folder
        4) parent folder that contains DATA.SET

    It will connect:
        raw.dataset/metal_001.png  -> box.dataset/metal_001.png
        raw.dataset/pvc_001.png    -> box.dataset/pvc_001.png
    """
    selected = os.path.abspath(selected)

    raw_names = {"raw.dataset", "original.dataset", "unboxed"}
    box_names = {"box.dataset", "boxed", "box.datset"}

    def child_by_names(parent, names):
        if not os.path.isdir(parent):
            return ""
        for name in os.listdir(parent):
            path = os.path.join(parent, name)
            if os.path.isdir(path) and name.lower() in names:
                return path
        return ""

    name = os.path.basename(selected).lower()

    # Case 1: user selected raw.dataset directly
    if name in raw_names:
        main = os.path.dirname(selected)
        raw = selected
        boxed = child_by_names(main, box_names)
        return main, raw, boxed

    # Case 2: user selected box.dataset directly
    if name in box_names:
        main = os.path.dirname(selected)
        raw = child_by_names(main, raw_names)
        boxed = selected
        return main, raw, boxed

    # Case 3: user selected DATA.SET / main folder directly
    raw = child_by_names(selected, raw_names)
    boxed = child_by_names(selected, box_names)
    if raw and boxed:
        return selected, raw, boxed

    # Case 4: user selected parent folder that contains DATA.SET
    # Search one level deeper for a folder containing raw.dataset and box.dataset
    if os.path.isdir(selected):
        for sub in os.listdir(selected):
            sub_path = os.path.join(selected, sub)
            if not os.path.isdir(sub_path):
                continue
            raw = child_by_names(sub_path, raw_names)
            boxed = child_by_names(sub_path, box_names)
            if raw and boxed:
                return sub_path, raw, boxed

    return selected, "", ""

def collect_images(folder):
    out = []
    for root_dir, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d.lower() != "output"]
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root_dir, f))
    return sorted(out, key=lambda p: os.path.relpath(p, folder).lower())

def label_box_for_raw(raw_path):
    if not label_folder or not os.path.isdir(label_folder):
        return None

    stem = os.path.splitext(os.path.basename(raw_path))[0]
    label_path = os.path.join(label_folder, stem + ".txt")

    if not os.path.isfile(label_path):
        return None

    img = cv2.imread(raw_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return None

    parts = lines[0].strip().split()
    if len(parts) < 5:
        return None

    xc = float(parts[1])
    yc = float(parts[2])
    bw = float(parts[3])
    bh = float(parts[4])

    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    pad = 35

    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad)
    )


def boxed_for_raw(raw_path):
    rel = os.path.relpath(raw_path, raw_folder)
    candidate = os.path.join(boxed_folder, rel)
    if os.path.isfile(candidate):
        return candidate

    name = os.path.basename(raw_path).lower()
    for r, _, files in os.walk(boxed_folder):
        for f in files:
            if f.lower() == name and f.lower().endswith(IMG_EXTS):
                return os.path.join(r, f)
    return ""

def set_middle_message(text, color="#111827"):
    global current_pil_image
    current_pil_image = None
    image_canvas.delete("all")
    image_canvas.create_text(400, 250, text=text, fill=color, font=("Segoe UI", 14, "bold"), justify="center")

def show_raw_preview(raw_path):
    global raw_preview_photo, current_raw_path, current_boxed_path, raw_preview_display_size, raw_preview_original_size, current_raw_cv, apex_allowed_box
    current_raw_path = raw_path
    current_boxed_path = boxed_for_raw(raw_path)
    print("RAW PREVIEW PATH:", current_raw_path)
    print("MATCHED BOX PATH:", current_boxed_path or "NOT FOUND")
    image_name_label.config(text=os.path.basename(raw_path))

    img = cv2.imread(raw_path)
    if img is None:
        current_raw_cv = None
        raw_image_label.config(image="", text="Raw not readable")
        return
    current_raw_cv = img.copy()
    apex_allowed_box = label_box_for_raw(current_raw_path)
    print('DETECTED APEX BOX:', apex_allowed_box)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    raw_preview_original_size = pil.size
    pil.thumbnail((280, 280), Image.LANCZOS)
    raw_preview_display_size = pil.size

    raw_preview_photo = ImageTk.PhotoImage(pil)
    raw_image_label.config(image=raw_preview_photo, text="")
    raw_image_label.image = raw_preview_photo
   
    global current_pil_image
    current_pil_image = None
    image_canvas.delete("all")
    image_canvas.create_text(
    400, 250,
    text="Curve Preview",
    fill="#111827",
    font=("Segoe UI", 14, "bold")
)
    update_count()

def show_big_image(path):
    global current_pil_image, image_canvas_photo, image_canvas_item, zoom_factor
    img = cv2.imread(path)
    if img is None:
        set_middle_message("Boxed image not readable", "#b91c1c")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    current_pil_image = Image.fromarray(img)
    zoom_factor = 1.0
    render_canvas()
    selected_label.config(text="BOXED: " + os.path.basename(path))

def predict_current_crop():
    if not current_boxed_path or not os.path.isfile(current_boxed_path):
        messagebox.showwarning(
            "No Crop Image",
            "First click the hyperbola curve to open the boxed/cropped image."
        )
        return

    try:
        img = cv2.imread(current_boxed_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            messagebox.showerror("Error", "Could not read crop image.")
            return

        img = cv2.resize(img, (128, 128))

        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = cnn_model.predict(img_batch, verbose=0)[0]

        classes = ["metal", "pvc"]
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id]) * 100

        messagebox.showinfo(
            "CNN Prediction",
            f"Prediction: {classes[class_id].upper()}\nConfidence: {confidence:.2f}%"
        )

        selected_label.config(
            text=f"CNN: {classes[class_id].upper()} ({confidence:.2f}%)"
        )

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))
def preview_click_to_original(event):
    dw, dh = raw_preview_display_size
    ow, oh = raw_preview_original_size

    if dw <= 0 or dh <= 0 or ow <= 0 or oh <= 0:
        return None

    label_w = raw_image_label.winfo_width()
    label_h = raw_image_label.winfo_height()

    offset_x = max((label_w - dw) // 2, 0)
    offset_y = max((label_h - dh) // 2, 0)

    px = event.x - offset_x
    py = event.y - offset_y

    if px < 0 or py < 0 or px >= dw or py >= dh:
        return None

    ox = int(px * ow / dw)
    oy = int(py * oh / dh)

    return ox, oy
def detect_apex_allowed_box(img):
    """
    Broad apex-zone detector.
    The previous version was too strict, so curve click was not opening.
    This creates an allowed area around the upper bend region.
    """
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Search region: where apex/bend usually exists
    sx1 = int(w * 0.18)
    sx2 = int(w * 0.88)
    sy1 = int(h * 0.12)
    sy2 = int(h * 0.62)

    roi = gray[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(roi)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    bg = cv2.GaussianBlur(blur, (0, 0), 11)
    detail = cv2.absdiff(blur, bg)
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)

    gx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)

    agx = np.abs(gx)
    agy = np.abs(gy)

    # Curve energy. Suppress horizontal line but keep faint bend.
    energy = agx + 0.60 * np.abs(agx - agy) - 0.35 * agy
    energy[energy < 0] = 0

    if energy.max() <= 0:
        return None

    energy = (energy / energy.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(energy, 10, 255, cv2.THRESH_BINARY)

    # remove long horizontal top/boundary line
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, int(mask.shape[1] * 0.18)), 1)
    )
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    mask = cv2.subtract(mask, horizontal)

    ys, xs = np.where(mask > 0)

    if len(xs) < 8:
        # fallback broad red-mark area
        ax = int(w * 0.55)
        ay = int(h * 0.30)
    else:
        # top bend pixels
        top_y = int(np.percentile(ys, 10))
        band = ys <= top_y + 45

        if np.count_nonzero(band) < 4:
            ax = int(sx1 + np.median(xs))
            ay = int(sy1 + top_y)
        else:
            ax = int(sx1 + np.median(xs[band]))
            ay = int(sy1 + np.median(ys[band]))

    # Bigger allowed zone so real curve click opens
    allowed_x = 210
    allowed_y = 150

    return (
        max(0, ax - allowed_x),
        max(0, ay - allowed_y),
        min(w, ax + allowed_x),
        min(h, ay + allowed_y)
    )

    return True


def open_boxed_only_if_hyperbola_clicked(event):
    global apex_allowed_box
    coords = preview_click_to_original(event)

    if not coords:
        selected_label.config(text="Click inside RAW image")
        return

    ox, oy = coords
    if apex_allowed_box is None:
        selected_label.config(text="Apex region unavailable. Opening boxed image.")
        open_boxed_from_raw_preview(event)
        return True

    x1, y1, x2, y2 = apex_allowed_box
    if x1 <= ox <= x2 and y1 <= oy <= y2:
        open_boxed_from_raw_preview(event)
        return True

    selected_label.config(text="Only apex/bend curve allowed")
    return False

def open_boxed_from_raw_preview(event=None):
    if not current_raw_path:
        messagebox.showinfo("Select RAW", "First click one RAW image from the left list.")
        return
    if not current_boxed_path or not os.path.isfile(current_boxed_path):
        messagebox.showerror("Boxed pair missing", f"No matching boxed image found for:\n{os.path.basename(current_raw_path)}")
        return
    show_big_image(current_boxed_path)

def render_canvas(keep_position=False):
    global image_canvas_photo, image_canvas_item
    if current_pil_image is None: return
    old = image_canvas.coords(image_canvas_item) if keep_position and image_canvas_item else None
    cw, ch = max(image_canvas.winfo_width(), 600), max(image_canvas.winfo_height(), 450)
    iw, ih = current_pil_image.size
    fit = min((cw - 30) / iw, (ch - 30) / ih)
    scale = max(fit * zoom_factor, 0.05)
    preview = current_pil_image.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), Image.LANCZOS)
    image_canvas_photo = ImageTk.PhotoImage(preview)
    image_canvas.delete("all")
    x, y = old if old else (cw // 2, ch // 2)
    image_canvas_item = image_canvas.create_image(x, y, image=image_canvas_photo, anchor="center")
    image_canvas.configure(scrollregion=image_canvas.bbox("all"))


def add_output_tree_node(path):
    """Add one generated/saved output image into the left slider tree."""
    rel = os.path.relpath(path, output_folder)
    parts = rel.split(os.sep)

    root_nodes = tree.get_children("")
    output_parent = None
    for node in root_nodes:
        if tree.item(node, "text").startswith("✅ Augmented Output"):
            output_parent = node
            break

    if output_parent is None:
        output_parent = tree.insert(
            "",
            "end",
            text=f"✅ Augmented Output ({len(output_images)})",
            open=True,
            values=("", "folder")
        )

    current = output_parent
    current_key = "OUTPUT"
    for folder_name in parts[:-1]:
        current_key = current_key + "/" + folder_name
        found = None
        for child in tree.get_children(current):
            if tree.item(child, "text") == f"📁 {folder_name}":
                found = child
                break
        if found is None:
            found = tree.insert(current, "end", text=f"📁 {folder_name}", open=False, values=("", "folder"))
        current = found

    tree.insert(current, "end", text=parts[-1], values=(path, "output"))

    # refresh output parent count
    try:
        tree.item(output_parent, text=f"✅ Augmented Output ({len(output_images)})")
    except Exception:
        pass


def load_existing_outputs():
    """When the app opens again, existing output files are shown in the slider automatically."""
    output_images.clear()
    if not output_folder or not os.path.isdir(output_folder):
        return

    for root_dir, _, files in os.walk(output_folder):
        for file in sorted(files):
            if file.lower().endswith(IMG_EXTS):
                output_images.append(os.path.join(root_dir, file))

    for path in output_images:
        add_output_tree_node(path)


def show_output_image(path):
    """Show augmented output in the big middle view."""
    global active_list, current_index
    active_list = output_images if output_images else [path]
    try:
        current_index = active_list.index(path)
    except ValueError:
        current_index = 0
    show_big_image(path)
    image_name_label.config(text="OUTPUT: " + os.path.basename(path))

def load_folder():
    global main_folder, raw_folder, boxed_folder, output_folder, raw_images, output_images, active_list, current_index
    global label_folder
    selected = filedialog.askdirectory(title="Select MAIN dataset folder (contains raw.dataset and box.dataset)")
    if not selected:
        return
    main_folder, raw_folder, boxed_folder = detect_folders(selected)
    label_folder = os.path.join(main_folder, "labels")
    print("SELECTED:", selected); print("MAIN:", main_folder); print("RAW FOLDER:", raw_folder); print("BOX FOLDER:", boxed_folder)
    if not os.path.isdir(raw_folder) or not os.path.isdir(boxed_folder):
        messagebox.showerror(
            "Folder format problem",
            "Required format:\n\ndataset/\n  raw.dataset/        raw unboxed images\n  box.dataset/        boxed images\n\nSelect the main dataset folder, not any wrong subfolder."
        )
        return
    raw_images = collect_images(raw_folder)
    output_images = []
    active_list = raw_images
    current_index = 0
    output_folder = os.path.join(main_folder, "output")
    tree.delete(*tree.get_children())
    folder_label.config(text=f"RAW: {os.path.basename(raw_folder)} | BOXED: {os.path.basename(boxed_folder)}")
    image_count_label.config(text=f"RAW Images: {len(raw_images)}")
    parent = tree.insert("", "end", text=f"📂 RAW / Unboxed Images ({len(raw_images)})", open=True, values=("", "folder"))
    for i, path in enumerate(raw_images, 1):
        rel_dir = os.path.dirname(os.path.relpath(path, raw_folder))
        # simple flat display to avoid clicking box.dataset by mistake
        label = (
            f"{i:03d}. {os.path.basename(path)}"
            if rel_dir in ("", ".")
            else f"{i:03d}. {rel_dir}/{os.path.basename(path)}"
        )
        tree.insert(parent, "end", text=label, values=(path, "raw"))
    load_existing_outputs()
    update_count()
    if raw_images:
        show_raw_preview(raw_images[0])

def on_tree_select(event=None):
    sel = tree.selection()
    if not sel:
        return
    values = tree.item(sel[0], "values")
    if not values or not values[0]:
        return

    path = values[0]
    kind = values[1] if len(values) > 1 else "raw"

    global current_index, active_list

    if kind == "output":
        show_output_image(path)
        return

    if path in raw_images:
        active_list = raw_images
        current_index = raw_images.index(path)
        show_raw_preview(path)

def show_previous():
    global current_index
    if not active_list:
        return
    current_index = (current_index - 1) % len(active_list)
    path = active_list[current_index]
    if path in raw_images:
        show_raw_preview(path)
    else:
        show_output_image(path)


def show_next():
    global current_index
    if not active_list:
        return
    current_index = (current_index + 1) % len(active_list)
    path = active_list[current_index]
    if path in raw_images:
        show_raw_preview(path)
    else:
        show_output_image(path)

def save_image(img, folder, subfolder, name):
    path_dir = os.path.join(output_folder, folder, subfolder)
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, name)
    cv2.imwrite(path, img)
    output_images.append(path)
    root.after(0, lambda p=path: add_output_tree_node(p))

def process_aug():
    if not raw_images:
        root.after(0, lambda: messagebox.showwarning("No RAW images", "Upload dataset first.")); return
    os.makedirs(output_folder, exist_ok=True)
    output_images.clear()

    def clear_old_output_tree():
        for child in tree.get_children(""):
            if tree.item(child, "text").startswith("✅ Augmented Output"):
                tree.delete(child)
    root.after(0, clear_old_output_tree)
    if aug_mode.get() == "selected":
        if current_raw_path:
            images_to_process = [current_raw_path]
        else:
            root.after(0, lambda: messagebox.showwarning("No image selected", "Select one RAW image first."))
            return
    else:
        images_to_process = raw_images

    for raw_path in images_to_process:
        box_path = boxed_for_raw(raw_path)
        if not box_path: continue
        img = cv2.imread(box_path)  # Read cropped/boxed image for augmentation
        if img is None: continue
        base = os.path.splitext(os.path.basename(raw_path))[0]
        save_image(img, "original", "cropped", os.path.basename(box_path))
        for p, v in noise_vars.items():
            if v.get(): save_image(add_noise(img, p), "noise", str(p), f"{base}_n{p}.jpg")
        for k, v in blur_vars.items():
            if v.get(): save_image(blur_img(img, k), "blur", str(k), f"{base}_b{k}.jpg")
        if vflip_var.get(): save_image(flip_img(img), "flip", "lr", f"{base}_flip.jpg")
        for p, v in bright_vars.items():
            if v.get(): save_image(brighten(img, p), "bright", str(p), f"{base}_br{p}.jpg")
        for a, v in rotate_vars.items():
            if v.get():
                save_image(rotate_img(img, a), "rotate", f"p{a}", f"{base}_r{a}.jpg")
                save_image(rotate_img(img, -a), "rotate", f"n{a}", f"{base}_r-{a}.jpg")
        for p, v in contrast_vars.items():
            if v.get(): save_image(contrast_img(img, p), "contrast", str(p), f"{base}_ct{p}.jpg")
        if sharpen_var.get(): save_image(sharpen_img(img), "sharpen", "normal", f"{base}_sharp.jpg")
        if clahe_var.get(): save_image(clahe_img(img), "clahe", "normal", f"{base}_clahe.jpg")
        if motion_var.get(): save_image(motion_blur(img), "motion", "normal", f"{base}_motion.jpg")
        if gamma1_var.get(): save_image(gamma_img(img, 1.2), "gamma", "12", f"{base}_gamma12.jpg")
        if gamma2_var.get(): save_image(gamma_img(img, 1.5), "gamma", "15", f"{base}_gamma15.jpg")
        if hsv1_var.get(): save_image(hsv_img(img, 1.1), "hsv", "10", f"{base}_hsv10.jpg")
        if hsv2_var.get(): save_image(hsv_img(img, 1.2), "hsv", "20", f"{base}_hsv20.jpg")
        if median_var.get(): save_image(median_blur(img), "median", "normal", f"{base}_median.jpg")
        if jpeg_var.get(): save_image(jpeg_compress(img), "jpeg", "50", f"{base}_jpeg.jpg")
        if color_var.get(): save_image(color_jitter(img), "color", "normal", f"{base}_color.jpg")
        if bilateral_var.get(): save_image(bilateral_img(img), "bilateral", "normal", f"{base}_bilateral.jpg")
        if histeq_var.get(): save_image(histeq_img(img), "histeq", "normal", f"{base}_histeq.jpg")
        if desat_var.get(): save_image(desat_img(img), "desat", "normal", f"{base}_desat.jpg")
        if warm_var.get(): save_image(warm_img(img), "warm", "normal", f"{base}_warm.jpg")
        if cool_var.get(): save_image(cool_img(img), "cool", "normal", f"{base}_cool.jpg")
        if unsharp_var.get(): save_image(unsharp_img(img), "unsharp", "normal", f"{base}_unsharp.jpg")
    def finish_aug():
        global active_list, current_index
        update_count()
        if output_images:
            active_list = output_images
            current_index = 0
            show_output_image(output_images[0])
        messagebox.showinfo("Completed", f"Augmentation completed. Generated files: {len(output_images)}")

    root.after(0, finish_aug)

def start_aug(): threading.Thread(target=process_aug, daemon=True).start()

def zoom_with_mouse(event):
    global zoom_factor
    if current_pil_image is None: return
    zoom_factor = min(zoom_factor * 1.15, 8.0) if event.delta > 0 else max(zoom_factor / 1.15, 0.2)
    render_canvas(keep_position=True)
def start_pan(event): image_canvas.scan_mark(event.x, event.y)
def drag_pan(event): image_canvas.scan_dragto(event.x, event.y, gain=1)
def fit_image_to_canvas(event=None): render_canvas(False)

def section(title): tk.Label(scroll_frame, text=title, font=("Segoe UI", 10, "bold"), bg="white", fg="#1f2937").pack(anchor="w", padx=14, pady=(12, 3))
def check(parent, text, variable): tk.Checkbutton(parent, text=text, variable=variable, command=update_count, bg="white", anchor="w", font=("Segoe UI", 9)).pack(anchor="w", fill="x", padx=20, pady=1)

# Augmentation checkboxes
tk.Label(scroll_frame, text="AUGMENTATIONS", font=("Segoe UI", 12, "bold"), bg="white", fg="#111827").pack(anchor="w", padx=14, pady=(10, 6))
section("Noise")
for p, name in {5:"Gaussian-L",10:"Gaussian-M",15:"Gaussian-H",20:"Gaussian-VH"}.items(): check(scroll_frame, name, noise_vars[p])
section("Blur")
for k, name in {5:"Light Blur",9:"Medium Blur",15:"Strong Blur"}.items(): check(scroll_frame, name, blur_vars[k])
section("Flip"); check(scroll_frame, "Left ↔ Right", vflip_var)
section("Brightness")
for p, name in {5:"B+5",10:"B+10",15:"B+15",20:"B+20"}.items(): check(scroll_frame, name, bright_vars[p])
section("Rotate")
for a in rotate_vars: check(scroll_frame, f"±{a}°", rotate_vars[a])
section("Contrast")
for p in contrast_vars: check(scroll_frame, f"{p}%", contrast_vars[p])
section("Sharpen"); check(scroll_frame, "Sharpen", sharpen_var)
section("CLAHE"); check(scroll_frame, "CLAHE", clahe_var)
section("Motion Blur"); check(scroll_frame, "Motion Blur", motion_var)
section("Gamma"); check(scroll_frame, "G1.2", gamma1_var); check(scroll_frame, "G1.5", gamma2_var)
section("HSV"); check(scroll_frame, "HSV 10", hsv1_var); check(scroll_frame, "HSV 20", hsv2_var)
section("Median Blur"); check(scroll_frame, "Median Blur", median_var)
section("JPEG Compression"); check(scroll_frame, "JPEG 50%", jpeg_var)
section("Color Jitter"); check(scroll_frame, "Color Jitter", color_var)
section("Bilateral"); check(scroll_frame, "Bilateral", bilateral_var)
section("HistEq"); check(scroll_frame, "HistEq", histeq_var)
section("Desaturation"); check(scroll_frame, "Desaturation", desat_var)
section("White Balance"); check(scroll_frame, "Warm", warm_var); check(scroll_frame, "Cool", cool_var)
section("Unsharp"); check(scroll_frame, "Unsharp Mask", unsharp_var)
tk.Label(scroll_frame, text="", bg="white").pack(pady=20)

# Bindings
tree.bind("<<TreeviewSelect>>", on_tree_select)
raw_image_label.bind("<Button-1>", open_boxed_only_if_hyperbola_clicked)
image_canvas.bind("<MouseWheel>", zoom_with_mouse)
image_canvas.bind("<ButtonPress-1>", start_pan)
image_canvas.bind("<B1-Motion>", drag_pan)
image_canvas.bind("<Double-Button-1>", fit_image_to_canvas)
view_frame.bind("<Configure>", lambda e: fit_image_to_canvas() if current_pil_image is not None else None)
root.after(200, update_count)
root.mainloop()
