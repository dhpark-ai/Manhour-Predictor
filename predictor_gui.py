import tkinter as tk
from tkinter import filedialog, messagebox
from stl import mesh
import pickle
import os
import numpy as np

# Model path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")

# Feature extract
def extract_features_from_stl(file_path):
    m = mesh.Mesh.from_file(file_path)
    vectors = m.vectors

    all_points = m.vectors.reshape(-1,3)
    min_corner = all_points.min(axis=0)
    max_corner = all_points.max(axis=0)
    bbox_size = max_corner - min_corner
    bbox_x, bbox_y, bbox_z = bbox_size

    def triangle_area(v):
        a = v[1] - v[0]
        b = v[2] - v[0]
        return 0.5 * np.linalg.norm(np.cross(a,b))

    triangle_areas = np.array([triangle_area(v) for v in vectors])

    vol = m.get_mass_properties()[0]
    surface_area = triangle_areas.sum()
    triangle_count = len(vectors)
    mean_area = triangle_areas.mean()
    std_area = triangle_areas.std()
    max_area = triangle_areas.max()
    min_area = triangle_areas.min()

    aspect_ratio = bbox_size.max() / bbox_size.min() if bbox_size.min() != 0 else 0
    compactness = (surface_area ** 1.5) / vol if vol != 0 else 0
    
    return [vol, bbox_x, bbox_y, bbox_z, surface_area, triangle_count, mean_area, std_area, max_area, min_area, aspect_ratio, compactness]


# load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# STL selection and prediction
def load_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl")])
    if not file_path:
        return
    try:
        feats = extract_features_from_stl(file_path)
        pred = model.predict([feats])[0]
        messagebox.showinfo("Prediction", f"Predicted man-hours: {pred:.2f} hours")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI
root = tk.Tk()
root.title("STL Manhour Predictor")
root.geometry("300x150")

tk.Label(root, text="Select an STL file to predict man-hours.", wraplength=250).pack(pady=20)
tk.Button(root, text="Load STL", command=load_and_predict, width=20).pack()

root.mainloop()