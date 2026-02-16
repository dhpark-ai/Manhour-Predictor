import os
import pandas as pd
from stl import mesh
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np



# Paths (same directory as this script)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "DATA")
excel_path = os.path.join(base_dir, "man_hour.csv")

# 엑셀 로드
df = pd.read_csv(excel_path, header=None)
df.columns = ['PartNumber', 'ManHour']
df['PartNumber'] = df['PartNumber'].astype(str)

# STL Feature 추출 함수
def extract_features(stl_path):
    m = mesh.Mesh.from_file(stl_path)
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


# Feature, Label 구성
features, labels = [], []
for part, mh in zip(df['PartNumber'], df['ManHour']):
    path = os.path.join(data_dir, f"{part}.stl")
    if os.path.exists(path):
        feats = extract_features(path)
        features.append(feats)
        labels.append(mh)

# 모델 학습 및 저장
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, labels)

with open(os.path.join(base_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("model.pkl saved successfully.")