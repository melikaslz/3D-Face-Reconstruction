import numpy as np
import pandas as pd
import cv2
from scipy.optimize import least_squares

# Load the keypoints CSV file
def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Load the data
csv_path = "facial_keypoints.csv"
df = load_keypoints(csv_path)
print(df)
