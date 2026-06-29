import pandas as pd, numpy as np

d = pd.read_csv('datasets/diabetes.csv')
print('=== DIABETES ===')
print('Shape:', d.shape)
print('Target:', d['diabetes'].value_counts().to_dict())

h = pd.read_csv('datasets/heart.csv')
print('\n=== HEART ===')
print('Shape:', h.shape)
print('Target:', h['HeartDisease'].value_counts().to_dict())

p = pd.read_csv('datasets/Parkinsons disease.csv')
print('\n=== PARKINSONS ===')
print('Shape:', p.shape)
print('Target:', p['status'].value_counts().to_dict())
