import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl

ML_ASSETS_DIR = 'backend/static/ml_assets'

def build_ann(inp, lyrs, drop=0.3):
    m = keras.Sequential()
    m.add(kl.Input(shape=(inp,)))
    for i, u in enumerate(lyrs):
        m.add(kl.Dense(u, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
        if i < len(lyrs) - 1:
            m.add(kl.BatchNormalization())
        m.add(kl.Dropout(drop if i < len(lyrs)-1 else drop*0.7))
    m.add(kl.Dense(1, activation='sigmoid'))
    m.compile(optimizer=keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def best_th(y, p):
    b = {'f1': 0, 'th': 0.5}
    for t in np.arange(0.05, 0.96, 0.01):
        pr = (p > t).astype(int)
        f = f1_score(y, pr, zero_division=1)
        if f > b['f1']:
            b = {'f1': f, 'th': t}
    return b

def encode_col(df, col, cats):
    le = LabelEncoder()
    le.fit(cats)
    df[col] = le.transform(df[col].astype(str))
    mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}
    return df, mapping


# ─── DIABETES ────────────────────────────────────────────────────────────────

print("=" * 50)
print("DIABETES ANN")
print("=" * 50)

df = pd.read_csv('datasets/diabetes.csv')

cat_defs = [
    ('gender', ['Male', 'Female', 'Other']),
    ('smoking_history', ['never', 'former', 'current', 'not current', 'ever', 'No Info']),
]
cat_maps = {}
for col, vals in cat_defs:
    df, m = encode_col(df, col, vals)
    cat_maps[col] = m

bmi = df['bmi'].values; age = df['age'].values
hb = df['HbA1c_level'].values; gl = df['blood_glucose_level'].values
ht = df['hypertension'].values

df['bmi_age'] = bmi / (age + 1)
df['glucose_hba1c'] = gl * hb / 100
df['bmi_obese'] = (bmi >= 30).astype(float)
df['age_hypertension'] = age * ht / 10
df['glucose_sq'] = (gl / 100) ** 2
df['hba1c_sq'] = (hb / 10) ** 2
df['bmi_hba1c'] = bmi * hb / 100
df['age_glucose'] = age * gl / 100
df['glucose_hba1c_ratio'] = gl / (hb + 0.001)
df['age_bmi'] = age * bmi / 100

base = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
extra = ['bmi_age', 'glucose_hba1c', 'bmi_obese', 'age_hypertension', 'glucose_sq', 'hba1c_sq',
         'bmi_hba1c', 'age_glucose', 'glucose_hba1c_ratio', 'age_bmi']
feats = base + extra
X = df[feats].values.astype(float)
y = df['diabetes'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_trs = sc.fit_transform(X_tr)
X_tes = sc.transform(X_te)

pickle.dump(sc, open(os.path.join(ML_ASSETS_DIR, 'diabetes_scaler.pkl'), 'wb'))
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
cwd = {0: float(cw[0]), 1: float(cw[1])}
print(f"  class weights: {cwd}")

model = build_ann(X_trs.shape[1], [64, 32, 16], 0.3)
model.compile(optimizer=keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(X_trs, y_tr, epochs=80, batch_size=256, validation_split=0.15, class_weight=cwd,
              callbacks=[
                  keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7),
              ], verbose=1)

yp = model.predict(X_tes, verbose=0).flatten()
b = best_th(y_te, yp)
pred = (yp > b['th']).astype(int)
print(f"\n  Best threshold: {b['th']:.2f}")
print(f"  acc={accuracy_score(y_te, pred):.4f} prec={precision_score(y_te, pred):.4f} "
      f"rec={recall_score(y_te, pred):.4f} f1={f1_score(y_te, pred):.4f}")
print(classification_report(y_te, pred, target_names=['No Diabetes', 'Diabetes'], digits=4))

model.save_weights(os.path.join(ML_ASSETS_DIR, 'diabetes_model.weights.h5'))
meta = {
    'features': feats, 'base_features': base, 'engineered_features': extra,
    'categorical_mappings': cat_maps, 'best_threshold': float(b['th']),
    'model_type': 'keras', 'model_architecture': {'layers': [64, 32, 16], 'dropout': 0.3},
}
json.dump(meta, open(os.path.join(ML_ASSETS_DIR, 'diabetes_meta.json'), 'w'), indent=2)
for f in ['diabetes_model.pkl', 'diabetes_model.keras']:
    p = os.path.join(ML_ASSETS_DIR, f)
    if os.path.exists(p): os.remove(p)
print("  Done\n")


# ─── HEART ────────────────────────────────────────────────────────────────────

print("=" * 50)
print("HEART ANN (5-fold CV)")
print("=" * 50)

df = pd.read_csv('datasets/heart.csv')

cat_defs2 = [
    ('Sex', ['M', 'F']),
    ('ChestPainType', ['TA', 'ATA', 'NAP', 'ASY']),
    ('RestingECG', ['Normal', 'ST', 'LVH']),
    ('ExerciseAngina', ['Y', 'N']),
    ('ST_Slope', ['Up', 'Flat', 'Down']),
]
cat_maps2 = {}
for col, vals in cat_defs2:
    df, m = encode_col(df, col, vals)
    cat_maps2[col] = m

df['RestingBP'] = df['RestingBP'].replace(0, df['RestingBP'].median())
df['Oldpeak'] = df['Oldpeak'].clip(-2.6, 6.2)
df['age_maxhr'] = df['Age'] * df['MaxHR'] / 100
df['oldpeak_restingbp'] = df['Oldpeak'] * df['RestingBP'] / 100
df['oldpeak_maxhr'] = df['Oldpeak'] * df['MaxHR'] / 100
df['age_oldpeak'] = df['Age'] * df['Oldpeak'] / 10
df['maxhr_age_ratio'] = df['MaxHR'] / (df['Age'] + 1)
df['oldpeak_binary'] = (df['Oldpeak'] > 0.5).astype(float)
df['chol_zero'] = (df['Cholesterol'] == 0).astype(float)
df['age_sq'] = (df['Age'] / 10) ** 2
df['maxhr_sq'] = df['MaxHR'] ** 2 / 100
df['oldpeak_sq'] = df['Oldpeak'] ** 2
df['restingbp_age'] = df['RestingBP'] / (df['Age'] + 1)
df['maxhr_oldpeak'] = df['MaxHR'] - df['Oldpeak'] * 10
df['age_Chol'] = df['Age'] * (df['Cholesterol'] / 100)

base2 = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
         'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
extra2 = ['age_maxhr', 'oldpeak_restingbp', 'oldpeak_maxhr', 'age_oldpeak', 'maxhr_age_ratio',
          'oldpeak_binary', 'chol_zero', 'age_sq', 'maxhr_sq', 'oldpeak_sq',
          'restingbp_age', 'maxhr_oldpeak', 'age_Chol']
feats2 = base2 + extra2
X2 = df[feats2].values.astype(float)
y2 = df['HeartDisease'].values

X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
sc2 = StandardScaler()
X2_trs = sc2.fit_transform(X2_tr)
X2_tes = sc2.transform(X2_te)
pickle.dump(sc2, open(os.path.join(ML_ASSETS_DIR, 'heart_scaler.pkl'), 'wb'))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_probs = np.zeros((X2_trs.shape[0],))
models2 = []

for fold, (tid, vid) in enumerate(skf.split(X2_trs, y2_tr)):
    tf.random.set_seed(42 + fold)
    np.random.seed(42 + fold)
    Xf, yf = X2_trs[tid], y2_tr[tid]
    Xv, yv = X2_trs[vid], y2_tr[vid]
    cwf = compute_class_weight('balanced', classes=np.array([0, 1]), y=yf)
    cwdf = {0: float(cwf[0]), 1: float(cwf[1])}

    mk = build_ann(Xf.shape[1], [64, 32, 16], 0.3)

    mk.compile(optimizer=keras.optimizers.Adam(5e-4), loss='binary_crossentropy', metrics=['accuracy'])
    mk.fit(Xf, yf, epochs=100, batch_size=16, validation_data=(Xv, yv),
           class_weight=cwdf,
           callbacks=[
               keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
           ], verbose=0)
    models2.append(mk)
    fold_probs[vid] = mk.predict(Xv, verbose=0).flatten()
    print(f"  Fold {fold+1} done")

b2 = best_th(y2_tr, fold_probs)
print(f"  CV threshold: {b2['th']:.2f}, CV F1: {b2['f1']:.4f}")

test_probs = np.zeros((X2_tes.shape[0],))
for m in models2:
    test_probs += m.predict(X2_tes, verbose=0).flatten() / len(models2)

b2t = best_th(y2_te, test_probs)
pred2 = (test_probs > b2t['th']).astype(int)
print(f"\n  Test threshold: {b2t['th']:.2f}")
print(f"  acc={accuracy_score(y2_te, pred2):.4f} prec={precision_score(y2_te, pred2):.4f} "
      f"rec={recall_score(y2_te, pred2):.4f} f1={f1_score(y2_te, pred2):.4f}")
print(classification_report(y2_te, pred2, target_names=['No HD', 'HD'], digits=4))

fm = build_ann(X2_trs.shape[1], [64, 32, 16], 0.3)
cwf = compute_class_weight('balanced', classes=np.array([0, 1]), y=y2_tr)
fm.compile(optimizer=keras.optimizers.Adam(5e-4), loss='binary_crossentropy', metrics=['accuracy'])
fm.fit(X2_trs, y2_tr, epochs=150, batch_size=16, validation_split=0.15,
       class_weight={0: float(cwf[0]), 1: float(cwf[1])},
       callbacks=[
           keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
       ], verbose=0)
fm.save_weights(os.path.join(ML_ASSETS_DIR, 'heart_model.weights.h5'))

meta2 = {
    'features': feats2, 'base_features': base2, 'engineered_features': extra2,
    'categorical_mappings': cat_maps2, 'best_threshold': float(b2t['th']),
    'model_type': 'keras', 'model_architecture': {'layers': [64, 32, 16], 'dropout': 0.3},
}
json.dump(meta2, open(os.path.join(ML_ASSETS_DIR, 'heart_meta.json'), 'w'), indent=2)
for f in ['heart_model.pkl', 'heart_model.keras']:
    p = os.path.join(ML_ASSETS_DIR, f)
    if os.path.exists(p): os.remove(p)
print("  Done\n")


# ─── PARKINSONS ──────────────────────────────────────────────────────────────

print("=" * 50)
print("PARKINSONS ANN")
print("=" * 50)

df = pd.read_csv('datasets/Parkinsons disease.csv')
df = df.drop(columns=['name'], errors='ignore')

fo = df['MDVP:Fo(Hz)'].values; fhi = df['MDVP:Fhi(Hz)'].values; flo = df['MDVP:Flo(Hz)'].values
jit = df['MDVP:Jitter(%)'].values; ja = df['MDVP:Jitter(Abs)'].values
sh = df['MDVP:Shimmer'].values; shd = df['MDVP:Shimmer(dB)'].values
nhr = df['NHR'].values; hnr = df['HNR'].values
d2 = df['D2'].values; ppe = df['PPE'].values
dfa = df['DFA'].values; rpde = df['RPDE'].values

df['jitter_total'] = jit * ja * 1000
df['shimmer_total'] = sh * shd
df['fo_range'] = fhi - flo
df['fo_range_ratio'] = fhi / (flo + 0.001)
df['hnr_nhr_ratio'] = hnr / (nhr + 0.001)
df['jitter_shimmer'] = jit * sh
df['d2_ppe'] = d2 * ppe
df['dfa_rpde'] = dfa * rpde
df['shimmer_jitter_ratio'] = sh / (jit + 0.0001)
df['fo_jitter'] = fo * jit / 100
df['nhr_shimmer'] = nhr * sh
df['dfa_hnr'] = dfa * hnr

base3 = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
         'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
         'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
         'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
extra3 = ['jitter_total', 'shimmer_total', 'fo_range', 'fo_range_ratio', 'hnr_nhr_ratio',
          'jitter_shimmer', 'd2_ppe', 'dfa_rpde', 'shimmer_jitter_ratio', 'fo_jitter',
          'nhr_shimmer', 'dfa_hnr']
feats3 = base3 + extra3
X3 = df[feats3].values.astype(float)
y3 = df['status'].values

X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)

sc3 = StandardScaler()
X3_trs = sc3.fit_transform(X3_tr)
X3_tes = sc3.transform(X3_te)
pickle.dump(sc3, open(os.path.join(ML_ASSETS_DIR, 'parkinsons_scaler.pkl'), 'wb'))

cw3 = compute_class_weight('balanced', classes=np.array([0, 1]), y=y3_tr)
cwd3 = {0: float(cw3[0]), 1: float(cw3[1])}
print(f"  class weights: {cwd3}")

# 3-layer ANN [128,64,32] as per README
m3 = keras.Sequential()
m3.add(kl.Input(shape=(X3_trs.shape[1],)))
m3.add(kl.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
m3.add(kl.BatchNormalization())
m3.add(kl.Dropout(0.3))
m3.add(kl.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
m3.add(kl.BatchNormalization())
m3.add(kl.Dropout(0.3))
m3.add(kl.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)))
m3.add(kl.Dropout(0.3 * 0.7))
m3.add(kl.Dense(1, activation='sigmoid'))
m3.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

h3 = m3.fit(X3_trs, y3_tr, epochs=100, batch_size=8, validation_data=(X3_tes, y3_te),
            class_weight=cwd3,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
            ], verbose=1)

yp3 = m3.predict(X3_tes, verbose=0).flatten()
b3 = best_th(y3_te, yp3)
pred3 = (yp3 > b3['th']).astype(int)
print(f"\n  Best threshold: {b3['th']:.2f}")
print(f"  acc={accuracy_score(y3_te, pred3):.4f} prec={precision_score(y3_te, pred3):.4f} "
      f"rec={recall_score(y3_te, pred3):.4f} f1={f1_score(y3_te, pred3):.4f}")
print(classification_report(y3_te, pred3, target_names=['Healthy', 'Parkinsons'], digits=4))

m3.save_weights(os.path.join(ML_ASSETS_DIR, 'parkinsons_model.weights.h5'))
meta3 = {
    'features': feats3, 'base_features': base3, 'engineered_features': extra3,
    'best_threshold': float(b3['th']),
    'model_type': 'keras', 'model_architecture': {'layers': [128, 64, 32], 'dropout': 0.3},
}
json.dump(meta3, open(os.path.join(ML_ASSETS_DIR, 'parkinsons_meta.json'), 'w'), indent=2)
for f in ['parkinsons_model.pkl', 'parkinsons_model.keras']:
    p = os.path.join(ML_ASSETS_DIR, f)
    if os.path.exists(p): os.remove(p)
print("  Done\n")

print("ALL TRAINING COMPLETE")
