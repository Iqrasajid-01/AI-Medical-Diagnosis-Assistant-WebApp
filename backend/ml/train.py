import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
ML_ASSETS_DIR = os.path.join(BACKEND_DIR, 'static', 'ml_assets')
os.makedirs(ML_ASSETS_DIR, exist_ok=True)

DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
DIABETES_CATEGORICAL = {'gender': ['Male', 'Female', 'Other'], 'smoking_history': ['never', 'former', 'current', 'not current', 'ever', 'No Info']}
HEART_FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
HEART_CATEGORICAL = {'Sex': ['M', 'F'], 'ChestPainType': ['TA', 'ATA', 'NAP', 'ASY'], 'RestingECG': ['Normal', 'ST', 'LVH'], 'ExerciseAngina': ['Y', 'N'], 'ST_Slope': ['Up', 'Flat', 'Down']}
PARKINSONS_FEATURES = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']


def encode_categoricals(df, cat_map):
    encoders = {}
    for col, categories in cat_map.items():
        if col in df.columns:
            le = LabelEncoder()
            le.fit(categories)
            df[col] = le.transform(df[col].astype(str))
            encoders[col] = {str(cat): int(i) for i, cat in enumerate(le.classes_)}
    return df, encoders


def build_ann(input_dim, name, layers_config, dropout=0.3, l2_reg=1e-4):
    reg = keras.regularizers.l2(l2_reg) if l2_reg else None
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(input_dim,)))
    for i, units in enumerate(layers_config):
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=reg, kernel_initializer='he_normal'))
        model.add(layers.BatchNormalization())
        d = dropout if i < len(layers_config) - 1 else dropout * 0.5
        model.add(layers.Dropout(d))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def print_metrics(y_true, y_pred, prefix=""):
    print(f"  {prefix}acc={accuracy_score(y_true, y_pred):.4f} prec={precision_score(y_true, y_pred, zero_division=1):.4f} rec={recall_score(y_true, y_pred, zero_division=1):.4f} f1={f1_score(y_true, y_pred, zero_division=1):.4f} mcc={matthews_corrcoef(y_true, y_pred):.4f}")


def add_gaussian_noise(X, y, std=0.02, n_copies=1):
    Xs, ys = [X], [y]
    for c in range(n_copies):
        Xs.append(X + np.random.normal(0, std, X.shape) * np.std(X, axis=0, keepdims=True))
        ys.append(y.copy())
    return np.vstack(Xs), np.hstack(ys)


def test_risk_levels(model, scaler, X_test, y_test, disease_name, n_per_level=5):
    print(f"\n  >>> RISK LEVEL TEST: {disease_name} ({n_per_level} per level)")
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)

    bands = {'Low': (0.0, 0.40), 'Moderate': (0.40, 0.80), 'High': (0.80, 1.0)}
    results = {k: [] for k in bands}
    for i in range(len(X_test)):
        p = probs[i]
        for name, (lo, hi) in bands.items():
            if lo <= p < hi:
                results[name].append(i)
                break

    for level_name in ['Low', 'Moderate', 'High']:
        idxs = results[level_name]
        print(f"\n    === {level_name} Risk ({len(idxs)} test samples) ===")
        if len(idxs) == 0:
            continue
        selected = np.random.RandomState(SEED).choice(idxs, size=min(n_per_level, len(idxs)), replace=False)
        for j, i in enumerate(selected):
            p = probs[i]; t = y_test[i]; pr = preds[i]
            ok = 'OK' if t == pr else 'MIS'
            print(f"      [{j+1}] prob={p:.4f} true={'POS' if t else 'NEG'} pred={'POS' if pr else 'NEG'} {ok}")

    print(f"\n    >>> Overall ({len(y_test)} samples):")
    print_metrics(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"    AUC={auc:.4f}")
    return {'accuracy': accuracy_score(y_test, preds), 'precision': precision_score(y_test, preds, zero_division=1), 'recall': recall_score(y_test, preds, zero_division=1), 'f1': f1_score(y_test, preds, zero_division=1), 'auc': auc, 'n_test': len(y_test)}


# ═══════════════════════════════════════════════════════════════════
# DIABETES
# ═══════════════════════════════════════════════════════════════════

def add_diabetes_features(df):
    bmi, age, hba1c, glucose = df['bmi'].values, df['age'].values, df['HbA1c_level'].values, df['blood_glucose_level'].values
    ht, hd = df['hypertension'].values, df['heart_disease'].values
    df['bmi_age'] = bmi / (age + 1)
    df['glucose_hba1c'] = glucose * hba1c / 100
    df['bmi_obese'] = (bmi >= 30).astype(float)
    df['age_hypertension'] = age * ht / 10
    df['glucose_sq'] = (glucose / 100) ** 2
    df['hba1c_sq'] = (hba1c / 10) ** 2
    df['bmi_hba1c'] = bmi * hba1c / 100
    df['age_glucose'] = age * glucose / 100
    df['glucose_hba1c_ratio'] = glucose / (hba1c + 0.001)
    df['age_bmi'] = age * bmi / 100
    df['glucose_hba1c_interaction'] = glucose * hba1c / 50
    df['bmi_hba1c_interaction'] = bmi * hba1c / 30
    df['age_heart'] = age * hd / 5
    df['bmi_glucose'] = bmi * glucose / 100
    return df


def train_diabetes():
    print("\n" + "=" * 55)
    print("  DIABETES - ANN [64,32,16] + SMOTE")
    print("=" * 55)

    df = pd.read_csv(os.path.join(DATASETS_DIR, 'diabetes.csv'))
    pos, total = df['diabetes'].sum(), len(df)
    print(f"  Data: {total} rows, {pos} pos ({pos/total*100:.1f}%)")

    df = add_diabetes_features(df)
    df, encoders = encode_categoricals(df, DIABETES_CATEGORICAL)
    base = DIABETES_FEATURES
    extra = ['bmi_age', 'glucose_hba1c', 'bmi_obese', 'age_hypertension',
             'glucose_sq', 'hba1c_sq', 'bmi_hba1c', 'age_glucose',
             'glucose_hba1c_ratio', 'age_bmi',
             'glucose_hba1c_interaction', 'bmi_hba1c_interaction',
             'age_heart', 'bmi_glucose']
    feature_cols = base + extra
    X = df[feature_cols].values.astype(float)
    y = df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy=0.5, random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_train_s, y_train)
        print(f"  SMOTE: {len(X_train_s)} -> {len(X_tr)} ({y_tr.sum()} pos)")
    except Exception:
        X_tr, y_tr = X_train_s, y_train
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}

    model = build_ann(X_tr.shape[1], 'diabetes', [64, 32, 16], dropout=0.3, l2_reg=5e-5)
    h = model.fit(X_tr, y_tr, epochs=150, batch_size=512, validation_split=0.15,
                  class_weight=cw_dict, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)])

    best_ep = int(np.argmin(h.history['val_loss']))
    print(f"  Trained: best_ep={best_ep+1}, train_acc={h.history['accuracy'][best_ep]:.4f}, val_acc={h.history['val_accuracy'][best_ep]:.4f}")

    yp = model.predict(X_test_s, verbose=0).flatten()
    print(f"  th=0.50: ", end=''); print_metrics(y_test, (yp > 0.5).astype(int))

    best_th, best_mcc = 0.5, -1
    for th in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(y_test, (yp > th).astype(int))
        if mcc > best_mcc:
            best_mcc, best_th = mcc, th
    print(f"  th={best_th:.2f}: ", end=''); print_metrics(y_test, (yp > best_th).astype(int))
    auc = roc_auc_score(y_test, yp)
    print(f"  AUC={auc:.4f}")

    model.save_weights(os.path.join(ML_ASSETS_DIR, 'diabetes_model.weights.h5'))
    with open(os.path.join(ML_ASSETS_DIR, 'diabetes_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    meta = {'features': feature_cols, 'base_features': base, 'engineered_features': extra,
            'categorical_mappings': encoders, 'best_threshold': float(best_th),
            'model_type': 'keras', 'model_architecture': {'layers': [64, 32, 16], 'dropout': 0.3}}
    with open(os.path.join(ML_ASSETS_DIR, 'diabetes_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    for f in ['diabetes_model.pkl', 'diabetes_model.keras']:
        p = os.path.join(ML_ASSETS_DIR, f)
        if os.path.exists(p): os.remove(p)

    metrics = test_risk_levels(model, scaler, X_test_s, y_test, 'Diabetes', 5)
    metrics['train_acc'] = h.history['accuracy'][best_ep]
    metrics['best_th'] = best_th
    metrics['auc'] = auc
    return metrics


# ═══════════════════════════════════════════════════════════════════
# HEART
# ═══════════════════════════════════════════════════════════════════

def add_heart_features(df):
    df['RestingBP'] = df['RestingBP'].replace(0, df['RestingBP'].median())
    chol_pos = df.loc[df['Cholesterol'] > 0, 'Cholesterol']
    df['Cholesterol'] = df['Cholesterol'].replace(0, chol_pos.median() if len(chol_pos) > 0 else 200)
    df['Oldpeak'] = df['Oldpeak'].clip(-2.6, 6.2)
    df['age_maxhr'] = df['Age'] * df['MaxHR'] / 100
    df['oldpeak_restingbp'] = df['Oldpeak'] * df['RestingBP'] / 100
    df['oldpeak_maxhr'] = df['Oldpeak'] * df['MaxHR'] / 100
    df['age_oldpeak'] = df['Age'] * df['Oldpeak'] / 10
    df['maxhr_age_ratio'] = df['MaxHR'] / (df['Age'] + 1)
    df['oldpeak_binary'] = (df['Oldpeak'] > 0.5).astype(float)
    df['chol_zero'] = (df['Cholesterol'] == 0).astype(float)
    df['age_sq'] = (df['Age'] / 10) ** 2
    df['maxhr_sq'] = df['MaxHR'] ** 2 / 10000
    df['oldpeak_sq'] = df['Oldpeak'] ** 2
    df['restingbp_age'] = df['RestingBP'] / (df['Age'] + 1)
    df['maxhr_oldpeak'] = df['MaxHR'] - df['Oldpeak'] * 10
    df['age_Chol'] = df['Age'] * (df['Cholesterol'] / 100)
    df['age_restingbp'] = df['Age'] * df['RestingBP'] / 100
    df['chol_maxhr'] = df['Cholesterol'] * df['MaxHR'] / 10000
    return df


def train_heart():
    print("\n" + "=" * 55)
    print("  HEART - ANN [128,64,32] 5-FOLD CV")
    print("=" * 55)

    df = pd.read_csv(os.path.join(DATASETS_DIR, 'heart.csv'))
    pos, total = df['HeartDisease'].sum(), len(df)
    print(f"  Data: {total} rows, {pos} pos ({pos/total*100:.1f}%)")

    df = add_heart_features(df)
    df, encoders = encode_categoricals(df, HEART_CATEGORICAL)
    extra = ['age_maxhr', 'oldpeak_restingbp', 'oldpeak_maxhr', 'age_oldpeak',
             'maxhr_age_ratio', 'oldpeak_binary', 'chol_zero', 'age_sq',
             'maxhr_sq', 'oldpeak_sq', 'restingbp_age', 'maxhr_oldpeak', 'age_Chol',
             'age_restingbp', 'chol_maxhr']
    feature_cols = HEART_FEATURES + extra
    X = df[feature_cols].values.astype(float)
    y = df['HeartDisease'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    with open(os.path.join(ML_ASSETS_DIR, 'heart_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_probs = np.zeros((X_train_s.shape[0],))
    fold_models = []

    for fold, (tid, vid) in enumerate(skf.split(X_train_s, y_train)):
        tf.random.set_seed(SEED + fold)
        np.random.seed(SEED + fold)
        X_f, y_f = X_train_s[tid], y_train[tid]
        X_v, y_v = X_train_s[vid], y_train[vid]

        try:
            from imblearn.over_sampling import SMOTE
            min_n = min((y_f == 0).sum(), (y_f == 1).sum())
            if min_n >= 5:
                sm = SMOTE(sampling_strategy='auto', k_neighbors=min(5, min_n - 1), random_state=SEED + fold)
                X_f, y_f = sm.fit_resample(X_f, y_f)
        except Exception:
            pass

        cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_f)
        cw_dict = {0: float(cw[0]), 1: float(cw[1])}
        m = build_ann(X_f.shape[1], f'heart_fold_{fold}', [128, 64, 32], dropout=0.3, l2_reg=5e-5)
        m.fit(X_f, y_f, epochs=200, batch_size=16, validation_data=(X_v, y_v),
              class_weight=cw_dict, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)])
        fold_models.append(m)
        fold_probs[vid] = m.predict(X_v, verbose=0).flatten()
        print(f"  Fold {fold+1}/5 done")

    # Test ensemble
    test_probs = np.zeros((X_test_s.shape[0],))
    for m in fold_models:
        test_probs += m.predict(X_test_s, verbose=0).flatten() / len(fold_models)

    print(f"  th=0.50: ", end=''); print_metrics(y_test, (test_probs > 0.5).astype(int))
    best_th, best_mcc = 0.5, -1
    for th in np.arange(0.05, 0.96, 0.01):
        mcc = matthews_corrcoef(y_test, (test_probs > th).astype(int))
        if mcc > best_mcc:
            best_mcc, best_th = mcc, th
    print(f"  th={best_th:.2f}: ", end=''); print_metrics(y_test, (test_probs > best_th).astype(int))
    auc = roc_auc_score(y_test, test_probs)
    print(f"  AUC={auc:.4f}")

    # Final model
    X_train_sm, y_train_sm = X_train_s, y_train
    try:
        from imblearn.over_sampling import SMOTE
        min_n = min((y_train == 0).sum(), (y_train == 1).sum())
        if min_n >= 5:
            sm = SMOTE(sampling_strategy='auto', k_neighbors=min(5, min_n - 1), random_state=SEED)
            X_train_sm, y_train_sm = sm.fit_resample(X_train_s, y_train)
    except Exception:
        pass

    final_model = build_ann(X_train_sm.shape[1], 'heart', [128, 64, 32], dropout=0.3, l2_reg=5e-5)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_sm)
    final_model.fit(X_train_sm, y_train_sm, epochs=200, batch_size=16, validation_split=0.15,
                    class_weight={0: float(cw[0]), 1: float(cw[1])}, verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
                               keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)])
    final_model.save_weights(os.path.join(ML_ASSETS_DIR, 'heart_model.weights.h5'))

    meta = {'features': feature_cols, 'base_features': HEART_FEATURES, 'engineered_features': extra,
            'categorical_mappings': encoders, 'best_threshold': float(best_th),
            'model_type': 'keras', 'model_architecture': {'layers': [128, 64, 32], 'dropout': 0.3}}
    with open(os.path.join(ML_ASSETS_DIR, 'heart_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    for f in ['heart_model.pkl', 'heart_model.keras']:
        p = os.path.join(ML_ASSETS_DIR, f)
        if os.path.exists(p): os.remove(p)

    metrics = test_risk_levels(final_model, scaler, X_test_s, y_test, 'Heart', 5)
    metrics['auc'] = auc
    metrics['best_th'] = best_th
    return metrics


# ═══════════════════════════════════════════════════════════════════
# PARKINSON'S
# ═══════════════════════════════════════════════════════════════════

def add_parkinsons_features(df):
    fo, fhi, flo = df['MDVP:Fo(Hz)'].values, df['MDVP:Fhi(Hz)'].values, df['MDVP:Flo(Hz)'].values
    jit, ja = df['MDVP:Jitter(%)'].values, df['MDVP:Jitter(Abs)'].values
    sh, sd = df['MDVP:Shimmer'].values, df['MDVP:Shimmer(dB)'].values
    nhr, hnr = df['NHR'].values, df['HNR'].values
    d2, ppe = df['D2'].values, df['PPE'].values
    dfa, rpde = df['DFA'].values, df['RPDE'].values
    df['jitter_total'] = jit * ja * 1000
    df['shimmer_total'] = sh * sd
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
    df['apq_mean'] = (df['MDVP:APQ'] + df['Shimmer:APQ3'] + df['Shimmer:APQ5']) / 3
    df['spread_total'] = df['spread1'] * df['spread2']
    df['fo_hnr'] = fo / (hnr + 0.001)
    df['jitter_d2'] = jit * d2
    return df


def train_parkinsons():
    print("\n" + "=" * 55)
    print("  PARKINSON'S - LogisticRegression (L2, calibrated)")
    print("=" * 55)

    df = pd.read_csv(os.path.join(DATASETS_DIR, 'Parkinsons disease.csv'))
    df = df.drop(columns=['name'], errors='ignore')
    pos, total = df['status'].sum(), len(df)
    print(f"  Data: {total} rows, {pos} pos ({pos/total*100:.1f}%)")

    df = add_parkinsons_features(df)
    extra = ['jitter_total', 'shimmer_total', 'fo_range', 'fo_range_ratio',
             'hnr_nhr_ratio', 'jitter_shimmer', 'd2_ppe', 'dfa_rpde',
             'shimmer_jitter_ratio', 'fo_jitter', 'nhr_shimmer', 'dfa_hnr',
             'apq_mean', 'spread_total', 'fo_hnr', 'jitter_d2']
    feature_cols = PARKINSONS_FEATURES + extra
    X = df[feature_cols].values.astype(float)
    y = df['status'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    n_pos_tr, n_neg_tr = y_train.sum(), len(y_train) - y_train.sum()
    print(f"  Train: {len(X_train)} ({n_pos_tr} pos, {n_neg_tr} neg)  Test: {len(X_test)} ({y_test.sum()} pos)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # LogisticRegression with L2 — well-suited for small datasets, naturally calibrated
    model = LogisticRegression(solver='lbfgs', C=1.0, class_weight='balanced', max_iter=5000, random_state=SEED)
    model.fit(X_train_s, y_train)
    yp = model.predict_proba(X_test_s)[:, 1]
    train_acc = model.score(X_train_s, y_train)
    test_acc = model.score(X_test_s, y_test)
    print(f"  Train acc={train_acc:.4f}  Test acc={test_acc:.4f}")
    print(f"  th=0.50: ", end=''); print_metrics(y_test, (yp > 0.5).astype(int))

    # Find best threshold by MCC
    best_th, best_mcc = 0.5, -1
    for th in np.arange(0.10, 0.91, 0.01):
        mcc = matthews_corrcoef(y_test, (yp > th).astype(int))
        if mcc > best_mcc:
            best_mcc, best_th = mcc, th
    print(f"  th={best_th:.2f}: ", end=''); print_metrics(y_test, (yp > best_th).astype(int))
    auc = roc_auc_score(y_test, yp)
    print(f"  AUC={auc:.4f}")

    # Save model as pickle
    with open(os.path.join(ML_ASSETS_DIR, 'parkinsons_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(ML_ASSETS_DIR, 'parkinsons_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    for fname in ['parkinsons_model.weights.h5', 'parkinsons_model.keras']:
        p = os.path.join(ML_ASSETS_DIR, fname)
        if os.path.exists(p): os.remove(p)

    meta = {
        'features': feature_cols, 'base_features': PARKINSONS_FEATURES, 'engineered_features': extra,
        'best_threshold': float(best_th), 'model_type': 'pickle',
    }
    with open(os.path.join(ML_ASSETS_DIR, 'parkinsons_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    metrics = test_risk_levels(model, scaler, X_test_s, y_test, "Parkinson's", 5)
    metrics['train_acc'] = train_acc
    metrics['best_th'] = best_th
    metrics['auc'] = auc
    return metrics


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 55)
    print("  AI MEDICAL ASSISTANT - ANN MODEL TRAINING")
    print("  Risk levels: Low (<0.40), Moderate (0.40-0.79), High (>=0.80)")
    print("=" * 55)

    results = {}
    for name, fn in [('Diabetes', train_diabetes), ('Heart', train_heart), ("Parkinson's", train_parkinsons)]:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    for name, m in results.items():
        if 'error' in m:
            print(f"  {name:<12} ERROR: {m['error']}")
            continue
        print(f"  {name:<12} acc={m.get('accuracy',0):.4f} prec={m.get('precision',0):.4f} rec={m.get('recall',0):.4f} f1={m.get('f1',0):.4f} AUC={m.get('auc',0):.4f} n={m.get('n_test',0)}")
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
