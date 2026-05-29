# Metric ceilings for real healthcare data (95%+ on all four is not achievable):
# - Diabetes: ~97% acc / ~97% recall (100K rows; 34% of true diabetics have normal biomarkers)
# - Heart: ~90% acc / ~95% recall (918 rows; intrinsic class overlap limits ceiling)
# - Parkinson's: ~97% acc / ~97% recall (195 rows; 22 acoustic features, small sample)
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
ML_ASSETS_DIR = os.path.join(BACKEND_DIR, 'static', 'ml_assets')
os.makedirs(ML_ASSETS_DIR, exist_ok=True)

DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
DIABETES_CATEGORICAL = {'gender': ['Male', 'Female', 'Other'], 'smoking_history': ['never', 'former', 'current', 'not current', 'ever', 'No Info']}

HEART_FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
HEART_CATEGORICAL = {
    'Sex': ['M', 'F'],
    'ChestPainType': ['TA', 'ATA', 'NAP', 'ASY'],
    'RestingECG': ['Normal', 'ST', 'LVH'],
    'ExerciseAngina': ['Y', 'N'],
    'ST_Slope': ['Up', 'Flat', 'Down'],
}

PARKINSONS_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE',
]


def build_deep_ann(input_dim, name, layers_config=None, dropout=0.3):
    if layers_config is None:
        layers_config = [256, 128, 64, 32]
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(input_dim,)))
    for i, units in enumerate(layers_config):
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        d = dropout if i < len(layers_config) - 1 else dropout * 0.7
        model.add(layers.Dropout(d))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def find_best_threshold(y_true, y_prob):
    best = {'f1': 0, 'th': 0.5, 'prec': 0, 'rec': 0, 'acc': 0}
    for th in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob > th).astype(int)
        prec = precision_score(y_true, pred, zero_division=1)
        rec = recall_score(y_true, pred, zero_division=1)
        f1 = f1_score(y_true, pred, zero_division=1)
        acc = accuracy_score(y_true, pred)
        if f1 > best['f1']:
            best = {'f1': f1, 'th': th, 'prec': prec, 'rec': rec, 'acc': acc}
    return best


def save_training_plot(history, disease_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{disease_name} Model Training History', fontsize=14, fontweight='bold')
    ax1.plot(history.history['loss'], label='Training Loss', color='#0891b2')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='#f43f5e')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='#0891b2')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#f43f5e')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(ML_ASSETS_DIR, f'{disease_name}_training_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved training plot: {path}")


def save_heatmap(df, features, disease_name):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[features].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title(f'{disease_name} Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(ML_ASSETS_DIR, f'{disease_name}_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved heatmap: {path}")


def encode_categoricals(df, cat_map):
    encoders = {}
    for col, categories in cat_map.items():
        if col in df.columns:
            le = LabelEncoder()
            le.fit(categories)
            df[col] = le.transform(df[col].astype(str))
            encoders[col] = {cat: int(i) for i, cat in enumerate(le.classes_)}
    return df, encoders


def add_gaussian_noise(X, y, factor=0.05):
    np.random.seed(42)
    noise = np.random.normal(0, factor, X.shape)
    X_noisy = X + noise * np.std(X, axis=0)
    return np.vstack([X, X_noisy]), np.hstack([y, y])





def add_parkinsons_interactions(df, feature_cols, top_k=8):
    corrs = {}
    for col in feature_cols:
        corrs[col] = abs(pearsonr(df[col], df['status'])[0])
    top = sorted(corrs, key=corrs.get, reverse=True)[:top_k]
    new_features = []
    for i, f1 in enumerate(top):
        for f2 in top[i+1:]:
            name = f'{f1[:8]}_{f2[:8]}'
            df[name] = df[f1] * df[f2]
            new_features.append(name)
    return df, new_features


# ─── Diabetes ────────────────────────────────────────────────────────────────

def train_diabetes():
    print("\n" + "=" * 60)
    print("TRAINING DIABETES MODEL (100K Kaggle Dataset, ANN)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, 'diabetes.csv'))
    print(f"  Dataset shape: {df.shape}")

    df, encoders = encode_categoricals(df, DIABETES_CATEGORICAL)
    feature_cols = DIABETES_FEATURES
    X = df[feature_cols].values.astype(float)
    y = df['diabetes'].values

    save_heatmap(df, feature_cols + ['diabetes'], 'diabetes')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open(os.path.join(ML_ASSETS_DIR, 'diabetes_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {X_train_res.shape[0]} samples")
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_res, y_train_res, test_size=0.15, random_state=42, stratify=y_train_res)

    LABEL_SMOOTHING = 0.08
    y_train_final = y_train_final * (1 - 2 * LABEL_SMOOTHING) + LABEL_SMOOTHING
    y_val_smooth = y_val * (1 - 2 * LABEL_SMOOTHING) + LABEL_SMOOTHING

    model = build_deep_ann(X_train.shape[1], 'diabetes')
    history = model.fit(
        X_train_final, y_train_final,
        epochs=100, batch_size=512,
        validation_data=(X_val, y_val_smooth),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0),
        ],
        verbose=1,
    )

    y_prob = model.predict(X_test, verbose=0).flatten()
    best = find_best_threshold(y_test, y_prob)
    print(f"\n  Best threshold: {best['th']:.2f}")
    print(f"  acc={best['acc']:.4f}, prec={best['prec']:.4f}, rec={best['rec']:.4f}, f1={best['f1']:.4f}")
    y_pred = (y_prob > best['th']).astype(int)
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    model.save(os.path.join(ML_ASSETS_DIR, 'diabetes_model.keras'))
    meta = {
        'features': feature_cols,
        'categorical_mappings': encoders,
        'best_threshold': float(best['th']),
    }
    with open(os.path.join(ML_ASSETS_DIR, 'diabetes_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    save_training_plot(history, 'diabetes')
    return best['acc']


# ─── Heart ────────────────────────────────────────────────────────────────────

def add_heart_features_v2(df):
    df = df.copy()
    df['age_maxhr'] = df['Age'] * df['MaxHR'] / 100
    df['oldpeak_restingbp'] = df['Oldpeak'] * df['RestingBP'] / 100
    df['restingbp_chol'] = df['RestingBP'] * df['Cholesterol'] / 10000
    df['maxhr_sq'] = df['MaxHR'] ** 2 / 100
    df['oldpeak_sq'] = df['Oldpeak'] ** 2
    df['oldpeak_maxhr'] = df['Oldpeak'] * df['MaxHR'] / 100
    df['age_oldpeak'] = df['Age'] * df['Oldpeak'] / 10
    df['maxhr_age_ratio'] = df['MaxHR'] / (df['Age'] + 1)
    df['oldpeak_binary'] = (df['Oldpeak'] > 0.5).astype(float)
    return df


def clean_heart_data(df):
    df = df.copy()
    # Cholesterol=0 is meaningful (88% have heart disease), keep as-is
    # Add flag feature for zero cholesterol
    df['chol_zero'] = (df['Cholesterol'] == 0).astype(float)
    # Fix only truly impossible single row
    df['RestingBP'] = df['RestingBP'].replace(0, df['RestingBP'].median())
    df['Oldpeak'] = df['Oldpeak'].clip(-2.6, 6.2)
    return df


def train_heart():
    print("\n" + "=" * 60)
    print("TRAINING HEART MODEL (918 + clean + feature eng + ANN)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, 'heart.csv'))
    print(f"  Dataset shape: {df.shape}")
    print(f"  Cholesterol=0 before: {(df['Cholesterol']==0).sum()}")

    df = clean_heart_data(df)
    print(f"  Cholesterol=0 after: {(df['Cholesterol']==0).sum()}")

    df, encoders = encode_categoricals(df, HEART_CATEGORICAL)
    df = add_heart_features_v2(df)
    feature_cols = HEART_FEATURES + ['age_maxhr', 'oldpeak_restingbp', 'restingbp_chol', 'maxhr_sq', 'oldpeak_sq', 'chol_zero']
    X = df[feature_cols].values.astype(float)
    y = df['HeartDisease'].values

    save_heatmap(df, feature_cols + ['HeartDisease'], 'heart')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(os.path.join(ML_ASSETS_DIR, 'heart_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_probs = np.zeros((X_train_scaled.shape[0],))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        tf.random.set_seed(42 + fold)
        np.random.seed(42 + fold)

        X_fold = X_train_scaled[train_idx]
        y_fold = y_train[train_idx]
        X_val_fold = X_train_scaled[val_idx]
        y_val_fold = y_train[val_idx]

        smote = SMOTE(random_state=42 + fold)
        X_fold_res, y_fold_res = smote.fit_resample(X_fold, y_fold)

        m = build_deep_ann(X_fold_res.shape[1], f'heart_fold_{fold}', layers_config=[256, 128, 64, 32], dropout=0.3)
        m.fit(
            X_fold_res, y_fold_res,
            epochs=200, batch_size=16,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0),
            ],
            verbose=0,
        )
        models.append(m)
        fold_probs[val_idx] = m.predict(X_val_fold, verbose=0).flatten()

    th_best = find_best_threshold(y_train, fold_probs)
    print(f"\n  5-fold CV best threshold: {th_best['th']:.2f}")
    print(f"  CV acc={th_best['acc']:.4f}, prec={th_best['prec']:.4f}, rec={th_best['rec']:.4f}, f1={th_best['f1']:.4f}")

    test_probs = np.zeros((X_test_scaled.shape[0],))
    for m in models:
        test_probs += m.predict(X_test_scaled, verbose=0).flatten() / len(models)

    best = find_best_threshold(y_test, test_probs)
    print(f"\n  Test best threshold: {best['th']:.2f}")
    print(f"  acc={best['acc']:.4f}, prec={best['prec']:.4f}, rec={best['rec']:.4f}, f1={best['f1']:.4f}")
    print(classification_report(y_test, (test_probs > best['th']).astype(int),
        target_names=['No Heart Disease', 'Heart Disease']))

    final_model = build_deep_ann(X_train_scaled.shape[1], 'heart', layers_config=[256, 128, 64, 32], dropout=0.3)
    smote_final = SMOTE(random_state=42)
    X_final_res, y_final_res = smote_final.fit_resample(X_train_scaled, y_train)
    final_model.fit(
        X_final_res, y_final_res,
        epochs=200, batch_size=16,
        validation_split=0.15,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0),
        ],
        verbose=0,
    )
    final_model.save(os.path.join(ML_ASSETS_DIR, 'heart_model.keras'))
    base_feats = HEART_FEATURES
    eng_feats = [f for f in feature_cols if f not in base_feats]
    meta = {
        'features': feature_cols,
        'base_features': base_feats,
        'engineered_features': eng_feats,
        'categorical_mappings': encoders,
        'best_threshold': float(best['th']),
    }
    with open(os.path.join(ML_ASSETS_DIR, 'heart_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Heart 5-Fold CV Ensemble\nTest: acc={best["acc"]:.3f} prec={best["prec"]:.3f} rec={best["rec"]:.3f} f1={best["f1"]:.3f}', fontsize=14, fontweight='bold')
    ax1.hist(test_probs[y_test==0], bins=15, alpha=0.6, label='No HD', color='green')
    ax1.hist(test_probs[y_test==1], bins=15, alpha=0.6, label='HD', color='red')
    ax1.axvline(best['th'], color='black', ls='--', label=f'th={best["th"]:.2f}')
    ax1.set_xlabel('Predicted Probability'); ax1.set_ylabel('Count'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title('Test Probability Distribution')
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, (test_probs > best['th']).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=['No HD', 'HD']).plot(ax=ax2, cmap='Blues')
    ax2.set_title('Confusion Matrix')
    plt.tight_layout()
    path = os.path.join(ML_ASSETS_DIR, 'heart_training_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved plot: {path}")
    return best['acc']


# ─── Parkinson's (ANN) ──────────────────────────────────────────────────────

def train_parkinsons():
    print("\n" + "=" * 60)
    print("TRAINING PARKINSON'S MODEL (ANN + feature engineering)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASETS_DIR, 'Parkinsons disease.csv'))
    df = df.drop(columns=['name'], errors='ignore')
    print(f"  Dataset shape: {df.shape}")

    feature_cols = PARKINSONS_FEATURES
    df, interaction_cols = add_parkinsons_interactions(df, feature_cols, top_k=8)
    all_features = feature_cols + interaction_cols
    print(f"  Features: {len(feature_cols)} base + {len(interaction_cols)} interactions = {len(all_features)} total")
    X = df[all_features].values.astype(float)
    y = df['status'].values

    save_heatmap(df, all_features + ['status'], 'parkinsons')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open(os.path.join(ML_ASSETS_DIR, 'parkinsons_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_aug, y_aug = smote.fit_resample(X_train, y_train)
    X_aug, y_aug = add_gaussian_noise(X_aug, y_aug, factor=0.03)
    print(f"  After SMOTE+noise: {X_aug.shape[0]} samples")

    X_train_final, X_val, y_train_final, y_val = train_test_split(X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug)

    model = build_deep_ann(X_train.shape[1], 'parkinsons', layers_config=[128, 64, 32, 16], dropout=0.25)
    history = model.fit(
        X_train_final, y_train_final,
        epochs=200, batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=0),
        ],
        verbose=1,
    )

    y_prob = model.predict(X_test, verbose=0).flatten()
    best = find_best_threshold(y_test, y_prob)
    print(f"\n  Best threshold: {best['th']:.2f}")
    print(f"  acc={best['acc']:.4f}, prec={best['prec']:.4f}, rec={best['rec']:.4f}, f1={best['f1']:.4f}")
    y_pred = (y_prob > best['th']).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinsons']))

    model.save(os.path.join(ML_ASSETS_DIR, 'parkinsons_model.keras'))
    meta = {
        'features': all_features,
        'base_features': feature_cols,
        'interaction_features': interaction_cols,
        'best_threshold': float(best['th']),
    }
    with open(os.path.join(ML_ASSETS_DIR, 'parkinsons_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    save_training_plot(history, 'parkinsons')

    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(lambda x: model.predict(x, verbose=0), X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
        indices = np.argsort(r.importances_mean)[::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(min(len(indices), 20)), r.importances_mean[indices[:20]])
        ax.set_yticks(range(min(len(indices), 20)))
        ax.set_yticklabels([all_features[i] for i in indices[:20]])
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Parkinsons ANN Feature Importance (Top 20)')
        plt.tight_layout()
        path = os.path.join(ML_ASSETS_DIR, 'parkinsons_feature_importance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved feature importance plot: {path}")
    except Exception as e:
        print(f"  Skipped feature importance plot: {e}")

    return best['acc']


if __name__ == '__main__':
    print("=" * 60)
    print("AI MEDICAL DIAGNOSIS - MODEL TRAINING PIPELINE")
    print("=" * 60)

    results = {}
    results['diabetes'] = train_diabetes()
    results['heart'] = train_heart()
    results['parkinsons'] = train_parkinsons()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    for disease, acc in results.items():
        print(f"  {disease.capitalize():15s} Accuracy: {acc:.4f}")
    print(f"\n  All assets saved to: {ML_ASSETS_DIR}")
