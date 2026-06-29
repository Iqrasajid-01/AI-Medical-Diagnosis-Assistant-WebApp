import os, json, pickle, zipfile, tempfile
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERLAS'] = '0'
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
ML_ASSETS_DIR = os.path.join(BACKEND_DIR, 'static', 'ml_assets')

MODEL_CONFIGS = {
    'diabetes':   {'layers': [256, 128, 64, 32], 'dropout': 0.3},
    'heart':      {'layers': [256, 128, 64, 32], 'dropout': 0.3},
    'parkinsons': {'layers': [128, 64, 32, 16],  'dropout': 0.25},
}

_cache = {}

def _get_asset_path(filename):
    return os.path.join(ML_ASSETS_DIR, filename)

def _build_model(input_dim, cfg, name):
    layers_cfg = cfg['layers']
    dropout = cfg['dropout']
    model = keras.Sequential(name=name)
    model.add(layers.Input(shape=(input_dim,)))
    for i, units in enumerate(layers_cfg):
        model.add(layers.Dense(units, activation='relu'))
        if i < len(layers_cfg) - 1:
            model.add(layers.BatchNormalization())
        d = dropout if i < len(layers_cfg) - 1 else dropout * 0.7
        model.add(layers.Dropout(d))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def _encode_input(input_data, meta, categorical_mappings):
    features = meta['features']
    vals = []
    for col in features:
        val = input_data.get(col)
        if categorical_mappings and col in categorical_mappings:
            mapping = categorical_mappings[col]
            if val is not None and str(val) in mapping:
                val = mapping[str(val)]
            else:
                val = 0.0
        if val is None or val == '' or val == 'null':
            val = 0.0
        vals.append(float(val))
    return np.array([vals])

def _encode_diabetes_input(input_data, meta, cat_map):
    base_feats = meta.get('base_features', [])
    vals = []
    for col in base_feats:
        val = input_data.get(col)
        if cat_map and col in cat_map:
            mapping = cat_map[col]
            if val is not None and str(val) in mapping:
                val = mapping[str(val)]
            else:
                val = 0.0
        if val is None or val == '' or val == 'null':
            val = 0.0
        vals.append(float(val))

    gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose = vals
    eng = {
        'bmi_age': bmi / (age + 1),
        'glucose_hba1c': glucose * hba1c / 100,
        'bmi_obese': 1.0 if bmi >= 30 else 0.0,
        'age_hypertension': age * hypertension / 10,
        'glucose_sq': (glucose / 100) ** 2,
        'hba1c_sq': (hba1c / 10) ** 2,
        'bmi_hba1c': bmi * hba1c / 100,
        'age_glucose': age * glucose / 100,
        'glucose_hba1c_ratio': glucose / (hba1c + 0.001),
        'age_bmi': age * bmi / 100,
    }
    eng_feats = meta.get('engineered_features', [])
    for col in eng_feats:
        vals.append(eng.get(col, 0.0))
    return np.array([vals])

def _load_keras_safe(disease, keras_path, meta):
    try:
        return keras.models.load_model(keras_path)
    except Exception:
        pass
    cfg = _get_model_cfg(meta, disease)
    input_dim = len(meta.get('features', []))
    model = _build_model(input_dim, cfg, disease)
    try:
        with zipfile.ZipFile(keras_path, 'r') as z:
            weights_data = z.read('model.weights.h5')
        tmp = tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False)
        try:
            tmp.write(weights_data)
            tmp.close()
            model.load_weights(tmp.name)
        finally:
            os.unlink(tmp.name)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{disease}' model even via weight fallback: {e}. "
            "Re-train the model using the admin panel."
        )

def _get_model_cfg(meta, disease):
    arch = meta.get('model_architecture')
    if arch:
        return {'layers': arch['layers'], 'dropout': arch.get('dropout', 0.3)}
    return MODEL_CONFIGS.get(disease, {'layers': [64, 32, 16], 'dropout': 0.3})


def _load_model(disease):
    if disease in _cache:
        return _cache[disease]
    scaler_path = _get_asset_path(f'{disease}_scaler.pkl')
    meta_path = _get_asset_path(f'{disease}_meta.json')
    weights_path = _get_asset_path(f'{disease}_model.weights.h5')
    keras_path = _get_asset_path(f'{disease}_model.keras')
    pkl_path = _get_asset_path(f'{disease}_model.pkl')

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    model = None
    model_type = meta.get('model_type', 'keras')

    if model_type == 'pickle':
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle model not found for '{disease}' at {pkl_path}")
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
    else:
        if os.path.exists(weights_path):
            cfg = _get_model_cfg(meta, disease)
            input_dim = len(meta.get('features', []))
            model = _build_model(input_dim, cfg, disease)
            model.load_weights(weights_path)
        elif os.path.exists(keras_path):
            model = _load_keras_safe(disease, keras_path, meta)
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            model_type = 'pickle'
        else:
            raise FileNotFoundError(f"Model not found for '{disease}'. Train the model first.")
    _cache[disease] = {'model': model, 'model_type': model_type, 'scaler': scaler, 'meta': meta}
    return _cache[disease]

def clear_cache(disease=None):
    if disease:
        _cache.pop(disease, None)
    else:
        _cache.clear()

def _classify_risk(confidence):
    if confidence >= 0.80:
        return 'High'
    elif confidence >= 0.40:
        return 'Moderate'
    return 'Low'


def _classify_risk_tuned(prob, threshold):
    if prob >= 0.80:
        return 'High'
    elif prob >= 0.40:
        return 'Moderate'
    return 'Low'

def _encode_heart_input(input_data, meta, cat_map):
    base_feats = meta.get('base_features', meta['features'])
    vals = []
    for col in base_feats:
        val = input_data.get(col)
        if cat_map and col in cat_map:
            mapping = cat_map[col]
            if val is not None and str(val) in mapping:
                val = mapping[str(val)]
            else:
                val = 0.0
        if val is None or val == '' or val == 'null':
            val = 0.0
        vals.append(float(val))

    age, restingbp, cholesterol, maxhr, oldpeak = vals[0], vals[3], vals[4], vals[7], vals[9]
    eng = {
        'age_maxhr': age * maxhr / 100,
        'oldpeak_restingbp': oldpeak * restingbp / 100,
        'oldpeak_maxhr': oldpeak * maxhr / 100,
        'age_oldpeak': age * oldpeak / 10,
        'maxhr_age_ratio': maxhr / (age + 1),
        'oldpeak_binary': 1.0 if oldpeak > 0.5 else 0.0,
        'chol_zero': 1.0 if cholesterol == 0 else 0.0,
        'age_sq': (age / 10) ** 2,
        'maxhr_sq': maxhr ** 2 / 10000,
        'oldpeak_sq': oldpeak ** 2,
        'restingbp_age': restingbp / (age + 1),
        'maxhr_oldpeak': maxhr - oldpeak * 10,
        'age_Chol': age * (cholesterol / 100),
        'age_restingbp': age * restingbp / 100,
        'chol_maxhr': cholesterol * maxhr / 10000,
    }
    eng_feats = meta.get('engineered_features', [])
    for col in eng_feats:
        vals.append(eng.get(col, 0.0))
    return np.array([vals])


def _predict(assets, input_data, disease=None):
    meta = assets['meta']
    cat_map = meta.get('categorical_mappings', {})
    if disease == 'heart':
        X = _encode_heart_input(input_data, meta, cat_map)
    elif disease == 'diabetes':
        X = _encode_diabetes_input(input_data, meta, cat_map)
    else:
        X = _encode_input(input_data, meta, cat_map)
    X_scaled = assets['scaler'].transform(X) if assets['scaler'] else X
    if assets['model_type'] == 'pickle':
        prob = float(assets['model'].predict_proba(X_scaled)[:, 1][0])
    else:
        prob = float(assets['model'].predict(X_scaled, verbose=0)[0][0])
    return prob, None

def _get_threshold(meta, default=0.5):
    return meta.get('best_threshold', default)


def predict_diabetes(input_data):
    assets = _load_model('diabetes')
    prob, _ = _predict(assets, input_data, disease='diabetes')
    th = _get_threshold(assets['meta'])
    return {
        'prediction': int(prob > th),
        'confidence': round(prob, 4),
        'risk_level': _classify_risk_tuned(prob, th),
        'disease': 'diabetes',
    }

def predict_heart(input_data):
    assets = _load_model('heart')
    prob, _ = _predict(assets, input_data, disease='heart')
    th = _get_threshold(assets['meta'])
    return {
        'prediction': int(prob > th),
        'confidence': round(prob, 4),
        'risk_level': _classify_risk_tuned(prob, th),
        'disease': 'heart',
    }

def _add_parkinsons_engineered(features_dict, meta):
    base_features = meta.get('base_features', [])

    def _get(key):
        return float(features_dict.get(key, 0.0))

    result = dict(features_dict)
    fo = _get('MDVP:Fo(Hz)')
    fhi = _get('MDVP:Fhi(Hz)')
    flo = _get('MDVP:Flo(Hz)')
    jit = _get('MDVP:Jitter(%)')
    jit_abs = _get('MDVP:Jitter(Abs)')
    shim = _get('MDVP:Shimmer')
    shim_db = _get('MDVP:Shimmer(dB)')
    nhr = _get('NHR')
    hnr = _get('HNR')
    d2 = _get('D2')
    ppe = _get('PPE')
    dfa = _get('DFA')
    rpde = _get('RPDE')

    result['jitter_total'] = jit * jit_abs * 1000
    result['shimmer_total'] = shim * shim_db
    result['fo_range'] = fhi - flo
    result['fo_range_ratio'] = fhi / (flo + 0.001)
    result['hnr_nhr_ratio'] = hnr / (nhr + 0.001)
    result['jitter_shimmer'] = jit * shim
    result['d2_ppe'] = d2 * ppe
    result['dfa_rpde'] = dfa * rpde
    result['shimmer_jitter_ratio'] = shim / (jit + 0.0001)
    result['fo_jitter'] = fo * jit / 100
    result['nhr_shimmer'] = nhr * shim
    result['dfa_hnr'] = dfa * hnr
    return result


def _calibrate_prob(prob, meta):
    cal = meta.get('calibration')
    if cal:
        coef = cal.get('coef', 1.0)
        intercept = cal.get('intercept', 0.0)
        logit = coef * prob + intercept
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))
    return prob


def predict_parkinsons(features_dict):
    assets = _load_model('parkinsons')
    meta = assets['meta']
    th = _get_threshold(meta)

    complete = _add_parkinsons_engineered(features_dict, meta)

    features = meta['features']
    vals = []
    for col in features:
        vals.append(float(complete.get(col, 0.0)))
    X = np.array([vals])
    X_scaled = assets['scaler'].transform(X) if assets['scaler'] else X
    if assets['model_type'] == 'pickle':
        raw_prob = float(assets['model'].predict_proba(X_scaled)[:, 1][0])
    else:
        raw_prob = float(assets['model'].predict(X_scaled, verbose=0)[0][0])
    prob = _calibrate_prob(raw_prob, meta)
    return {
        'prediction': int(prob > th),
        'confidence': round(prob, 4),
        'raw_probability': round(raw_prob, 4),
        'risk_level': _classify_risk_tuned(prob, th),
        'disease': 'parkinsons',
        'extracted_features': features_dict,
    }
