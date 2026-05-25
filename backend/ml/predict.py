import os, json, pickle, functools
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERLAS'] = '0'
from tensorflow import keras
from tensorflow.keras import layers


def _patch_keras():
    _patched = {}
    for name in ['Dense', 'BatchNormalization', 'Dropout', 'InputLayer']:
        cls = getattr(layers, name)
        orig = cls.__init__

        @functools.wraps(orig)
        def new_init(self, *args, __orig=orig, **kwargs):
            kwargs.pop('quantization_config', None)
            __orig(self, *args, **kwargs)

        cls.__init__ = new_init
        _patched[name] = (cls, orig)
    return _patched


def _unpatch_keras(patched):
    for _, (cls, orig) in patched.items():
        cls.__init__ = orig

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
    model_type = 'keras'

    if os.path.exists(weights_path):
        cfg = MODEL_CONFIGS.get(disease, {'layers': [64, 32, 16], 'dropout': 0.3})
        input_dim = len(meta.get('features', []))
        model = _build_model(input_dim, cfg, disease)
        model.load_weights(weights_path)
        model_type = 'keras'
    elif os.path.exists(keras_path):
        try:
            model = keras.models.load_model(keras_path)
        except (TypeError, ValueError) as _e:
            if 'quantization_config' not in str(_e):
                raise
            _patched = _patch_keras()
            try:
                model = keras.models.load_model(keras_path)
            finally:
                _unpatch_keras(_patched)
        model_type = 'keras'
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
    if confidence >= 0.75:
        return 'High'
    elif confidence >= 0.45:
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
        'restingbp_chol': restingbp * cholesterol / 10000,
        'maxhr_sq': maxhr ** 2 / 100,
        'oldpeak_sq': oldpeak ** 2,
        'chol_zero': 1.0 if cholesterol == 0 else 0.0,
    }
    for col in meta.get('engineered_features', []):
        if col not in eng:
            eng[col] = 0.0
    eng_feats = meta.get('engineered_features', [])
    for col in eng_feats:
        vals.append(eng.get(col, 0.0))
    return np.array([vals])


def _predict(assets, input_data, disease=None):
    meta = assets['meta']
    cat_map = meta.get('categorical_mappings', {})
    if disease == 'heart':
        X = _encode_heart_input(input_data, meta, cat_map)
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
    prob, _ = _predict(assets, input_data)
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

def predict_parkinsons(features_dict):
    assets = _load_model('parkinsons')
    meta = assets['meta']
    th = _get_threshold(meta)
    features = meta['features']
    vals = []
    for col in features:
        vals.append(float(features_dict.get(col, 0.0)))
    X = np.array([vals])
    X_scaled = assets['scaler'].transform(X) if assets['scaler'] else X
    if assets['model_type'] == 'pickle':
        prob = float(assets['model'].predict_proba(X_scaled)[:, 1][0])
    else:
        prob = float(assets['model'].predict(X_scaled, verbose=0)[0][0])
    confidence = round(2.0 * abs(prob - th), 4)
    risk_level = _classify_risk_tuned(prob, th)
    return {
        'prediction': int(prob > th),
        'confidence': confidence,
        'raw_probability': round(prob, 4),
        'risk_level': risk_level,
        'disease': 'parkinsons',
        'extracted_features': features_dict,
    }
