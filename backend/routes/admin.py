"""
Admin routes — user management, prediction overview, model retraining, dataset upload.
"""
import os
import shutil
import json

from flask import Blueprint, request, jsonify, current_app

from backend.models.db_models import db, User, PredictionHistory
from backend.routes.auth import admin_required
from backend.ml.predict import clear_cache

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')


@admin_bp.route('/users', methods=['GET'])
@admin_required
def list_users(current_user):
    """List all registered users."""
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({'users': [u.to_dict() for u in users]}), 200


@admin_bp.route('/predictions', methods=['GET'])
@admin_required
def list_predictions(current_user):
    """List all predictions from all users."""
    records = (
        PredictionHistory.query
        .order_by(PredictionHistory.created_at.desc())
        .limit(500)
        .all()
    )
    results = []
    for r in records:
        d = r.to_dict()
        user = User.query.get(r.user_id)
        d['username'] = user.username if user else 'Unknown'
        results.append(d)
    return jsonify({'predictions': results}), 200


@admin_bp.route('/retrain/<disease>', methods=['POST'])
@admin_required
def retrain_model(current_user, disease):
    """Retrain a specific disease model."""
    valid = ['diabetes', 'heart', 'parkinsons']
    if disease not in valid:
        return jsonify({'error': f'Invalid disease. Choose from: {valid}'}), 400

    try:
        # Import training functions
        from backend.ml.train import train_diabetes, train_heart, train_parkinsons

        trainers = {
            'diabetes': train_diabetes,
            'heart': train_heart,
            'parkinsons': train_parkinsons,
        }

        accuracy = trainers[disease]()
        clear_cache(disease)  # Clear cached model so next prediction loads the new one

        # Read training plot as base64 for frontend display
        import base64
        plot_path = os.path.join(
            current_app.config['ML_ASSETS_DIR'],
            f'{disease}_training_plot.png'
        )
        plot_b64 = None
        if os.path.exists(plot_path):
            with open(plot_path, 'rb') as f:
                plot_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            'message': f'{disease.capitalize()} model retrained successfully',
            'accuracy': round(accuracy, 4),
            'training_plot': plot_b64,
        }), 200

    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500


@admin_bp.route('/upload-dataset/<disease>', methods=['POST'])
@admin_required
def upload_dataset(current_user, disease):
    """Upload a new CSV dataset for a disease."""
    valid_map = {
        'diabetes': 'diabetes.csv',
        'heart': 'heart.csv',
        'parkinsons': 'Parkinsons disease.csv',
    }
    if disease not in valid_map:
        return jsonify({'error': f'Invalid disease type'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are accepted'}), 400

    datasets_dir = current_app.config['DATASETS_DIR']
    target = os.path.join(datasets_dir, valid_map[disease])

    # Backup existing
    if os.path.exists(target):
        backup = target + '.backup'
        shutil.copy2(target, backup)

    file.save(target)

    return jsonify({
        'message': f'Dataset uploaded for {disease}. Use the retrain endpoint to update the model.',
        'filename': valid_map[disease],
    }), 200


@admin_bp.route('/model-info/<disease>', methods=['GET'])
@admin_required
def model_info(current_user, disease):
    """Get model metadata and training plot for a disease."""
    valid = ['diabetes', 'heart', 'parkinsons']
    if disease not in valid:
        return jsonify({'error': 'Invalid disease'}), 400

    import base64
    ml_dir = current_app.config['ML_ASSETS_DIR']

    # Meta
    meta_path = os.path.join(ml_dir, f'{disease}_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    # Training plot
    plot_path = os.path.join(ml_dir, f'{disease}_training_plot.png')
    plot_b64 = None
    if os.path.exists(plot_path):
        with open(plot_path, 'rb') as f:
            plot_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Heatmap
    hm_path = os.path.join(ml_dir, f'{disease}_heatmap.png')
    hm_b64 = None
    if os.path.exists(hm_path):
        with open(hm_path, 'rb') as f:
            hm_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Check if model exists (prefer .weights.h5 over .keras)
    model_exists = (
        os.path.exists(os.path.join(ml_dir, f'{disease}_model.weights.h5')) or
        os.path.exists(os.path.join(ml_dir, f'{disease}_model.keras'))
    )

    return jsonify({
        'disease': disease,
        'model_exists': model_exists,
        'meta': meta,
        'training_plot': plot_b64,
        'heatmap': hm_b64,
    }), 200
