import os
import json
from flask import Flask
from flask_cors import CORS
from backend.config import Config
from backend.models.db_models import db
from backend.routes.auth import auth_bp
from backend.routes.prediction import prediction_bp
from backend.routes.admin import admin_bp


def create_app(config_class=Config):
    app = Flask(__name__, instance_path=getattr(config_class, 'INSTANCE_PATH', None))
    app.config.from_object(config_class)

    CORS(app, origins=Config.CORS_ORIGINS)

    db.init_app(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(admin_bp)

    os.makedirs(app.config['ML_ASSETS_DIR'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Ensure SQLite database directory exists (prevents instance-path ambiguity)
    _db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    if _db_uri.startswith('sqlite:///'):
        _db_file = _db_uri[len('sqlite:///'):]
        _db_dir = os.path.dirname(_db_file)
        if _db_dir:
            os.makedirs(_db_dir, exist_ok=True)

    with app.app_context():
        db.create_all()

    @app.route('/')
    def home():
        return json.dumps({
            'status': 'running',
            'api': 'AI Medical Assistant Backend',
            'docs': 'http://127.0.0.1:5000/docs',
        }), 200, {'Content-Type': 'application/json'}

    @app.route('/docs')
    def docs():
        return json.dumps({
            'api': 'AI Medical Assistant Backend',
            'base_url': 'http://127.0.0.1:5000',
            'frontend': 'http://localhost:5173',
            'endpoints': {
                'POST /api/auth/register': 'Register new user',
                'POST /api/auth/login': 'Login, returns JWT token',
                'GET /api/auth/profile': 'Get user profile (auth required)',
                'POST /api/predict/diabetes': 'Predict diabetes risk',
                'POST /api/predict/heart': 'Predict heart disease risk',
                'POST /api/predict/parkinsons': 'Predict Parkinson\'s from audio',
                'GET /api/predictions/history': 'Get prediction history',
                'GET /api/predictions/<id>/pdf': 'Download prediction PDF',
                'GET /api/admin/users': 'List users (admin)',
                'GET /api/admin/predictions': 'List all predictions (admin)',
                'POST /api/admin/retrain/<disease>': 'Retrain model (admin)',
            },
            'note': 'Open http://localhost:5173 in your browser for the web UI'
        }), 200, {'Content-Type': 'application/json'}

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
