"""
Application configuration loaded from environment variables.
"""
import os
from dotenv import load_dotenv

# Load .env file from the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BACKEND_DIR, '.env'))

ON_VERCEL = os.environ.get('VERCEL', '') == '1'
ON_HUGGINGFACE = os.environ.get('SPACE_ID', '') != ''


class Config:
    """Flask application configuration."""

    # Flask core
    SECRET_KEY = os.getenv(
        'SECRET_KEY',
        os.urandom(32).hex() if ON_VERCEL else 'medical-ai-secret-key-2024'
    )

    # Use /tmp for instance folder on cloud platforms (permissions + ephemeral)
    if ON_VERCEL or ON_HUGGINGFACE:
        INSTANCE_PATH = '/tmp/instance'

    # Database — use absolute path to avoid Flask instance path ambiguity.
    # On cloud platforms use /tmp/ (ephemeral); locally use backend/instance/.
    _local_db = os.path.join(BACKEND_DIR, 'instance', 'medical_ai.db')
    _default_db = (
        'sqlite:////tmp/medical_ai.db'
        if (ON_VERCEL or ON_HUGGINGFACE)
        else f'sqlite:///{_local_db}'
    )
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', _default_db)
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

    # File uploads — /tmp on cloud platforms (permissions + ephemeral)
    _default_upload = '/tmp/uploads' if (ON_VERCEL or ON_HUGGINGFACE) else 'uploads'
    UPLOAD_FOLDER = os.path.join(
        BACKEND_DIR, os.getenv('UPLOAD_FOLDER', _default_upload)
    )
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB

    # ML assets
    ML_ASSETS_DIR = os.path.join(
        BACKEND_DIR, os.getenv('ML_ASSETS_DIR', 'static/ml_assets')
    )

    # Datasets
    DATASETS_DIR = os.path.join(BACKEND_DIR, '..', 'datasets')

    # CORS — permissive on Vercel or Hugging Face,
    # locked in development
    CORS_ORIGINS = (
        ['*']
        if ON_VERCEL or ON_HUGGINGFACE
        else [
            'http://localhost:5173',
            'http://127.0.0.1:5173',
            'https://ai-medical-diagnosis-assistant.vercel.app',
        ]
    )
