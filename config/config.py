from __future__ import annotations

"""Configuration settings for the Civic Complaint System.

This module loads configuration from environment variables and provides
default values for development.
"""

import os
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Database Configuration
DATABASE_PATH = os.getenv('CIVIC_DB_PATH', str(PROJECT_ROOT / 'data' / 'civic_system.db'))
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')

# Use Supabase if configured, otherwise use local SQLite
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_ANON_KEY)

# Email Configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SMTP_FROM = os.getenv('SMTP_FROM', 'noreply@civiccomplaints.gov')

# SMS Configuration (Twilio)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')

# Application Settings
SECRET_KEY = os.getenv('SECRET_KEY', 'civic-complaint-secret-key-change-in-production')
SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', '30'))
MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', '3'))

# Default Admin Credentials
DEFAULT_ADMIN_EMAIL = os.getenv('DEFAULT_ADMIN_EMAIL', 'admin@civiccomplaints.gov')
DEFAULT_ADMIN_PASSWORD = os.getenv('DEFAULT_ADMIN_PASSWORD', 'Admin@123')

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Model Paths
MODELS_DIR = PROJECT_ROOT / 'models'
MURIL_MODEL_PATH = MODELS_DIR / 'muril_category_classifier'
XGBOOST_MODEL_PATH = MODELS_DIR / 'xgboost_urgency_predictor.pkl'
SCALER_PATH = MODELS_DIR / 'feature_scaler.pkl'

# Performance Targets
TARGET_CATEGORY_ACCURACY = 0.94
TARGET_URGENCY_ACCURACY = 0.89
TARGET_PROCESSING_TIME_SECONDS = 4.0
TARGET_SHAP_TIME_SECONDS = 2.0

# Complaint Categories
CATEGORIES = ['Sanitation', 'Water Supply', 'Transportation']

# Urgency Levels
URGENCY_LEVELS = ['Critical', 'High', 'Medium', 'Low']

# Supported Languages
LANGUAGES = ['English', 'Hindi', 'Hinglish']

# User Roles
ROLES = ['citizen', 'official', 'admin']

# Complaint Status Options
COMPLAINT_STATUSES = ['Registered', 'Under Review', 'In Progress', 'Work Scheduled', 'Resolved', 'Rejected']

__all__ = [
    'PROJECT_ROOT',
    'DATABASE_PATH',
    'USE_SUPABASE',
    'SUPABASE_URL',
    'SUPABASE_ANON_KEY',
    'SUPABASE_SERVICE_ROLE_KEY',
    'SECRET_KEY',
    'SESSION_TIMEOUT_MINUTES',
    'MAX_LOGIN_ATTEMPTS',
    'CATEGORIES',
    'URGENCY_LEVELS',
    'LANGUAGES',
    'ROLES',
    'COMPLAINT_STATUSES',
]
