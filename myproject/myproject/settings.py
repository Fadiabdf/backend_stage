"""
Django settings for myproject project.

Generated by 'django-admin startproject' using Django 4.1.13.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.1/ref/settings/
"""

from pathlib import Path
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-5&^$i^u0&k5an42=(*#(i(0a*mop^1o^#p^+a=q&0d&@j(h9zw'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myproject',
    'validation',
    'rest_framework',
    'corsheaders',
]
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ),
}
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

import mongoengine
mongoengine.connect(db="projet_1cs", host="mongodb+srv://maroua:maroua2003@cluster0.p6t0qwx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0", username="maroua", password="maroua2003")
# import mongoengine

# # MongoDB connection settings
# MONGODB_NAME = 'projet_1cs'
 
 
# MONGODB_USER = 'maroua'
# MONGODB_PASS = 'MAROUA2003'

# mongo_uri = f"mongodb+srv://maroua:maroua2003@cluster0.p6t0qwx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongoengine.connect(
#     db=MONGODB_NAME,
#     username=MONGODB_USER,
#     password=MONGODB_PASS,
    
# )
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'myproject.wsgi.application'
DATA_UPLOAD_MAX_MEMORY_SIZE = 10242880  # 5 MB (adjust the size as needed)

CORS_ALLOWED_ORIGINS = [
    "http://localhost:8081",  # Add your frontend URL
    "http://192.168.1.23:8080",
    "http://localhost:8080"
    
]
# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases


CORS_ALLOW_CREDENTIALS = True
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8081',
    "http://192.168.1.23:8080/",
    "http://localhost:8080"
]

# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
