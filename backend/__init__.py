# backend/__init__.py
"""
FastAPI Backend — Spam Message Classifier API
===============================================
Eğitilmiş ML modelini kullanarak gerçek zamanlı spam sınıflandırma
hizmeti sunan RESTful API uygulaması.

Mimari katmanlar:
    - config:   Ortam konfigürasyonu (pydantic-settings)
    - schemas:  Request/Response validasyon modelleri
    - services: İş mantığı (model yükleme, inference)
    - routers:  HTTP endpoint tanımları
"""

__version__ = "1.0.0"
