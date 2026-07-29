# backend/config.py
"""
Application Configuration
==========================
Pydantic-settings tabanlı merkezi konfigürasyon modülü.

Tüm ayarlar ortam değişkenlerinden veya .env dosyasından okunabilir.
Varsayılan değerler development ortamı için optimize edilmiştir.

Kullanım:
    >>> from backend.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.MODEL_PATH)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Proje kök dizini — config.py'nin iki üst dizini
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Uygulama ayarları.

    Tüm alanlar ortam değişkenleri veya .env dosyası üzerinden
    override edilebilir. Prefix: ``SPAM_`` (opsiyonel).

    Attributes:
        APP_NAME: API'nin görünen adı (Swagger UI başlığı).
        APP_VERSION: Semantik versiyon numarası.
        DEBUG: Debug modu aktif mi (development için True).
        MODEL_PATH: Eğitilmiş pipeline'ın (.joblib) dosya yolu.
        ALLOWED_ORIGINS: CORS için izin verilen origin listesi.
        MAX_TEXT_LENGTH: Kabul edilen maksimum metin uzunluğu (karakter).
        MIN_TEXT_LENGTH: Kabul edilen minimum metin uzunluğu (karakter).
        SPAM_THRESHOLD: Spam sınıflandırma eşik değeri (0-1 arası).
    """

    APP_NAME: str = "Spam Message Classifier API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Model yolu — proje kökünden göreli
    MODEL_PATH: Path = _PROJECT_ROOT / "models" / "artifacts" / "best_model_pipeline.joblib"
    MODEL_METADATA_PATH: Path = _PROJECT_ROOT / "models" / "artifacts" / "model_metadata.json"

    # CORS ayarları
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Input validasyon sınırları
    MAX_TEXT_LENGTH: int = 5000
    MIN_TEXT_LENGTH: int = 1

    # Sınıflandırma eşiği
    SPAM_THRESHOLD: float = 0.5

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton Settings instance'ı döndürür.

    lru_cache ile uygulama boyunca tek bir Settings nesnesi oluşturulur.
    Bu, her request'te .env dosyasının tekrar okunmasını önler.

    Returns:
        Yapılandırılmış Settings nesnesi.
    """
    return Settings()
