# backend/main.py
"""
FastAPI Application — Spam Message Classifier API
====================================================
Uygulama giriş noktası. Lifespan yönetimi, CORS middleware,
router montajı ve health check endpoint'ini içerir.

Çalıştırma:
    $ cd spam-message-collector
    $ uvicorn backend.main:app --reload --port 8000

Swagger UI:   http://127.0.0.1:8000/docs
ReDoc:        http://127.0.0.1:8000/redoc
Health Check: http://127.0.0.1:8000/health
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers.classify import router as classify_router
from backend.schemas import HealthResponse
from backend.services.classifier import SpamClassifierService

# ---------------------------------------------------------------------------
# Logging konfigürasyonu
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — Startup/Shutdown yönetimi
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama yaşam döngüsünü yönetir.

    Startup:
        - Konfigürasyonu yükle
        - ML modelini belleğe yükle (warm-up)
        - TextCleaner'ı başlat

    Shutdown:
        - Kaynakları temizle (garbage collection'a bırak)

    Bu pattern, eski @app.on_event("startup") yaklaşımı yerine
    FastAPI'nin modern lifespan context manager'ını kullanır.
    """
    # --- STARTUP ---
    logger.info("=" * 60)
    logger.info("🚀 Spam Classifier API başlatılıyor...")
    logger.info("=" * 60)

    settings = get_settings()
    logger.info("Konfigürasyon yüklendi:")
    logger.info("  App:        %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("  Debug:      %s", settings.DEBUG)
    logger.info("  Model:      %s", settings.MODEL_PATH)
    logger.info("  CORS:       %s", settings.ALLOWED_ORIGINS)

    # Modeli startup'ta yükle → ilk request'te gecikme olmasın
    try:
        service = SpamClassifierService.get_instance()
        service.load_model()
        logger.info("✅ Model başarıyla yüklendi ve hazır!")
    except Exception as e:
        logger.error("❌ Model yüklenirken hata: %s", e, exc_info=True)
        logger.warning("⚠️ API çalışmaya devam edecek, ancak /classify 503 döndürecek.")

    logger.info("=" * 60)
    logger.info("🟢 API hazır — http://127.0.0.1:8000/docs")
    logger.info("=" * 60)

    yield  # Uygulama çalışır

    # --- SHUTDOWN ---
    logger.info("🔴 Spam Classifier API kapatılıyor...")


# ---------------------------------------------------------------------------
# FastAPI Uygulaması
# ---------------------------------------------------------------------------
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "## 🛡️ Spam Mesaj Sınıflandırma API'si\n\n"
        "Makine öğrenmesi tabanlı gerçek zamanlı spam tespit servisi.\n\n"
        "### Özellikler\n"
        "- **Yüksek Doğruluk:** %98+ test doğruluğu ile LinearSVC modeli\n"
        "- **NLP Pipeline:** Otomatik metin ön işleme (lowercase, stopword, lemmatization)\n"
        "- **TF-IDF Vektörizasyon:** Unigram + Bigram özellik çıkarımı\n"
        "- **Toplu Sınıflandırma:** Tek istekte 50'ye kadar mesaj\n"
        "- **Güvenilirlik Skoru:** Her tahmin için olasılık ve güvenilirlik seviyesi\n\n"
        "### Teknoloji\n"
        "- **Model:** LinearSVC (scikit-learn)\n"
        "- **Backend:** FastAPI + Uvicorn\n"
        "- **Veri:** UCI SMS Spam Collection Dataset\n"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router Montajı
# ---------------------------------------------------------------------------
app.include_router(classify_router)


# ---------------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistem"],
    summary="Sağlık Kontrolü",
    description="API ve model durumunu kontrol eder.",
)
async def health_check() -> HealthResponse:
    """API ve modelin çalışır durumda olduğunu doğrular.

    Returns:
        Servis ve model durum bilgileri.
    """
    service = SpamClassifierService.get_instance()

    return HealthResponse(
        status="healthy" if service.is_loaded else "unhealthy",
        model_loaded=service.is_loaded,
        model_name=service.model_name,
        model_accuracy=service.model_accuracy,
    )


# ---------------------------------------------------------------------------
# Root Endpoint — API bilgileri
# ---------------------------------------------------------------------------
@app.get(
    "/",
    tags=["Sistem"],
    summary="API Bilgileri",
    description="API hakkında genel bilgi döndürür.",
)
async def root() -> dict:
    """API karşılama mesajı ve temel bilgileri döndürür."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "classify": "POST /api/v1/classify",
            "classify_batch": "POST /api/v1/classify/batch",
        },
    }
