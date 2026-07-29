# backend/routers/classify.py
"""
Classification Router — /api/v1/classify Endpoint
====================================================
Spam sınıflandırma API endpoint'lerini tanımlar.

Endpoint'ler:
    POST /api/v1/classify       — Tek bir mesajı sınıflandırır
    POST /api/v1/classify/batch — Birden fazla mesajı toplu sınıflandırır
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from backend.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from backend.services.classifier import SpamClassifierService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Sınıflandırma"],
    responses={
        500: {"description": "Model yükleme hatası veya dahili sunucu hatası"},
    },
)


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Tek Mesaj Sınıflandırma",
    description=(
        "Verilen metin mesajını **spam** veya **ham** (güvenli) olarak sınıflandırır.\n\n"
        "Model, TF-IDF vektörizasyon ve Linear SVC algoritması kullanarak "
        "yüksek doğrulukla (%98+) sınıflandırma yapar.\n\n"
        "**İşlem akışı:**\n"
        "1. Metin NLP pipeline ile ön işlenir (lowercase, stopword, lemmatization)\n"
        "2. TF-IDF ile vektörleştirilir\n"
        "3. Eğitilmiş model ile sınıflandırılır\n"
        "4. Olasılık ve güvenilirlik seviyesi hesaplanır"
    ),
)
async def classify_message(request: ClassifyRequest) -> ClassifyResponse:
    """Tek bir metin mesajını spam/ham olarak sınıflandırır.

    Args:
        request: Sınıflandırılacak metni içeren istek.

    Returns:
        Sınıflandırma sonucu (label, olasılık, güvenilirlik).

    Raises:
        HTTPException 503: Model henüz yüklenmemişse.
        HTTPException 500: Sınıflandırma sırasında hata oluşursa.
    """
    t_start = time.perf_counter()

    try:
        service = SpamClassifierService.get_instance()

        if not service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Model henüz yüklenmedi.",
                    "message": "Servis başlatılıyor, lütfen birkaç saniye sonra tekrar deneyin.",
                },
            )

        result = service.classify(request.text)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Sınıflandırma tamamlandı — label=%s, prob=%.4f, süre=%.3fms",
            result["label"],
            result["spam_probability"],
            elapsed * 1000,
        )

        return ClassifyResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Sınıflandırma hatası: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Sınıflandırma sırasında bir hata oluştu.",
                "message": str(e),
            },
        )


@router.post(
    "/classify/batch",
    response_model=BatchClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Toplu Mesaj Sınıflandırma",
    description=(
        "Birden fazla metin mesajını tek bir istekte toplu olarak sınıflandırır.\n\n"
        "Maksimum 50 mesaj gönderilebilir."
    ),
)
async def classify_batch(request: BatchClassifyRequest) -> BatchClassifyResponse:
    """Birden fazla mesajı toplu olarak sınıflandırır.

    Args:
        request: Sınıflandırılacak metin listesini içeren istek.

    Returns:
        Her mesaj için ayrı sonuç ve istatistikler.

    Raises:
        HTTPException 503: Model henüz yüklenmemişse.
        HTTPException 500: Sınıflandırma sırasında hata oluşursa.
    """
    t_start = time.perf_counter()

    try:
        service = SpamClassifierService.get_instance()

        if not service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Model henüz yüklenmedi.",
                    "message": "Servis başlatılıyor, lütfen birkaç saniye sonra tekrar deneyin.",
                },
            )

        results = service.classify_batch(request.messages)
        responses = [ClassifyResponse(**r) for r in results]

        spam_count = sum(1 for r in responses if r.is_spam)
        ham_count = len(responses) - spam_count

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Toplu sınıflandırma tamamlandı — %d mesaj, %d spam, %d ham, süre=%.3fms",
            len(responses),
            spam_count,
            ham_count,
            elapsed * 1000,
        )

        return BatchClassifyResponse(
            results=responses,
            total=len(responses),
            spam_count=spam_count,
            ham_count=ham_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Toplu sınıflandırma hatası: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Toplu sınıflandırma sırasında bir hata oluştu.",
                "message": str(e),
            },
        )
