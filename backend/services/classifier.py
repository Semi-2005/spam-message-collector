# backend/services/classifier.py
"""
Spam Classifier Service — Singleton Model Yükleme ve Inference
================================================================
Eğitilmiş ML pipeline'ını yükleyip, metin sınıflandırma işlemlerini
gerçekleştiren servis katmanı.

Tasarım kararları:
    - Singleton pattern: Model bellekte tek bir kez yüklenir, tüm
      request'ler aynı instance'ı paylaşır → hafıza verimli.
    - Lazy loading: Model ilk istek gelene kadar yüklenmez
      (lifespan ile startup'ta da tetiklenebilir).
    - TextCleaner entegrasyonu: Ham metin → preprocessing → pipeline.predict()
      zincirleme akış tek metotta kapsüllenir.
    - LinearSVC decision_function → sigmoid dönüşümü: SVC doğrudan
      olasılık üretmediğinden, decision function değerini Platt scaling
      benzeri sigmoid ile [0,1] aralığına normalize ediyoruz.

Kullanım:
    >>> from backend.services.classifier import SpamClassifierService
    >>> service = SpamClassifierService.get_instance()
    >>> result = service.classify("Win a free iPhone NOW!!!")
    >>> print(result)
    {'label': 'spam', 'is_spam': True, 'spam_probability': 0.97, ...}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from preprocessing.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class SpamClassifierService:
    """Spam sınıflandırma servisi — Singleton pattern ile model yönetimi.

    Bu sınıf:
        1. Eğitilmiş sklearn Pipeline'ı (.joblib) diskten yükler
        2. TextCleaner ile gelen metni ön işler
        3. Pipeline üzerinden tahmin yapar
        4. Decision function → olasılık dönüşümü uygular
        5. Yapılandırılmış sonuç döndürür

    Thread-safe Singleton: Çoklu worker/thread ortamında bile tek
    bir model instance'ı garanti edilir.

    Attributes:
        _instance: Singleton instance referansı.
        _lock: Thread-safety için kilit.
        _pipeline: Yüklü sklearn Pipeline nesnesi.
        _cleaner: TextCleaner instance'ı.
        _metadata: Model metadata bilgileri.
        _is_loaded: Modelin yüklenip yüklenmediğini belirtir.
    """

    _instance: SpamClassifierService | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> SpamClassifierService:
        """Thread-safe Singleton implementasyonu."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        """Servis konfigürasyonunu ayarlar (model henüz yüklenmez).

        Args:
            model_path: .joblib pipeline dosyasının yolu.
            metadata_path: model_metadata.json dosyasının yolu.
        """
        # Singleton: Zaten initialize edildiyse tekrar yapma
        if self._initialized:
            return

        from backend.config import get_settings

        settings = get_settings()

        self._model_path = Path(model_path) if model_path else settings.MODEL_PATH
        self._metadata_path = (
            Path(metadata_path) if metadata_path else settings.MODEL_METADATA_PATH
        )
        self._spam_threshold = settings.SPAM_THRESHOLD

        self._pipeline: Any | None = None
        self._cleaner: TextCleaner | None = None
        self._metadata: dict[str, Any] = {}
        self._is_loaded: bool = False
        self._load_time: float = 0.0

        self._initialized = True
        logger.info("SpamClassifierService yapılandırıldı (model henüz yüklenmedi).")

    # ------------------------------------------------------------------
    # Singleton Erişim
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(
        cls,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> SpamClassifierService:
        """Singleton instance'ı döndürür, gerekirse oluşturur.

        Args:
            model_path: İlk oluşturmada kullanılacak model yolu.
            metadata_path: İlk oluşturmada kullanılacak metadata yolu.

        Returns:
            SpamClassifierService singleton instance'ı.
        """
        return cls(model_path=model_path, metadata_path=metadata_path)

    # ------------------------------------------------------------------
    # Model Yükleme
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Pipeline ve metadata'yı diskten yükler.

        Bu metot idempotent'tir — model zaten yüklüyse tekrar yüklemez.
        Lifespan startup sırasında veya ilk request'te çağrılabilir.

        Raises:
            FileNotFoundError: Model dosyası bulunamazsa.
            RuntimeError: Model yüklenirken hata oluşursa.
        """
        if self._is_loaded:
            logger.info("Model zaten yüklü, tekrar yükleme atlanıyor.")
            return

        t_start = time.perf_counter()

        # Pipeline yükle
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {self._model_path}\n"
                f"Lütfen önce model eğitim pipeline'ını çalıştırın:\n"
                f"  python -m models.train_model"
            )

        logger.info("Model yükleniyor: %s", self._model_path)
        self._pipeline = joblib.load(self._model_path)

        # Metadata yükle (opsiyonel — yoksa devam et)
        if self._metadata_path.exists():
            with open(self._metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                "Metadata yüklendi — Model: %s, Test F1: %.4f",
                self._metadata.get("model_name", "Bilinmiyor"),
                self._metadata.get("performance", {}).get("test_f1", 0.0),
            )
        else:
            logger.warning("Metadata dosyası bulunamadı: %s", self._metadata_path)

        # TextCleaner oluştur
        self._cleaner = TextCleaner(remove_numbers=True, min_token_length=2)

        self._is_loaded = True
        self._load_time = time.perf_counter() - t_start

        logger.info("Model başarıyla yüklendi! (%.3f saniye)", self._load_time)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def classify(self, text: str) -> dict[str, Any]:
        """Tek bir metin mesajını sınıflandırır.

        İşlem akışı:
            1. TextCleaner ile metin ön işleme
            2. Pipeline.predict() ile sınıflandırma
            3. Decision function → sigmoid ile olasılık hesaplama
            4. Güvenilirlik seviyesi belirleme
            5. Yapılandırılmış sonuç döndürme

        Args:
            text: Sınıflandırılacak ham metin.

        Returns:
            Sınıflandırma sonuçlarını içeren sözlük:
                - text: Orijinal metin
                - label: "spam" veya "ham"
                - is_spam: bool
                - spam_probability: float (0-1)
                - confidence_level: "Düşük", "Orta" veya "Yüksek"

        Raises:
            RuntimeError: Model yüklenmemişse.
        """
        if not self._is_loaded:
            self.load_model()

        # Adım 1: Metin ön işleme
        cleaned_text = self._cleaner.clean_text(text)

        # Guard: Temizleme sonrası boş metin
        if not cleaned_text.strip():
            return {
                "text": text,
                "label": "ham",
                "is_spam": False,
                "spam_probability": 0.0,
                "confidence_level": "Düşük",
            }

        # Adım 2: Tahmin
        prediction = self._pipeline.predict([cleaned_text])[0]

        # Adım 3: Olasılık hesaplama
        spam_probability = self._compute_probability(cleaned_text)

        # Adım 4: Eşik değerine göre nihai karar
        is_spam = spam_probability >= self._spam_threshold
        label = "spam" if is_spam else "ham"

        # Adım 5: Güvenilirlik seviyesi
        confidence_level = self._get_confidence_level(spam_probability)

        return {
            "text": text,
            "label": label,
            "is_spam": is_spam,
            "spam_probability": round(spam_probability, 4),
            "confidence_level": confidence_level,
        }

    def classify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Birden fazla mesajı toplu olarak sınıflandırır.

        Args:
            texts: Sınıflandırılacak metin listesi.

        Returns:
            Her metin için sınıflandırma sonuçlarını içeren liste.
        """
        return [self.classify(text) for text in texts]

    # ------------------------------------------------------------------
    # Olasılık Hesaplama (LinearSVC → Sigmoid dönüşümü)
    # ------------------------------------------------------------------
    def _compute_probability(self, cleaned_text: str) -> float:
        """Decision function değerini [0,1] aralığında olasılığa dönüştürür.

        LinearSVC, predict_proba() desteklemediğinden, decision_function()
        çıktısını sigmoid fonksiyonu ile olasılığa çeviriyoruz.

        Sigmoid: σ(x) = 1 / (1 + e^(-x))
            - x > 0 → spam tarafında (olasılık > 0.5)
            - x < 0 → ham tarafında (olasılık < 0.5)
            - |x| büyüdükçe → güvenilirlik artar

        Args:
            cleaned_text: Ön işlemden geçmiş temiz metin.

        Returns:
            [0.0, 1.0] aralığında spam olasılığı.
        """
        try:
            # decision_function: pozitif = spam (1), negatif = ham (0)
            decision_value = self._pipeline.decision_function([cleaned_text])[0]
            # Sigmoid dönüşümü
            probability = float(1.0 / (1.0 + np.exp(-decision_value)))
        except AttributeError:
            # Eğer model predict_proba destekliyorsa (NB, LR gibi)
            try:
                proba = self._pipeline.predict_proba([cleaned_text])[0]
                probability = float(proba[1])  # Spam sınıfının olasılığı
            except AttributeError:
                # Son çare: sadece predict sonucunu kullan
                prediction = self._pipeline.predict([cleaned_text])[0]
                probability = 1.0 if prediction == 1 else 0.0

        return probability

    @staticmethod
    def _get_confidence_level(probability: float) -> str:
        """Olasılık değerine göre güvenilirlik seviyesi belirler.

        Seviye eşikleri:
            - |p - 0.5| > 0.3 → "Yüksek"   (p < 0.2 veya p > 0.8)
            - |p - 0.5| > 0.15 → "Orta"     (p < 0.35 veya p > 0.65)
            - diğer             → "Düşük"    (0.35 ≤ p ≤ 0.65)

        Args:
            probability: Spam olasılık değeri (0-1).

        Returns:
            Güvenilirlik seviyesi string'i.
        """
        distance_from_center = abs(probability - 0.5)

        if distance_from_center > 0.3:
            return "Yüksek"
        elif distance_from_center > 0.15:
            return "Orta"
        else:
            return "Düşük"

    # ------------------------------------------------------------------
    # Durum Bilgileri
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """Model yüklü mü?"""
        return self._is_loaded

    @property
    def model_name(self) -> str | None:
        """Yüklü modelin adı."""
        return self._metadata.get("model_name")

    @property
    def model_accuracy(self) -> float | None:
        """Modelin test doğruluğu."""
        return self._metadata.get("performance", {}).get("test_accuracy")

    @property
    def metadata(self) -> dict:
        """Model metadata sözlüğü."""
        return self._metadata
