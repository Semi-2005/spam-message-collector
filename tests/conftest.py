"""Pytest Configuration and Shared Fixtures.
=============================================
Tüm testlerde paylaşılan fixture'lar, mock modeller ve
FastAPI TestClient yapılandırması.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.main import app
from backend.services.classifier import SpamClassifierService
from preprocessing.text_cleaner import TextCleaner


@pytest.fixture(scope="session")
def trained_dummy_pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Testler için hafif (lightweight) bir model pipeline'ı ve metadata dosyası üretir.

    Gerçek model dosyası diskte olmasa dahi CI ortamında testlerin %100 izole
    ve stabil çalışmasını garanti eder.
    """
    tmp_dir = tmp_path_factory.mktemp("test_models")
    model_path = tmp_dir / "test_pipeline.joblib"
    metadata_path = tmp_dir / "test_metadata.json"

    # Basit bir sentetik veri kümesi
    texts = [
        "win free cash now click here claim urgent prize",
        "exclusive offer free entry call now to win reward",
        "congratulations you won a lottery award claim immediately",
        "hello how are you doing today see you later",
        "can we meet for coffee tomorrow afternoon",
        "hey are we still going to the library",
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1: spam, 0: ham

    cleaner = TextCleaner()
    cleaned_texts = [cleaner.clean_text(t) for t in texts]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(random_state=42, C=1.0)),
        ]
    )
    pipeline.fit(cleaned_texts, labels)

    # Modeli ve metadata'yı diske kaydet
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "Test LinearSVC",
        "performance": {
            "test_accuracy": 0.99,
            "test_f1": 0.99,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return model_path, metadata_path


@pytest.fixture(autouse=True)
def init_test_classifier_service(trained_dummy_pipeline: tuple[Path, Path]):
    """Her testten önce SpamClassifierService singleton'ını test modeli ile başlatır."""
    model_path, metadata_path = trained_dummy_pipeline

    # Singleton'ı sıfırla/yeniden yapılandır
    SpamClassifierService._instance = None
    service = SpamClassifierService.get_instance(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    service.load_model()
    return service


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient fixture'ı."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_spam_text() -> str:
    """Tipik spam mesajı örneği."""
    return (
        "URGENT! You have won a 1000 cash prize. Call 09061701461 to claim your reward immediately."
    )


@pytest.fixture
def sample_ham_text() -> str:
    """Tipik güvenli (ham) mesaj örneği."""
    return "Hey, are you free tomorrow afternoon for lunch? Let me know."
