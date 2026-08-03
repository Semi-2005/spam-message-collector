"""Integration Tests for FastAPI Endpoints.
=========================================
HTTP endpoint'lerinin, status kodlarının ve Pydantic validasyon
kurallarının doğruluğunu test eder.
"""

from fastapi.testclient import TestClient


class TestSystemEndpoints:
    """Root ve Health check endpoint testleri."""

    def test_root_endpoint(self, client: TestClient):
        """GET / endpoint'inin 200 ve API bilgilerini döndürdüğünü doğrular."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "app" in data
        assert "version" in data
        assert "docs" in data
        assert data["health"] == "/health"

    def test_health_check_endpoint(self, client: TestClient):
        """GET /health endpoint'inin servis durumunu döndürdüğünü doğrular."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_name" in data


class TestClassificationEndpoints:
    """/api/v1/classify ve /api/v1/classify/batch endpoint testleri."""

    def test_classify_valid_spam(self, client: TestClient, sample_spam_text: str):
        """POST /api/v1/classify ile spam metin sınıflandırma."""
        payload = {"text": sample_spam_text}
        response = client.post("/api/v1/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == sample_spam_text
        assert data["is_spam"] is True
        assert data["label"] == "spam"
        assert "spam_probability" in data
        assert "confidence_level" in data

    def test_classify_valid_ham(self, client: TestClient, sample_ham_text: str):
        """POST /api/v1/classify ile ham metin sınıflandırma."""
        payload = {"text": sample_ham_text}
        response = client.post("/api/v1/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_spam"] is False
        assert data["label"] == "ham"

    def test_classify_validation_error_empty_body(self, client: TestClient):
        """Eksik veya boş payload gönderildiğinde 422 dönmeli."""
        response = client.post("/api/v1/classify", json={})
        assert response.status_code == 422

    def test_classify_batch_success(
        self, client: TestClient, sample_spam_text: str, sample_ham_text: str
    ):
        """POST /api/v1/classify/batch ile toplu sınıflandırma."""
        payload = {
            "messages": [
                sample_spam_text,
                sample_ham_text,
                "Just checking in, see you at 5pm.",
            ]
        }
        response = client.post("/api/v1/classify/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["spam_count"] >= 1
        assert data["ham_count"] >= 1
        assert len(data["results"]) == 3

    def test_classify_batch_validation_max_limit(self, client: TestClient):
        """50'den fazla mesaj gönderildiğinde 422 validasyon hatası vermeli."""
        too_many_messages = [f"Message {i}" for i in range(51)]
        payload = {"messages": too_many_messages}
        response = client.post("/api/v1/classify/batch", json=payload)
        assert response.status_code == 422
