"""Unit Tests for SpamClassifierService.
========================================
Model yükleme, inference, singleton davranışı ve olasılık
hesaplama mekanizmalarını test eder.
"""

from backend.services.classifier import SpamClassifierService


class TestSpamClassifierService:
    """SpamClassifierService test suite."""

    def test_singleton_behavior(self):
        """Service'in Singleton olarak tek bir instance sağladığını doğrular."""
        s1 = SpamClassifierService.get_instance()
        s2 = SpamClassifierService.get_instance()
        assert s1 is s2

    def test_classify_single_spam(self, sample_spam_text: str):
        """Spam mesajın doğru sınıflandırıldığını doğrular."""
        service = SpamClassifierService.get_instance()
        result = service.classify(sample_spam_text)

        assert isinstance(result, dict)
        assert result["text"] == sample_spam_text
        assert "label" in result
        assert "is_spam" in result
        assert "spam_probability" in result
        assert "confidence_level" in result
        assert 0.0 <= result["spam_probability"] <= 1.0
        assert result["confidence_level"] in ["Düşük", "Orta", "Yüksek"]

    def test_classify_single_ham(self, sample_ham_text: str):
        """Normal (ham) mesajın doğru sınıflandırıldığını doğrular."""
        service = SpamClassifierService.get_instance()
        result = service.classify(sample_ham_text)

        assert result["label"] == "ham"
        assert result["is_spam"] is False
        assert result["spam_probability"] < 0.5

    def test_classify_empty_string(self):
        """Boş metin geldiğinde güvenli varsayılan değer dönmeli."""
        service = SpamClassifierService.get_instance()
        result = service.classify("")

        assert result["label"] == "ham"
        assert result["is_spam"] is False
        assert result["spam_probability"] == 0.0
        assert result["confidence_level"] == "Düşük"

    def test_classify_batch(self, sample_spam_text: str, sample_ham_text: str):
        """Toplu sınıflandırma metodunu doğrular."""
        service = SpamClassifierService.get_instance()
        messages = [sample_spam_text, sample_ham_text, "Hello friend!"]
        results = service.classify_batch(messages)

        assert len(results) == 3
        assert results[0]["is_spam"] is True
        assert results[1]["is_spam"] is False

    def test_confidence_level_computation(self):
        """Güven seviyesi eşiklerinin doğruluğunu kontrol eder."""
        service = SpamClassifierService.get_instance()
        assert service._get_confidence_level(0.95) == "Yüksek"
        assert service._get_confidence_level(0.05) == "Yüksek"
        assert service._get_confidence_level(0.70) == "Orta"
        assert service._get_confidence_level(0.30) == "Orta"
        assert service._get_confidence_level(0.50) == "Düşük"
