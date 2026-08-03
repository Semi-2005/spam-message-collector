# backend/schemas.py
"""
Pydantic Schemas — Request/Response Validasyon Modelleri
=========================================================
API'nin input ve output kontratlarını tanımlayan Pydantic v2 modelleri.

Bu modeller aynı zamanda Swagger UI'daki interaktif dokümantasyonun
temelini oluşturur (otomatik example, description, validation mesajları).

Tasarım kararları:
    - Field seviyesinde validasyon → backend'e ulaşmadan hatalı istekler reddedilir.
    - Computed field (confidence_level) → frontend'in ek hesaplama yapmasına gerek kalmaz.
    - model_config ile JSON Schema örnekleri → Swagger UI'da "Try it out" kolaylığı.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Modeli
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    """Sınıflandırma isteği.

    Kullanıcının sınıflandırmak istediği metin mesajını içerir.

    Attributes:
        text: Sınıflandırılacak metin mesajı (1-5000 karakter).
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Spam olup olmadığını kontrol etmek istediğiniz metin mesajı.",
        examples=[
            "Congratulations! You have won a $1,000 gift card. Click here to claim!",
            "Hey, are we still meeting for lunch tomorrow?",
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "WINNER!! You've won $1,000! Call 09061234567 NOW!"},
                {"text": "Hi Mom, I'll be home for dinner tonight."},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response Modeli
# ---------------------------------------------------------------------------
class ClassifyResponse(BaseModel):
    """Sınıflandırma sonucu.

    Modelin tahmin sonucunu, olasılık skorunu ve güvenilirlik
    seviyesini içerir.

    Attributes:
        text: Orijinal giriş metni.
        label: Sınıflandırma etiketi ("spam" veya "ham").
        is_spam: Mesajın spam olarak sınıflandırılıp sınıflandırılmadığı.
        spam_probability: Spam olasılık skoru (0.0 - 1.0 arası).
        confidence_level: İnsan okunabilir güvenilirlik seviyesi.
    """

    text: str = Field(
        ...,
        description="Orijinal giriş metni.",
    )
    label: str = Field(
        ...,
        description="Sınıflandırma etiketi: 'spam' veya 'ham'.",
        examples=["spam", "ham"],
    )
    is_spam: bool = Field(
        ...,
        description="Mesaj spam olarak sınıflandırıldıysa True.",
    )
    spam_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Spam olasılık skoru (0.0 = kesinlikle ham, 1.0 = kesinlikle spam).",
    )
    confidence_level: str = Field(
        ...,
        description="Güvenilirlik seviyesi: 'Düşük', 'Orta' veya 'Yüksek'.",
        examples=["Yüksek", "Orta", "Düşük"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "WINNER!! You've won $1,000! Call NOW!",
                    "label": "spam",
                    "is_spam": True,
                    "spam_probability": 0.97,
                    "confidence_level": "Yüksek",
                },
                {
                    "text": "Hi Mom, I'll be home for dinner.",
                    "label": "ham",
                    "is_spam": False,
                    "spam_probability": 0.03,
                    "confidence_level": "Yüksek",
                },
            ]
        }
    }


# ---------------------------------------------------------------------------
# Batch Request/Response (opsiyonel genişletme için hazır)
# ---------------------------------------------------------------------------
class BatchClassifyRequest(BaseModel):
    """Toplu sınıflandırma isteği.

    Birden fazla mesajı tek bir istekte sınıflandırmak için kullanılır.

    Attributes:
        messages: Sınıflandırılacak metin listesi (1-50 mesaj).
    """

    messages: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Sınıflandırılacak metin mesajları listesi (maks. 50).",
    )


class BatchClassifyResponse(BaseModel):
    """Toplu sınıflandırma sonucu.

    Attributes:
        results: Her mesaj için ayrı ClassifyResponse listesi.
        total: Toplam işlenen mesaj sayısı.
        spam_count: Spam olarak sınıflandırılan mesaj sayısı.
        ham_count: Ham olarak sınıflandırılan mesaj sayısı.
    """

    results: list[ClassifyResponse] = Field(
        ...,
        description="Her mesaj için sınıflandırma sonuçları.",
    )
    total: int = Field(..., description="Toplam işlenen mesaj sayısı.")
    spam_count: int = Field(..., description="Spam olarak sınıflandırılan mesaj sayısı.")
    ham_count: int = Field(..., description="Ham (güvenli) mesaj sayısı.")


# ---------------------------------------------------------------------------
# Health Check Response
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı.

    Attributes:
        status: Servis durumu ("healthy" veya "unhealthy").
        model_loaded: Modelin başarıyla yüklenip yüklenmediği.
        model_name: Yüklü modelin adı.
        model_accuracy: Modelin test doğruluğu.
    """

    status: str = Field(..., description="Servis durumu.", examples=["healthy"])
    model_loaded: bool = Field(..., description="Model yüklü mü?")
    model_name: str | None = Field(None, description="Yüklü modelin adı.")
    model_accuracy: float | None = Field(None, description="Modelin test doğruluğu.")
