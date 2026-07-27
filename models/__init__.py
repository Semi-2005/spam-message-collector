# models/__init__.py
"""
Model Training & Serialization
===============================
Spam mesaj sınıflandırma projesi için model eğitim, değerlendirme ve
serileştirme modülü.

Bu modül, temizlenmiş veriyi TF-IDF ile vektörleştirip birden fazla
makine öğrenmesi modelini eğiterek en iyi performans gösteren modeli
disk üzerine kaydeder.

Kullanım:
    >>> from models import SpamModelTrainer
    >>> trainer = SpamModelTrainer()
    >>> trainer.run()
"""

from models.train_model import SpamModelTrainer

__all__ = ["SpamModelTrainer"]
__version__ = "1.0.0"
