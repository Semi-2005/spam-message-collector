# preprocessing/__init__.py
"""
NLP Preprocessing Pipeline
===========================
Spam mesaj sınıflandırma projesi için metin ön işleme modülü.

Bu modül, ham metin verisini makine öğrenmesi modelleri için uygun hale
getiren kapsamlı bir NLP pipeline sunar.

Kullanım:
    >>> from preprocessing import TextCleaner
    >>> cleaner = TextCleaner()
    >>> cleaner.clean_text("FREE entry!! Win a prize NOW!!!")
    'free entry win prize'
"""

from preprocessing.text_cleaner import TextCleaner

__all__ = ["TextCleaner"]
__version__ = "1.0.0"
