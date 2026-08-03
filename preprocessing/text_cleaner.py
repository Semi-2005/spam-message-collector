# preprocessing/text_cleaner.py
"""
Text Cleaner — NLP Ön İşleme Pipeline
======================================

Ham SMS metinlerini makine öğrenmesi modelleri için hazırlayan kapsamlı
bir metin temizleme sınıfı.

Pipeline adımları (sırasıyla):
    1. Küçük harfe çevirme (lowercasing)
    2. URL'leri kaldırma
    3. E-posta adreslerini kaldırma
    4. Telefon numaralarını kaldırma
    5. Noktalama işaretlerini ve özel karakterleri kaldırma
    6. Sayıları kaldırma (opsiyonel)
    7. Tokenizasyon
    8. Stop-word'leri silme
    9. Lemmatization (WordNetLemmatizer)
   10. Kısa token'ları filtreleme (min 2 karakter)

Tasarım kararları:
    - Her adım bağımsız bir metot → unit test ve debug kolaylığı.
    - `clean_text()` tüm pipeline'ı tek çağrıda çalıştırır.
    - `clean_dataframe()` toplu işlem için DataFrame bazlı API sunar.
    - Tüm NLTK kaynakları lazy-load edilir (ilk erişimde indirilir).
    - Thread-safe: Lemmatizer ve stopwords immutable/stateless.

Kullanım:
    >>> from preprocessing.text_cleaner import TextCleaner
    >>> cleaner = TextCleaner(remove_numbers=True)
    >>> cleaner.clean_text("WINNER!! You've won $1000!!! Call 09061234567")
    'winner win call'
    >>> cleaner.clean_text("Go until jurong point, crazy.. Available only in bugis")
    'go jurong point crazy available bugis'
"""

from __future__ import annotations

import logging
import re
import ssl
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK Kaynaklarını Güvenli Şekilde İndir
# ---------------------------------------------------------------------------
_REQUIRED_NLTK_RESOURCES = [
    ("corpora", "stopwords"),
    ("corpora", "wordnet"),
    ("tokenizers", "punkt_tab"),
    ("corpora", "omw-1.4"),
]


def _ensure_nltk_resources() -> None:
    """Gerekli NLTK kaynaklarının mevcut olduğunu kontrol eder, yoksa indirir.

    SSL sertifika sorunları yaşayan ortamlar için unverified context kullanır.
    Bu fonksiyon idempotent'tir — zaten indirilmiş kaynakları tekrar indirmez.
    """
    missing: list[tuple[str, str]] = []
    for category, resource in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(f"{category}/{resource}")
        except LookupError:
            missing.append((category, resource))

    if not missing:
        return

    # SSL sertifika sorunlarını önlemek için unverified context kullan
    _original_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        for category, resource in missing:
            logger.info("NLTK kaynağı indiriliyor: %s/%s", category, resource)
            success = nltk.download(resource, quiet=True)
            if not success:
                logger.error("NLTK kaynağı indirilemedi: %s", resource)
    finally:
        ssl._create_default_https_context = _original_ctx


# Modül yüklendiğinde kaynakları kontrol et
_ensure_nltk_resources()


# ---------------------------------------------------------------------------
# Derleme Zamanında Oluşturulan Regex Pattern'leri (performans optimizasyonu)
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_PHONE = re.compile(r"\b[\d\-\+\(\)]{7,15}\b")
_RE_NUMBERS = re.compile(r"\d+")
_RE_EXTRA_WHITESPACE = re.compile(r"\s+")
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")


class TextCleaner:
    """SMS metinlerini temizleyen ve normalize eden NLP pipeline sınıfı.

    Bu sınıf, ham metin verisini makine öğrenmesi modelleri tarafından
    kullanılabilecek temiz token dizilerine dönüştürür.

    Attributes:
        remove_numbers: True ise metindeki sayısal ifadeleri kaldırır.
        min_token_length: Minimum token uzunluğu. Daha kısa tokenlar filtrelenir.
        custom_stopwords: Varsayılan NLTK stopwords setine ek olarak
                          kaldırılacak kelimeler.

    Example:
        >>> cleaner = TextCleaner(remove_numbers=True, min_token_length=2)
        >>> cleaner.clean_text("Free entry in 2 a wkly comp!")
        'free entry wkly comp'
    """

    def __init__(
        self,
        remove_numbers: bool = True,
        min_token_length: int = 2,
        custom_stopwords: set[str] | None = None,
    ) -> None:
        # Lemmatizer — NLTK WordNet tabanlı, fiil+isim formlarına indirgeme
        self._lemmatizer = WordNetLemmatizer()

        # Stop-words seti (İngilizce)
        self._stop_words: set[str] = set(stopwords.words("english"))
        if custom_stopwords:
            self._stop_words |= custom_stopwords

        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length

        logger.info(
            "TextCleaner başlatıldı — remove_numbers=%s, min_token_length=%d, stopwords_count=%d",
            self.remove_numbers,
            self.min_token_length,
            len(self._stop_words),
        )

    # ------------------------------------------------------------------
    # Bireysel Temizleme Adımları
    # ------------------------------------------------------------------

    @staticmethod
    def to_lowercase(text: str) -> str:
        """Metni küçük harfe çevirir.

        Args:
            text: İşlenecek metin.

        Returns:
            Tüm karakterleri küçük harfe dönüştürülmüş metin.
        """
        return text.lower()

    @staticmethod
    def remove_urls(text: str) -> str:
        """URL'leri metinden kaldırır.

        HTTP/HTTPS ve www ile başlayan URL kalıplarını yakalar ve siler.

        Args:
            text: İşlenecek metin.

        Returns:
            URL'lerden arındırılmış metin.
        """
        return _RE_URL.sub("", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """E-posta adreslerini metinden kaldırır.

        Args:
            text: İşlenecek metin.

        Returns:
            E-posta adreslerinden arındırılmış metin.
        """
        return _RE_EMAIL.sub("", text)

    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Telefon numarası kalıplarını metinden kaldırır.

        7-15 basamak arasındaki sayı dizilerini (tire, parantez, artı dahil)
        telefon numarası olarak kabul edip siler.

        Args:
            text: İşlenecek metin.

        Returns:
            Telefon numaralarından arındırılmış metin.
        """
        return _RE_PHONE.sub("", text)

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Noktalama işaretlerini ve özel karakterleri kaldırır.

        Sadece küçük harf ve boşluk karakterlerini bırakır.
        Bu metot `to_lowercase` çağrısından SONRA kullanılmalıdır.

        Args:
            text: İşlenecek (küçük harfe çevrilmiş) metin.

        Returns:
            Yalnızca harf ve boşluk içeren metin.
        """
        return _RE_NON_ALPHA.sub("", text)

    @staticmethod
    def remove_numbers_from_text(text: str) -> str:
        """Sayısal ifadeleri metinden kaldırır.

        Args:
            text: İşlenecek metin.

        Returns:
            Sayılardan arındırılmış metin.
        """
        return _RE_NUMBERS.sub("", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Ardışık boşluk karakterlerini tek boşluğa indirger.

        Args:
            text: İşlenecek metin.

        Returns:
            Normalize edilmiş boşluklu metin.
        """
        return _RE_EXTRA_WHITESPACE.sub(" ", text).strip()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Stop-word'leri token listesinden kaldırır.

        NLTK'nın İngilizce stop-words seti kullanılır. Opsiyonel olarak
        __init__'te verilen custom_stopwords da dahil edilir.

        Args:
            tokens: Tokenize edilmiş kelime listesi.

        Returns:
            Stop-word'lerden arındırılmış token listesi.
        """
        return [t for t in tokens if t not in self._stop_words]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Token listesindeki kelimeleri kök formlarına indirger.

        WordNetLemmatizer kullanarak her token'ı hem fiil hem isim formuyla
        lemmatize eder. Sonuç olarak en kısa (en temel) formu döndürür.

        Örnek:
            'running' → 'run', 'better' → 'better', 'geese' → 'goose'

        Args:
            tokens: Tokenize edilmiş kelime listesi.

        Returns:
            Lemmatize edilmiş token listesi.
        """
        lemmatized = []
        for token in tokens:
            # Hem fiil hem isim olarak lemmatize et, en kısa olanı al
            lemma_verb = self._lemmatizer.lemmatize(token, pos="v")
            lemma_noun = self._lemmatizer.lemmatize(token, pos="n")
            # En kısa form genellikle kök formdur
            lemma = lemma_verb if len(lemma_verb) <= len(lemma_noun) else lemma_noun
            lemmatized.append(lemma)
        return lemmatized

    def filter_short_tokens(self, tokens: list[str]) -> list[str]:
        """Minimum uzunluğun altındaki token'ları filtreler.

        Tek harfli token'lar genellikle anlamsızdır ve model performansına
        katkı sağlamazlar (örn: 'a', 'I', 'u').

        Args:
            tokens: Tokenize edilmiş kelime listesi.

        Returns:
            Minimum uzunluk kriterini karşılayan token listesi.
        """
        return [t for t in tokens if len(t) >= self.min_token_length]

    # ------------------------------------------------------------------
    # Ana Pipeline
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Tek bir metin üzerinde tüm NLP ön işleme pipeline'ını çalıştırır.

        İşlem sırası:
            lowercase → URL kaldır → e-posta kaldır → telefon kaldır →
            (sayı kaldır) → noktalama kaldır → boşluk normalize et →
            tokenize → stopword kaldır → lemmatize → kısa token filtrele

        Args:
            text: İşlenecek ham metin.

        Returns:
            Temizlenmiş ve normalize edilmiş metin (token'lar boşlukla birleşik).
            Eğer girdi boş veya None ise, boş string döner.
        """
        # Guard: boş veya None girdi
        if not text or not isinstance(text, str):
            return ""

        # Adım 1: Küçük harfe çevir
        text = self.to_lowercase(text)

        # Adım 2: URL'leri kaldır
        text = self.remove_urls(text)

        # Adım 3: E-posta adreslerini kaldır
        text = self.remove_emails(text)

        # Adım 4: Telefon numaralarını kaldır
        text = self.remove_phone_numbers(text)

        # Adım 5: Sayıları kaldır (opsiyonel)
        if self.remove_numbers:
            text = self.remove_numbers_from_text(text)

        # Adım 6: Noktalama işaretlerini ve özel karakterleri kaldır
        text = self.remove_punctuation(text)

        # Adım 7: Boşlukları normalize et
        text = self.normalize_whitespace(text)

        # Adım 8: Tokenize et
        tokens: list[str] = word_tokenize(text)

        # Adım 9: Stop-word'leri kaldır
        tokens = self.remove_stopwords(tokens)

        # Adım 10: Lemmatize et
        tokens = self.lemmatize(tokens)

        # Adım 11: Kısa token'ları filtrele
        tokens = self.filter_short_tokens(tokens)

        return " ".join(tokens)

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "v2",
        label_column: str = "v1",
        output_text_col: str = "cleaned_text",
        output_label_col: str = "label",
    ) -> pd.DataFrame:
        """DataFrame üzerinde toplu metin temizleme işlemi yapar.

        Ham veri DataFrame'ini alır, metin sütununu temizler, etiketleri
        sayısal değerlere dönüştürür ve yeni bir DataFrame döndürür.

        Etiket kodlaması:
            - 'ham'  → 0 (güvenli mesaj)
            - 'spam' → 1 (spam mesaj)

        Args:
            df: Ham veriyi içeren pandas DataFrame.
            text_column: Ham metin sütununun adı.
            label_column: Etiket sütununun adı ('ham'/'spam').
            output_text_col: Temizlenmiş metin sütununun çıktı adı.
            output_label_col: Sayısal etiket sütununun çıktı adı.

        Returns:
            Temizlenmiş metin ve sayısal etiketleri içeren yeni DataFrame.
            Sütunlar: [output_label_col, output_text_col, 'original_text',
                        'word_count', 'char_count']
        """
        logger.info(
            "DataFrame temizleme başlatıldı — %d satır, metin sütunu: '%s'",
            len(df),
            text_column,
        )

        # Sadece ilgili sütunları al, diğerlerini (Unnamed) at
        result = pd.DataFrame()

        # Orijinal metni sakla (karşılaştırma/debug için)
        result["original_text"] = df[text_column].copy()

        # Etiketleri sayısal değerlere dönüştür
        label_map = {"ham": 0, "spam": 1}
        result[output_label_col] = df[label_column].map(label_map)

        # NLP pipeline'ını uygula
        result[output_text_col] = df[text_column].apply(self.clean_text)

        # İstatistiksel sütunlar ekle (EDA ve feature engineering için faydalı)
        result["word_count"] = result[output_text_col].apply(lambda x: len(x.split()) if x else 0)
        result["char_count"] = result[output_text_col].apply(len)

        # Boş temizlenmiş metinleri düşür (çok nadir)
        empty_count = (result[output_text_col] == "").sum()
        if empty_count > 0:
            logger.warning("%d satır temizleme sonrası boş kaldı ve düşürüldü.", empty_count)
            result = result[result[output_text_col] != ""].reset_index(drop=True)

        logger.info(
            "Temizleme tamamlandı — %d satır, ortalama kelime sayısı: %.1f",
            len(result),
            result["word_count"].mean(),
        )

        return result


# ---------------------------------------------------------------------------
# Standalone Script: Pipeline'ı çalıştır ve data/processed/cleaned_data.csv oluştur
# ---------------------------------------------------------------------------
def run_pipeline(
    input_path: str = "data/raw/spam.csv",
    output_path: str = "data/processed/cleaned_data.csv",
) -> pd.DataFrame:
    """NLP preprocessing pipeline'ını çalıştırarak temizlenmiş veriyi kaydeder.

    Bu fonksiyon, ham veriyi okur, TextCleaner ile temizler ve sonucu
    CSV formatında kaydeder. Ayrıca temel veri kalitesi istatistiklerini
    loglar.

    Args:
        input_path: Ham veri dosyasının yolu.
        output_path: Temizlenmiş verinin kaydedileceği dosya yolu.

    Returns:
        Temizlenmiş veriyi içeren pandas DataFrame.

    Raises:
        FileNotFoundError: Eğer input_path'teki dosya bulunamazsa.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Ham veri dosyası bulunamadı: {input_file.resolve()}\n"
            f"Lütfen önce 'python download_dataset.py' çalıştırın."
        )

    # Loglama konfigürasyonu
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("NLP Preprocessing Pipeline Başlatılıyor")
    logger.info("=" * 60)

    # 1. Ham veriyi oku
    logger.info("Ham veri okunuyor: %s", input_file)
    df_raw = pd.read_csv(input_file, encoding="latin-1")
    logger.info("Ham veri boyutu: %d satır, %d sütun", *df_raw.shape)

    # Veri seti hakkında bilgi
    spam_count = (df_raw["v1"] == "spam").sum()
    ham_count = (df_raw["v1"] == "ham").sum()
    logger.info(
        "Sınıf dağılımı — Ham: %d (%.1f%%), Spam: %d (%.1f%%)",
        ham_count,
        ham_count / len(df_raw) * 100,
        spam_count,
        spam_count / len(df_raw) * 100,
    )

    # 2. TextCleaner'ı başlat ve pipeline'ı çalıştır
    cleaner = TextCleaner(remove_numbers=True, min_token_length=2)
    df_clean = cleaner.clean_dataframe(df_raw)

    # 3. Çıktı dizinini oluştur ve kaydet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False, encoding="utf-8")
    logger.info("Temizlenmiş veri kaydedildi: %s", output_file)

    # 4. Kalite raporu
    logger.info("=" * 60)
    logger.info("KALITE RAPORU")
    logger.info("=" * 60)
    logger.info("Toplam satır:    %d", len(df_clean))
    logger.info("Null değer var mı: %s", df_clean.isnull().any().any())
    logger.info("Boş metin sayısı:  %d", (df_clean["cleaned_text"] == "").sum())
    logger.info("Ortalama kelime sayısı: %.1f", df_clean["word_count"].mean())
    logger.info("Medyan kelime sayısı:   %.1f", df_clean["word_count"].median())
    logger.info("Ort. karakter sayısı:   %.1f", df_clean["char_count"].mean())
    logger.info("Sınıf dağılımı (temiz):")
    for label, count in df_clean["label"].value_counts().items():
        label_name = "Ham" if label == 0 else "Spam"
        logger.info(
            "  %s (=%d): %d (%.1f%%)", label_name, label, count, count / len(df_clean) * 100
        )
    logger.info("=" * 60)
    logger.info("Pipeline başarıyla tamamlandı! ✅")
    logger.info("=" * 60)

    return df_clean


if __name__ == "__main__":
    run_pipeline()
