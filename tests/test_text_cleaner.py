"""Unit Tests for TextCleaner NLP Preprocessing.
=================================================
Metin temizleme, tokenization, lemmatization ve regex filtreleme
fonksiyonlarının doğruluğunu test eder.
"""

from preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """TextCleaner sınıfı için kapsamlı test senaryoları."""

    def test_basic_lowercasing(self):
        """Metnin küçük harfe çevrildiğini doğrular."""
        cleaner = TextCleaner()
        result = cleaner.clean_text("HELLO WORLD THIS IS A TEST")
        assert "hello" in result
        assert "world" in result
        assert "test" in result
        assert result == result.lower()

    def test_url_removal(self):
        """URL linklerinin temizlendiğini doğrular."""
        cleaner = TextCleaner()
        text = "Check out our free prize at https://spam-link.com/win or http://gift.org"
        cleaned = cleaner.clean_text(text)
        assert "http" not in cleaned
        assert "spam-link" not in cleaned
        assert "prize" in cleaned

    def test_email_removal(self):
        """E-posta adreslerinin temizlendiğini doğrular."""
        cleaner = TextCleaner()
        text = "Contact us at winner@lottery.com or info@spam.org to claim"
        cleaned = cleaner.clean_text(text)
        assert "winner@lottery.com" not in cleaned
        assert "info@spam.org" not in cleaned
        assert "claim" in cleaned

    def test_html_tag_removal(self):
        """HTML etiketlerinin temizlendiğini doğrular."""
        cleaner = TextCleaner()
        text = "<p>Congratulations! You have <b>won</b> a prize.</p>"
        cleaned = cleaner.clean_text(text)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "won" in cleaned
        assert "prize" in cleaned

    def test_number_removal(self):
        """remove_numbers=True olduğunda sayıların temizlendiğini doğrular."""
        cleaner = TextCleaner(remove_numbers=True)
        text = "Call 09061701461 to win 5000 dollars today"
        cleaned = cleaner.clean_text(text)
        assert "09061701461" not in cleaned
        assert "5000" not in cleaned
        assert "dollar" in cleaned or "dollars" in cleaned

    def test_empty_and_whitespace_input(self):
        """Boş metin veya sadece boşluk içeren girdilerde boş string dönmeli."""
        cleaner = TextCleaner()
        assert cleaner.clean_text("") == ""
        assert cleaner.clean_text("   \n\t  ") == ""
        assert cleaner.clean_text("!@#$%^&*()") == ""

    def test_batch_dataframe_clean(self):
        """Toplu DataFrame temizlemenin doğru çalıştığını doğrular."""
        import pandas as pd

        cleaner = TextCleaner()
        df = pd.DataFrame(
            {
                "v2": [
                    "Hello World!",
                    "Win $1000 NOW at http://win.com",
                    "Urgent message 123",
                ],
                "v1": ["ham", "spam", "spam"],
            }
        )
        cleaned_df = cleaner.clean_dataframe(
            df, text_column="v2", label_column="v1", output_text_col="cleaned_text"
        )
        assert "cleaned_text" in cleaned_df.columns
        assert len(cleaned_df) == 3
        assert all(isinstance(val, str) for val in cleaned_df["cleaned_text"])
