# models/train_model.py
"""
Spam Model Trainer — Eğitim, Değerlendirme ve Serileştirme Pipeline
====================================================================

Temizlenmiş SMS verisi üzerinde birden fazla makine öğrenmesi modelini
eğitip karşılaştıran ve en iyi modeli diske kaydeden kapsamlı training
pipeline.

Pipeline akışı:
    1. Temizlenmiş veriyi yükle (data/processed/cleaned_data.csv)
    2. Stratified train/test split (%80/%20)
    3. TF-IDF vektörizasyon (unigram + bigram)
    4. 4 farklı modeli eğit: MultinomialNB, LinearSVC, RandomForest, LogisticRegression
    5. Her model için Stratified K-Fold Cross-Validation (k=5)
    6. Test seti üzerinde nihai değerlendirme (Accuracy, Precision, Recall, F1)
    7. En iyi modeli + vectorizer'ı .joblib olarak kaydet
    8. Model metadata'yı JSON olarak kaydet

Tasarım kararları:
    - Pipeline nesnesi (sklearn.pipeline.Pipeline) kullanılarak vectorizer
      ile model tek bir artifact olarak serileştirilir → inference sırasında
      tek bir .predict() çağrısı yeterli olur.
    - Stratified split kullanılarak imbalanced sınıf dağılımının korunması
      sağlanır (ham: ~87%, spam: ~13%).
    - Cross-validation ile overfitting riski azaltılır ve model
      güvenilirliği ölçülür.
    - TF-IDF'te sublinear_tf=True → logaritmik term-frequency scaling,
      yüksek frekanslı terimlerin domine etmesini engeller.
    - max_features=5000 → vocabulary boyutunu sınırlar, hafıza verimli.

Kullanım:
    Terminal:
        $ python -m models.train_model
        $ python -m models.train_model --test-size 0.25 --cv-folds 10

    Python:
        >>> from models.train_model import SpamModelTrainer
        >>> trainer = SpamModelTrainer(test_size=0.2, cv_folds=5)
        >>> results = trainer.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Logging konfigürasyonu
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "artifacts"


# ---------------------------------------------------------------------------
# Model tanımları — her bir aday model ve hiperparametreleri
# ---------------------------------------------------------------------------
def _get_candidate_models() -> dict[str, Any]:
    """Eğitilecek aday modelleri ve konfigürasyonlarını döndürür.

    Her model için:
        - estimator: sklearn estimator nesnesi
        - description: Modelin Türkçe açıklaması
        - hyperparams: Anahtar hiperparametre bilgileri (metadata için)

    Returns:
        Model adını anahtar olarak kullanan sözlük.
    """
    return {
        "MultinomialNB": {
            "estimator": MultinomialNB(alpha=0.1),
            "description": "Multinomial Naive Bayes — Metin sınıflandırma "
                           "için temel ve hızlı bir olasılıksal model. "
                           "alpha=0.1 Laplace smoothing ile.",
            "hyperparams": {"alpha": 0.1, "fit_prior": True},
        },
        "ComplementNB": {
            "estimator": ComplementNB(alpha=0.5),
            "description": "Complement Naive Bayes — İmbalanced veri setleri "
                           "için optimize edilmiş NB varyantı. Spam/ham "
                           "dengesizliğine karşı dayanıklı.",
            "hyperparams": {"alpha": 0.5, "norm": False},
        },
        "LinearSVC": {
            "estimator": LinearSVC(
                C=1.0,
                max_iter=10000,
                class_weight="balanced",
                dual="auto",
            ),
            "description": "Linear Support Vector Classifier — Yüksek boyutlu "
                           "TF-IDF uzayında güçlü ayrım yeteneği. "
                           "class_weight='balanced' ile sınıf dengesizliği "
                           "telafi edilir.",
            "hyperparams": {
                "C": 1.0,
                "max_iter": 10000,
                "class_weight": "balanced",
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "description": "Random Forest — Ensemble tabanlı, overfitting'e "
                           "karşı dirençli model. Paralel eğitim ile hızlı.",
            "hyperparams": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 5,
                "class_weight": "balanced",
            },
        },
        "LogisticRegression": {
            "estimator": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
            "description": "Logistic Regression — Basit, yorumlanabilir ve "
                           "güçlü bir linear model. Baseline olarak "
                           "kullanılır. Olasılık çıktısı üretir.",
            "hyperparams": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
            },
        },
    }


# ---------------------------------------------------------------------------
# Ana Trainer Sınıfı
# ---------------------------------------------------------------------------
class SpamModelTrainer:
    """Spam sınıflandırma modellerini eğiten, değerlendiren ve kaydeden sınıf.

    Bu sınıf, uçtan uca model eğitim pipeline'ını yönetir:
    veri yükleme → split → vektörizasyon → model eğitimi →
    cross-validation → test değerlendirme → serileştirme.

    Attributes:
        data_path: Temizlenmiş verinin dosya yolu.
        model_dir: Eğitilmiş modellerin kaydedileceği dizin.
        test_size: Test seti oranı (0-1 arası).
        cv_folds: Cross-validation katlama sayısı.
        random_state: Tekrarlanabilirlik için rastgelelik tohumu.

    Example:
        >>> trainer = SpamModelTrainer(test_size=0.2, cv_folds=5)
        >>> results = trainer.run()
        >>> print(results["best_model_name"])
        'LinearSVC'
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.data_path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Çalışma zamanında doldurulacak alanlar
        self._df: pd.DataFrame | None = None
        self._X_train: np.ndarray | None = None
        self._X_test: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 1. Veri Yükleme
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Temizlenmiş veriyi CSV dosyasından yükler.

        Returns:
            Temizlenmiş veriyi içeren DataFrame.

        Raises:
            FileNotFoundError: Veri dosyası bulunamazsa.
            ValueError: Gerekli sütunlar eksikse.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Temizlenmiş veri dosyası bulunamadı: {self.data_path}\n"
                f"Lütfen önce preprocessing pipeline'ını çalıştırın:\n"
                f"  python -m preprocessing.text_cleaner"
            )

        logger.info("Veri yükleniyor: %s", self.data_path)
        self._df = pd.read_csv(self.data_path)

        # Gerekli sütunları doğrula
        required_cols = {"cleaned_text", "label"}
        missing = required_cols - set(self._df.columns)
        if missing:
            raise ValueError(
                f"Veri dosyasında gerekli sütunlar eksik: {missing}\n"
                f"Mevcut sütunlar: {list(self._df.columns)}"
            )

        # Boş veya NaN metin satırlarını düşür
        initial_len = len(self._df)
        self._df = self._df.dropna(subset=["cleaned_text"])
        self._df = self._df[self._df["cleaned_text"].str.strip() != ""]
        self._df = self._df.reset_index(drop=True)
        dropped = initial_len - len(self._df)
        if dropped > 0:
            logger.warning("%d boş/NaN satır düşürüldü.", dropped)

        logger.info(
            "Veri yüklendi — %d satır | Ham: %d | Spam: %d",
            len(self._df),
            (self._df["label"] == 0).sum(),
            (self._df["label"] == 1).sum(),
        )

        return self._df

    # ------------------------------------------------------------------
    # 2. Train/Test Split
    # ------------------------------------------------------------------
    def split_data(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Veriyi stratified train/test setlerine ayırır.

        Stratified split, orijinal sınıf dağılımının hem train hem test
        setlerinde korunmasını sağlar. Bu özellikle imbalanced veri
        setlerinde (ham:87%, spam:13%) kritik öneme sahiptir.

        Returns:
            (X_train, X_test, y_train, y_test) tuple'ı.
        """
        if self._df is None:
            self.load_data()

        X = self._df["cleaned_text"]
        y = self._df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "Train/Test split tamamlandı — "
            "Train: %d (%d ham, %d spam) | Test: %d (%d ham, %d spam)",
            len(X_train),
            (y_train == 0).sum(), (y_train == 1).sum(),
            len(X_test),
            (y_test == 0).sum(), (y_test == 1).sum(),
        )

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # 3. TF-IDF Vektörizasyon
    # ------------------------------------------------------------------
    def build_vectorizer(self) -> TfidfVectorizer:
        """TF-IDF vektörleştiriciyi yapılandırır ve döndürür.

        Konfigürasyon kararları:
            - ngram_range=(1,2): Unigram + bigram → "free call" gibi
              spam kalıplarını yakalar.
            - max_features=5000: Vocabulary boyutunu sınırlar → hafıza
              verimli ve overfitting'e karşı dirençli.
            - sublinear_tf=True: TF değerlerine log(1+tf) uygular →
              yüksek frekanslı terimlerin baskınlığını azaltır.
            - min_df=2: En az 2 dokümanda geçen terimleri dahil eder →
              çok nadir terimleri (typo vs.) filtreler.
            - max_df=0.95: Dokümanların %95'inden fazlasında geçen
              terimleri hariç tutar → quasi-stopwords'ü eler.

        Returns:
            Yapılandırılmış TfidfVectorizer nesnesi.
        """
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            dtype=np.float64,
        )

        logger.info(
            "TF-IDF Vectorizer yapılandırıldı — "
            "ngram=(1,2), max_features=5000, sublinear_tf=True"
        )

        return self._vectorizer

    # ------------------------------------------------------------------
    # 4. Model Eğitimi ve Cross-Validation
    # ------------------------------------------------------------------
    def train_and_evaluate(
        self,
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict[str, dict]:
        """Tüm aday modelleri eğitir, cross-validation ve test değerlendirmesi yapar.

        Her model için:
            1. TF-IDF vectorizer + model birleştirilip Pipeline oluşturulur
            2. Stratified 5-Fold CV ile eğitim seti üzerinde F1 skoru ölçülür
            3. Test seti üzerinde nihai metrikler hesaplanır
            4. Classification report ve confusion matrix üretilir

        Args:
            X_train: Eğitim metinleri.
            X_test: Test metinleri.
            y_train: Eğitim etiketleri.
            y_test: Test etiketleri.

        Returns:
            Her model için metrikleri içeren sözlük.
        """
        candidates = _get_candidate_models()
        self._results = {}

        cv_splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        logger.info("=" * 65)
        logger.info("MODEL EĞİTİMİ BAŞLATIYOR — %d aday model", len(candidates))
        logger.info("=" * 65)

        for name, config in candidates.items():
            logger.info("-" * 65)
            logger.info("📊 Eğitiliyor: %s", name)
            logger.info("   %s", config["description"])
            logger.info("-" * 65)

            # Pipeline oluştur: TF-IDF → Model
            # Her model kendi vectorizer instance'ına sahip olacak
            # böylece cross-validation sırasında data leakage önlenir.
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                    dtype=np.float64,
                )),
                ("clf", config["estimator"]),
            ])

            # --- Cross-Validation (eğitim seti üzerinde) ---
            t_start = time.perf_counter()

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring="f1",
                n_jobs=-1,
            )

            # --- Tam eğitim seti üzerinde eğit ---
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - t_start

            # --- Test seti üzerinde tahmin ---
            y_pred = pipeline.predict(X_test)

            # --- Metrikler ---
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                target_names=["Ham (0)", "Spam (1)"],
                output_dict=True,
            )

            self._results[name] = {
                "pipeline": pipeline,
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std()),
                "cv_f1_scores": cv_scores.tolist(),
                "test_accuracy": float(test_accuracy),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "test_f1": float(test_f1),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "train_time_seconds": round(train_time, 3),
                "description": config["description"],
                "hyperparams": config["hyperparams"],
            }

            # Sonuçları logla
            logger.info(
                "   CV F1:    %.4f (±%.4f)",
                cv_scores.mean(), cv_scores.std(),
            )
            logger.info("   Test Accuracy:  %.4f", test_accuracy)
            logger.info("   Test Precision: %.4f", test_precision)
            logger.info("   Test Recall:    %.4f", test_recall)
            logger.info("   Test F1:        %.4f", test_f1)
            logger.info(
                "   Confusion Matrix: TN=%d FP=%d FN=%d TP=%d",
                cm[0][0], cm[0][1], cm[1][0], cm[1][1],
            )
            logger.info("   Eğitim süresi:  %.3f sn", train_time)

        return self._results

    # ------------------------------------------------------------------
    # 5. Karşılaştırma Tablosu
    # ------------------------------------------------------------------
    def print_comparison_table(self) -> pd.DataFrame:
        """Tüm modellerin metriklerini karşılaştırmalı tablo olarak gösterir.

        Returns:
            Karşılaştırma tablosunu içeren DataFrame.
        """
        if not self._results:
            raise RuntimeError("Önce train_and_evaluate() çalıştırılmalı.")

        rows = []
        for name, metrics in self._results.items():
            rows.append({
                "Model": name,
                "CV F1 (mean)": f"{metrics['cv_f1_mean']:.4f}",
                "CV F1 (std)": f"±{metrics['cv_f1_std']:.4f}",
                "Test Accuracy": f"{metrics['test_accuracy']:.4f}",
                "Test Precision": f"{metrics['test_precision']:.4f}",
                "Test Recall": f"{metrics['test_recall']:.4f}",
                "Test F1": f"{metrics['test_f1']:.4f}",
                "Süre (sn)": f"{metrics['train_time_seconds']:.3f}",
            })

        df_comparison = pd.DataFrame(rows)
        # Test F1'e göre sırala
        df_comparison = df_comparison.sort_values(
            "Test F1", ascending=False
        ).reset_index(drop=True)

        logger.info("\n" + "=" * 65)
        logger.info("MODEL KARŞILAŞTIRMA TABLOSU (Test F1'e göre sıralı)")
        logger.info("=" * 65)
        print(df_comparison.to_string(index=False))
        print()

        return df_comparison

    # ------------------------------------------------------------------
    # 6. En İyi Modeli Seç
    # ------------------------------------------------------------------
    def select_best_model(self) -> tuple[str, Pipeline, dict]:
        """En iyi performans gösteren modeli seçer.

        Seçim kriterleri (öncelik sırasıyla):
            1. Test F1 Score (ana kriter — spam'de precision+recall dengesi)
            2. CV F1 Mean (eşitlik durumunda — generalizasyon gücü)

        Returns:
            (model_adı, pipeline, metrikler) tuple'ı.
        """
        if not self._results:
            raise RuntimeError("Önce train_and_evaluate() çalıştırılmalı.")

        best_name = max(
            self._results,
            key=lambda k: (
                self._results[k]["test_f1"],
                self._results[k]["cv_f1_mean"],
            ),
        )

        best_pipeline = self._results[best_name]["pipeline"]
        best_metrics = self._results[best_name]

        logger.info("🏆 En iyi model: %s (Test F1: %.4f)", best_name, best_metrics["test_f1"])

        return best_name, best_pipeline, best_metrics

    # ------------------------------------------------------------------
    # 7. Model Serileştirme
    # ------------------------------------------------------------------
    def save_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        metrics: dict,
    ) -> dict[str, Path]:
        """Eğitilmiş pipeline ve metadata'yı diske kaydeder.

        Kaydedilen dosyalar:
            - best_model_pipeline.joblib: TF-IDF + Model pipeline
            - model_metadata.json: Performans metrikleri, hiperparametreler,
              eğitim zamanı ve konfigürasyon bilgileri

        Args:
            model_name: Seçilen modelin adı.
            pipeline: Eğitilmiş sklearn Pipeline nesnesi.
            metrics: Model performans metrikleri.

        Returns:
            Kaydedilen dosya yollarını içeren sözlük.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline'ı kaydet (.joblib)
        pipeline_path = self.model_dir / "best_model_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path, compress=3)
        pipeline_size_mb = pipeline_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Pipeline kaydedildi: %s (%.2f MB)", pipeline_path, pipeline_size_mb
        )

        # Metadata JSON oluştur
        metadata = {
            "model_name": model_name,
            "model_description": metrics["description"],
            "hyperparameters": metrics["hyperparams"],
            "performance": {
                "cv_f1_mean": metrics["cv_f1_mean"],
                "cv_f1_std": metrics["cv_f1_std"],
                "cv_f1_scores": metrics["cv_f1_scores"],
                "test_accuracy": metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            },
            "training_config": {
                "test_size": self.test_size,
                "cv_folds": self.cv_folds,
                "random_state": self.random_state,
                "tfidf_config": {
                    "ngram_range": [1, 2],
                    "max_features": 5000,
                    "sublinear_tf": True,
                    "min_df": 2,
                    "max_df": 0.95,
                },
            },
            "training_info": {
                "train_time_seconds": metrics["train_time_seconds"],
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "data_path": str(self.data_path),
                "pipeline_path": str(pipeline_path),
                "pipeline_size_mb": round(pipeline_size_mb, 2),
            },
            "all_model_results": {
                name: {
                    "cv_f1_mean": m["cv_f1_mean"],
                    "test_f1": m["test_f1"],
                    "test_accuracy": m["test_accuracy"],
                    "description": m["description"],
                }
                for name, m in self._results.items()
            },
        }

        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("Metadata kaydedildi: %s", metadata_path)

        return {
            "pipeline": pipeline_path,
            "metadata": metadata_path,
        }

    # ------------------------------------------------------------------
    # 8. Ana Çalıştırma Metodu
    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Tam eğitim pipeline'ını sırasıyla çalıştırır.

        Returns:
            Pipeline sonuçlarını içeren sözlük:
                - best_model_name: En iyi modelin adı
                - best_test_f1: En iyi test F1 skoru
                - saved_paths: Kaydedilen dosya yolları
                - comparison_table: Model karşılaştırma DataFrame'i
                - all_results: Tüm modellerin detaylı sonuçları
        """
        total_start = time.perf_counter()

        logger.info("=" * 65)
        logger.info("🚀 SPAM MODEL EĞİTİM PIPELINE BAŞLATILIYOR")
        logger.info("=" * 65)
        logger.info("Konfigürasyon:")
        logger.info("  Veri yolu:     %s", self.data_path)
        logger.info("  Model dizini:  %s", self.model_dir)
        logger.info("  Test oranı:    %.0f%%", self.test_size * 100)
        logger.info("  CV katlamaları: %d", self.cv_folds)
        logger.info("  Random state:  %d", self.random_state)
        logger.info("=" * 65)

        # Adım 1: Veri yükle
        self.load_data()

        # Adım 2: Train/test split
        X_train, X_test, y_train, y_test = self.split_data()

        # Adım 3: Modelleri eğit ve değerlendir
        self.train_and_evaluate(X_train, X_test, y_train, y_test)

        # Adım 4: Karşılaştırma tablosu
        comparison_df = self.print_comparison_table()

        # Adım 5: En iyi modeli seç
        best_name, best_pipeline, best_metrics = self.select_best_model()

        # Adım 6: Kaydet
        saved_paths = self.save_model(best_name, best_pipeline, best_metrics)

        total_time = time.perf_counter() - total_start

        logger.info("\n" + "=" * 65)
        logger.info("✅ PIPELINE TAMAMLANDI!")
        logger.info("=" * 65)
        logger.info("🏆 En iyi model: %s", best_name)
        logger.info("   Test F1:      %.4f", best_metrics["test_f1"])
        logger.info("   Test Accuracy: %.4f", best_metrics["test_accuracy"])
        logger.info("   Pipeline:     %s", saved_paths["pipeline"])
        logger.info("   Metadata:     %s", saved_paths["metadata"])
        logger.info("   Toplam süre:  %.2f saniye", total_time)
        logger.info("=" * 65)

        return {
            "best_model_name": best_name,
            "best_test_f1": best_metrics["test_f1"],
            "saved_paths": {k: str(v) for k, v in saved_paths.items()},
            "comparison_table": comparison_df,
            "all_results": {
                name: {k: v for k, v in m.items() if k != "pipeline"}
                for name, m in self._results.items()
            },
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Komut satırından model eğitim pipeline'ını çalıştırır."""
    parser = argparse.ArgumentParser(
        description="Spam Mesaj Sınıflandırma — Model Eğitim Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python -m models.train_model\n"
            "  python -m models.train_model --test-size 0.25 --cv-folds 10\n"
            "  python -m models.train_model --data-path data/processed/cleaned_data.csv\n"
        ),
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Temizlenmiş veri dosyasının yolu (default: data/processed/cleaned_data.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Modellerin kaydedileceği dizin (default: models/artifacts/)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test seti oranı (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation katlama sayısı (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Rastgelelik tohumu (default: 42)",
    )

    args = parser.parse_args()

    # Logging konfigürasyonu
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    trainer = SpamModelTrainer(
        data_path=args.data_path,
        model_dir=args.model_dir,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    try:
        trainer.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error("❌ Hata: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("❌ Beklenmeyen hata: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
