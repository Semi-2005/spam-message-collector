// src/App.jsx
/**
 * App — Ana Uygulama Bileşeni (Gün 6 — Geliştirilmiş)
 * ======================================================
 * Spam Message Classifier frontend uygulamasının giriş noktası.
 *
 * Gün 6 İyileştirmeleri:
 *   - Toast notification sistemi (inline alert yerine)
 *   - LoadingSkeleton (API yanıtı beklerken)
 *   - HistoryPanel (localStorage ile kalıcı analiz geçmişi)
 *   - Gelişmiş state yönetimi
 */

import { useState, useEffect, useCallback } from "react";
import MessageForm from "./components/MessageForm";
import ResultCard from "./components/ResultCard";
import Toast from "./components/Toast";
import LoadingSkeleton from "./components/LoadingSkeleton";
import HistoryPanel from "./components/HistoryPanel";
import useToast from "./hooks/useToast";
import { classifyMessage, checkHealth } from "./services/api";

// ---------------------------------------------------------------------------
// LocalStorage Helpers
// ---------------------------------------------------------------------------
const HISTORY_KEY = "spam_detector_history";
const MAX_HISTORY = 20;

function loadHistory() {
  try {
    const saved = localStorage.getItem(HISTORY_KEY);
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
  } catch {
    // localStorage dolu veya erişilemez — sessizce devam et
  }
}

// ---------------------------------------------------------------------------
// App Component
// ---------------------------------------------------------------------------
export default function App() {
  // State
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [isHealthChecking, setIsHealthChecking] = useState(true);
  const [history, setHistory] = useState(loadHistory);

  // Toast hook
  const { toasts, addToast, removeToast } = useToast();

  // Startup health check
  useEffect(() => {
    async function fetchHealth() {
      try {
        const data = await checkHealth();
        setHealth(data);
        if (data.status === "healthy") {
          addToast(
            `API bağlantısı kuruldu — Model: ${data.model_name || "LinearSVC"}, Doğruluk: %${(data.model_accuracy * 100).toFixed(1)}`,
            "success"
          );
        }
      } catch {
        setHealth(null);
        addToast(
          "Backend bağlantısı kurulamadı. Lütfen API sunucusunu başlatın.",
          "warning",
          6000
        );
      } finally {
        setIsHealthChecking(false);
      }
    }
    fetchHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mesaj sınıflandırma handler
  const handleClassify = useCallback(async (text) => {
    setIsLoading(true);
    setResult(null);

    try {
      const data = await classifyMessage(text);

      // Timestamp ekle
      const resultWithTime = { ...data, timestamp: Date.now() };
      setResult(resultWithTime);

      // Geçmişe ekle (en yenisi başta)
      setHistory((prev) => {
        const updated = [resultWithTime, ...prev].slice(0, MAX_HISTORY);
        saveHistory(updated);
        return updated;
      });

      // Başarı toast
      addToast(
        data.is_spam
          ? `🛡️ Spam tespit edildi — %${(data.spam_probability * 100).toFixed(1)} olasılık`
          : `✅ Güvenli mesaj — %${((1 - data.spam_probability) * 100).toFixed(1)} güvenilirlik`,
        data.is_spam ? "warning" : "success"
      );
    } catch (err) {
      addToast(
        err.message || "Bir hata oluştu. Lütfen tekrar deneyin.",
        "error",
        5000
      );
    } finally {
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sonucu temizle
  const handleClear = useCallback(() => {
    setResult(null);
  }, []);

  // Geçmişi temizle
  const handleClearHistory = useCallback(() => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
    addToast("Analiz geçmişi temizlendi.", "info");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // API durumu
  const isApiOnline = health?.status === "healthy";
  const isModelLoaded = health?.model_loaded === true;

  return (
    <div className="app">
      {/* Arka Plan Dekorasyon */}
      <div className="bg-decoration">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      {/* Toast Notifications */}
      <Toast toasts={toasts} onRemove={removeToast} />

      <div className="container">
        {/* Header */}
        <header className="app-header">
          <div className="logo-section">
            <div className="logo-icon">
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="m9 12 2 2 4-4" />
              </svg>
            </div>
            <div>
              <h1 className="app-title">Spam Detector</h1>
              <p className="app-subtitle">
                AI destekli gerçek zamanlı mesaj sınıflandırma
              </p>
            </div>
          </div>

          {/* API Durum Göstergesi */}
          <div className="status-indicator">
            {isHealthChecking ? (
              <span className="status-badge status-checking">
                <span className="status-dot pulsing" />
                Kontrol ediliyor...
              </span>
            ) : isApiOnline && isModelLoaded ? (
              <span className="status-badge status-online">
                <span className="status-dot" />
                API Çevrimiçi
                {health.model_accuracy && (
                  <span className="status-accuracy">
                    · %{(health.model_accuracy * 100).toFixed(1)} doğruluk
                  </span>
                )}
              </span>
            ) : (
              <span className="status-badge status-offline">
                <span className="status-dot" />
                API Çevrimdışı
              </span>
            )}
          </div>
        </header>

        {/* Ana İçerik */}
        <main className="main-content">
          {/* API Çevrimdışı Uyarısı */}
          {!isHealthChecking && !isApiOnline && (
            <div className="alert alert-warning">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              <div>
                <strong>Backend bağlantısı kurulamadı.</strong>
                <p>
                  Lütfen API sunucusunun çalıştığından emin olun:
                  <code>uvicorn backend.main:app --reload --port 8000</code>
                </p>
              </div>
            </div>
          )}

          {/* Form */}
          <section className="form-section">
            <MessageForm onSubmit={handleClassify} isLoading={isLoading} />
          </section>

          {/* Loading Skeleton */}
          {isLoading && <LoadingSkeleton />}

          {/* Sonuç */}
          {!isLoading && result && (
            <ResultCard result={result} onClear={handleClear} />
          )}

          {/* Analiz Geçmişi */}
          <HistoryPanel
            history={history}
            onClearHistory={handleClearHistory}
          />
        </main>

        {/* Footer */}
        <footer className="app-footer">
          <p>
            Spam Message Collector — LinearSVC + TF-IDF ile desteklenmektedir
          </p>
        </footer>
      </div>
    </div>
  );
}
