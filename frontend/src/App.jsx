// src/App.jsx
/**
 * App — Ana Uygulama Bileşeni
 * =============================
 * Spam Message Classifier frontend uygulamasının giriş noktası.
 *
 * Sorumluluklar:
 *   - Global state yönetimi (result, loading, error, health)
 *   - API çağrı orkestratoru
 *   - Bileşen düzeni (layout)
 *   - Başlangıçta health check ile API durumu kontrolü
 */

import { useState, useEffect, useCallback } from "react";
import MessageForm from "./components/MessageForm";
import ResultCard from "./components/ResultCard";
import { classifyMessage, checkHealth } from "./services/api";

export default function App() {
  // State
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [isHealthChecking, setIsHealthChecking] = useState(true);

  // Startup health check
  useEffect(() => {
    async function fetchHealth() {
      try {
        const data = await checkHealth();
        setHealth(data);
      } catch {
        setHealth(null);
      } finally {
        setIsHealthChecking(false);
      }
    }
    fetchHealth();
  }, []);

  // Mesaj sınıflandırma handler
  const handleClassify = useCallback(async (text) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await classifyMessage(text);
      setResult(data);
    } catch (err) {
      setError(err.message || "Bir hata oluştu. Lütfen tekrar deneyin.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Sonucu temizle
  const handleClear = useCallback(() => {
    setResult(null);
    setError(null);
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
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
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

          {/* Hata Gösterimi */}
          {error && (
            <div className="alert alert-error">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <span>{error}</span>
              <button
                type="button"
                className="alert-dismiss"
                onClick={() => setError(null)}
              >
                ✕
              </button>
            </div>
          )}

          {/* Sonuç */}
          {result && <ResultCard result={result} onClear={handleClear} />}
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
