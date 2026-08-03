// src/components/ResultCard.jsx
/**
 * ResultCard — Gelişmiş Sınıflandırma Sonuç Bileşeni
 * =====================================================
 * API'den dönen sınıflandırma sonucunu premium UI ile gösterir.
 *
 * Gün 6 İyileştirmeleri:
 *   - Dairesel SVG gauge (animasyonlu olasılık göstergesi)
 *   - Radial glow efektleri (spam: kırmızı, ham: yeşil)
 *   - Tarih/saat bilgisi
 *   - Gelişmiş badge animasyonları
 *   - Güvenilirlik seviyesi bar gösterimi
 */

import { useEffect, useState } from "react";

export default function ResultCard({ result, onClear }) {
  if (!result) return null;

  const { text, label, is_spam, spam_probability, confidence_level, timestamp } = result;
  const probabilityPercent = (spam_probability * 100).toFixed(1);

  // Güvenilirlik seviyesi renk eşleştirmesi
  const confidenceConfig = {
    Yüksek: { color: "var(--confidence-high)", icon: "🟢", width: "100%" },
    Orta: { color: "var(--confidence-medium)", icon: "🟡", width: "60%" },
    Düşük: { color: "var(--confidence-low)", icon: "🔴", width: "30%" },
  };

  const conf = confidenceConfig[confidence_level] || confidenceConfig["Düşük"];

  return (
    <div className={`result-card ${is_spam ? "result-spam" : "result-ham"}`}>
      {/* Başlık Badge */}
      <div className="result-header">
        <div className={`result-badge ${is_spam ? "badge-spam" : "badge-ham"}`}>
          <span className="badge-icon">{is_spam ? "🛡️" : "✅"}</span>
          <span className="badge-text">
            {is_spam ? "SPAM TESPİT EDİLDİ" : "GÜVENLİ MESAJ"}
          </span>
        </div>
        <div className="result-header-right">
          {timestamp && (
            <span className="result-timestamp">
              {new Date(timestamp).toLocaleTimeString("tr-TR", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
          )}
          <button
            type="button"
            className="clear-btn"
            onClick={onClear}
            aria-label="Sonucu temizle"
            title="Temizle"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>

      {/* Orijinal Mesaj */}
      <div className="result-message">
        <p className="result-message-label">Analiz Edilen Mesaj</p>
        <blockquote className="result-message-text">&ldquo;{text}&rdquo;</blockquote>
      </div>

      {/* Dairesel Gauge + Metrikler */}
      <div className="result-gauge-section">
        {/* SVG Circular Gauge */}
        <div className="gauge-container">
          <CircularGauge
            percentage={parseFloat(probabilityPercent)}
            isSpam={is_spam}
          />
          <div className="gauge-label">
            <span className={`gauge-value ${is_spam ? "value-spam" : "value-ham"}`}>
              %{probabilityPercent}
            </span>
            <span className="gauge-sublabel">Spam Olasılığı</span>
          </div>
        </div>

        {/* Metrikler Grid */}
        <div className="result-metrics-vertical">
          {/* Sınıflandırma */}
          <div className="metric-card-v">
            <span className="metric-title">Sınıflandırma</span>
            <span className={`metric-label-value ${is_spam ? "label-spam" : "label-ham"}`}>
              {label.toUpperCase()}
            </span>
          </div>

          {/* Güvenilirlik */}
          <div className="metric-card-v">
            <span className="metric-title">Güvenilirlik</span>
            <div className="confidence-display">
              <span className="metric-confidence" style={{ color: conf.color }}>
                {conf.icon} {confidence_level}
              </span>
              <div className="confidence-bar-track">
                <div
                  className="confidence-bar-fill"
                  style={{ width: conf.width, background: conf.color }}
                />
              </div>
            </div>
          </div>

          {/* Karar */}
          <div className="metric-card-v">
            <span className="metric-title">Karar</span>
            <span className={`metric-decision ${is_spam ? "decision-spam" : "decision-ham"}`}>
              {is_spam ? "⛔ Engelle" : "📩 Teslim Et"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}


/**
 * CircularGauge — SVG Dairesel Olasılık Göstergesi
 */
function CircularGauge({ percentage, isSpam }) {
  const [animatedPercent, setAnimatedPercent] = useState(0);

  useEffect(() => {
    // Animasyonlu dolum
    const timer = setTimeout(() => setAnimatedPercent(percentage), 100);
    return () => clearTimeout(timer);
  }, [percentage]);

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (animatedPercent / 100) * circumference;

  return (
    <svg className="gauge-svg" viewBox="0 0 128 128" width="128" height="128">
      {/* Arka plan çemberi */}
      <circle
        cx="64"
        cy="64"
        r={radius}
        fill="none"
        stroke="rgba(255,255,255,0.06)"
        strokeWidth="8"
      />
      {/* Dolum çemberi */}
      <circle
        cx="64"
        cy="64"
        r={radius}
        fill="none"
        stroke={isSpam ? "url(#gauge-gradient-spam)" : "url(#gauge-gradient-ham)"}
        strokeWidth="8"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        transform="rotate(-90 64 64)"
        className="gauge-circle-fill"
      />
      {/* Gradient tanımları */}
      <defs>
        <linearGradient id="gauge-gradient-spam" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
        <linearGradient id="gauge-gradient-ham" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#22c55e" />
        </linearGradient>
      </defs>
    </svg>
  );
}
