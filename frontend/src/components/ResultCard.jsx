// src/components/ResultCard.jsx
/**
 * ResultCard — Sınıflandırma Sonuç Bileşeni
 * ============================================
 * API'den dönen sınıflandırma sonucunu görsel olarak gösterir.
 *
 * Özellikler:
 *   - Spam/Ham badge animasyonlu gösterim
 *   - Olasılık çubuğu (progress bar)
 *   - Güvenilirlik seviyesi göstergesi
 *   - Açılış animasyonu (slide-in + fade)
 */

export default function ResultCard({ result, onClear }) {
  if (!result) return null;

  const { text, label, is_spam, spam_probability, confidence_level } = result;
  const probabilityPercent = (spam_probability * 100).toFixed(1);

  // Güvenilirlik seviyesi renk eşleştirmesi
  const confidenceConfig = {
    Yüksek: { color: "var(--confidence-high)", icon: "🟢" },
    Orta: { color: "var(--confidence-medium)", icon: "🟡" },
    Düşük: { color: "var(--confidence-low)", icon: "🔴" },
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
        <button
          type="button"
          className="clear-btn"
          onClick={onClear}
          aria-label="Sonucu temizle"
          title="Temizle"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Orijinal Mesaj */}
      <div className="result-message">
        <p className="result-message-label">Analiz Edilen Mesaj</p>
        <blockquote className="result-message-text">&ldquo;{text}&rdquo;</blockquote>
      </div>

      {/* Metrikler */}
      <div className="result-metrics">
        {/* Spam Olasılığı */}
        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Spam Olasılığı</span>
            <span
              className={`metric-value ${is_spam ? "value-spam" : "value-ham"}`}
            >
              %{probabilityPercent}
            </span>
          </div>
          <div className="progress-bar-track">
            <div
              className={`progress-bar-fill ${is_spam ? "fill-spam" : "fill-ham"}`}
              style={{ width: `${probabilityPercent}%` }}
            />
          </div>
        </div>

        {/* Sınıflandırma & Güvenilirlik */}
        <div className="metric-row">
          <div className="metric-card metric-small">
            <span className="metric-title">Sınıflandırma</span>
            <span
              className={`metric-label-value ${is_spam ? "label-spam" : "label-ham"}`}
            >
              {label.toUpperCase()}
            </span>
          </div>
          <div className="metric-card metric-small">
            <span className="metric-title">Güvenilirlik</span>
            <span className="metric-confidence" style={{ color: conf.color }}>
              {conf.icon} {confidence_level}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
