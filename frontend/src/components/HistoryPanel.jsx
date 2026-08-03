// src/components/HistoryPanel.jsx
/**
 * HistoryPanel — Analiz Geçmişi Bileşeni
 * =========================================
 * Önceki sınıflandırma sonuçlarını kompakt kartlar halinde gösterir.
 * LocalStorage ile persist edilir.
 *
 * Özellikler:
 *   - Kompakt kart tasarımı (mesaj önizleme + badge + olasılık)
 *   - Tarih/saat gösterimi
 *   - Tümünü temizle butonu
 *   - Aç/kapa toggle animasyonu
 *   - Maksimum 20 kayıt tutma
 */

import { useState } from "react";

export default function HistoryPanel({ history, onClearHistory }) {
  const [isOpen, setIsOpen] = useState(false);

  if (history.length === 0) return null;

  const spamCount = history.filter((h) => h.is_spam).length;
  const hamCount = history.length - spamCount;

  return (
    <div className="history-panel">
      {/* Toggle Header */}
      <button
        type="button"
        className="history-toggle"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="history-toggle-left">
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
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span className="history-toggle-title">
            Analiz Geçmişi
          </span>
          <span className="history-count-badge">{history.length}</span>
        </div>
        <div className="history-toggle-right">
          <span className="history-stats">
            <span className="history-stat-spam">🚨 {spamCount}</span>
            <span className="history-stat-ham">✅ {hamCount}</span>
          </span>
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`history-chevron ${isOpen ? "chevron-open" : ""}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </button>

      {/* Panel Content */}
      {isOpen && (
        <div className="history-content">
          <div className="history-list">
            {history.map((item, index) => (
              <div
                key={item.timestamp || index}
                className={`history-item ${item.is_spam ? "history-item-spam" : "history-item-ham"}`}
              >
                <div className="history-item-top">
                  <span className={`history-badge ${item.is_spam ? "h-badge-spam" : "h-badge-ham"}`}>
                    {item.is_spam ? "SPAM" : "HAM"}
                  </span>
                  <span className="history-probability">
                    %{(item.spam_probability * 100).toFixed(1)}
                  </span>
                  <span className="history-time">
                    {formatTime(item.timestamp)}
                  </span>
                </div>
                <p className="history-item-text">{item.text}</p>
              </div>
            ))}
          </div>

          {/* Tümünü Temizle */}
          <button
            type="button"
            className="history-clear-btn"
            onClick={onClearHistory}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
            Geçmişi Temizle
          </button>
        </div>
      )}
    </div>
  );
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "Az önce";
  if (diffMins < 60) return `${diffMins} dk önce`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours} sa önce`;

  return date.toLocaleDateString("tr-TR", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}
