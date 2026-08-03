// src/components/LoadingSkeleton.jsx
/**
 * LoadingSkeleton — Yükleme Animasyonu Bileşeni
 * ================================================
 * ResultCard şeklinde shimmer/skeleton efekti.
 * API yanıtı beklenirken gösterilir.
 *
 * Özellikler:
 *   - ResultCard layout'unu taklit eden iskelet yapı
 *   - Shimmer (parlama) animasyonu
 *   - Pulse efekti ile canlı hissiyat
 */

export default function LoadingSkeleton() {
  return (
    <div className="skeleton-card" aria-busy="true" aria-label="Analiz ediliyor...">
      {/* Badge Skeleton */}
      <div className="skeleton-header">
        <div className="skeleton-badge skeleton-shimmer" />
      </div>

      {/* Mesaj Skeleton */}
      <div className="skeleton-message">
        <div className="skeleton-label skeleton-shimmer" />
        <div className="skeleton-text skeleton-shimmer" />
        <div className="skeleton-text skeleton-text-short skeleton-shimmer" />
      </div>

      {/* Gauge Skeleton */}
      <div className="skeleton-gauge-area">
        <div className="skeleton-gauge skeleton-shimmer" />
      </div>

      {/* Metrics Skeleton */}
      <div className="skeleton-metrics">
        <div className="skeleton-metric skeleton-shimmer" />
        <div className="skeleton-metric skeleton-shimmer" />
      </div>

      {/* Analyzing Pulse Text */}
      <div className="skeleton-analyzing">
        <span className="skeleton-dot" />
        <span>AI modeli mesajınızı analiz ediyor...</span>
      </div>
    </div>
  );
}
