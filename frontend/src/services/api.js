// src/services/api.js
/**
 * Spam Classifier API — Axios Servis Katmanı
 * ============================================
 * Backend FastAPI (/api/v1) ile iletişimi sağlayan merkezi HTTP istemcisi.
 *
 * Özellikler:
 *   - Merkezi base URL ve timeout yapılandırması
 *   - Request/Response interceptor'lar ile loglama
 *   - Yapılandırılmış hata yönetimi (network, validation, server)
 *   - Tekli ve toplu sınıflandırma endpoint'leri
 *   - Health check endpoint'i
 */

import axios from "axios";

// ---------------------------------------------------------------------------
// Axios Instance — Merkezi Konfigürasyon
// ---------------------------------------------------------------------------
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000, // 15 saniye
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

// ---------------------------------------------------------------------------
// Request Interceptor — Loglama
// ---------------------------------------------------------------------------
apiClient.interceptors.request.use(
  (config) => {
    if (import.meta.env.DEV) {
      console.log(
        `[API] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`,
        config.data || ""
      );
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ---------------------------------------------------------------------------
// Response Interceptor — Hata Normalizasyonu
// ---------------------------------------------------------------------------
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Yapılandırılmış hata nesnesi oluştur
    const apiError = {
      message: "Beklenmeyen bir hata oluştu.",
      status: null,
      detail: null,
    };

    if (error.response) {
      // Sunucudan gelen hata yanıtı (4xx, 5xx)
      apiError.status = error.response.status;
      const data = error.response.data;

      if (apiError.status === 422) {
        // Pydantic validation hatası
        const validationErrors = data?.detail || [];
        const messages = validationErrors.map(
          (err) => err.msg || "Geçersiz veri"
        );
        apiError.message = `Doğrulama hatası: ${messages.join(", ")}`;
        apiError.detail = validationErrors;
      } else if (apiError.status === 503) {
        apiError.message =
          "Model henüz yüklenmedi. Lütfen birkaç saniye sonra tekrar deneyin.";
      } else if (apiError.status >= 500) {
        apiError.message =
          data?.detail?.message || "Sunucu hatası. Lütfen tekrar deneyin.";
      } else {
        apiError.message =
          data?.detail?.message || data?.detail || "İstek başarısız oldu.";
      }
    } else if (error.request) {
      // Sunucuya ulaşılamadı
      apiError.message =
        "Sunucuya bağlanılamıyor. Backend'in çalıştığından emin olun.";
    } else {
      apiError.message = error.message || "İstek oluşturulurken hata oluştu.";
    }

    return Promise.reject(apiError);
  }
);

// ---------------------------------------------------------------------------
// API Fonksiyonları
// ---------------------------------------------------------------------------

/**
 * Tek bir mesajı spam/ham olarak sınıflandırır.
 *
 * @param {string} text — Sınıflandırılacak metin mesajı
 * @returns {Promise<Object>} — { text, label, is_spam, spam_probability, confidence_level }
 */
export async function classifyMessage(text) {
  const response = await apiClient.post("/api/v1/classify", { text });
  return response.data;
}

/**
 * Birden fazla mesajı toplu olarak sınıflandırır.
 *
 * @param {string[]} messages — Sınıflandırılacak metin dizisi (maks. 50)
 * @returns {Promise<Object>} — { results, total, spam_count, ham_count }
 */
export async function classifyBatch(messages) {
  const response = await apiClient.post("/api/v1/classify/batch", { messages });
  return response.data;
}

/**
 * API ve model sağlık durumunu kontrol eder.
 *
 * @returns {Promise<Object>} — { status, model_loaded, model_name, model_accuracy }
 */
export async function checkHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}

export default apiClient;
