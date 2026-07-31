// src/components/MessageForm.jsx
/**
 * MessageForm — Mesaj Giriş Bileşeni
 * ====================================
 * Kullanıcının spam kontrolü yapmak istediği mesajı girebileceği form.
 *
 * Özellikler:
 *   - Otomatik yeniden boyutlanan textarea
 *   - Karakter sayacı (max 5000)
 *   - Loading spinner animasyonu
 *   - Klavye kısayolu (Ctrl+Enter ile gönder)
 *   - Örnek mesajlarla hızlı test
 */

import { useState, useRef, useEffect } from "react";

const EXAMPLE_MESSAGES = [
  {
    label: "🚨 Spam Örneği",
    text: "WINNER!! You've been selected for a $1,000 prize! Call 09061234567 NOW to claim. Reply STOP to opt out.",
  },
  {
    label: "✅ Güvenli Örnek",
    text: "Hey, are we still meeting for lunch tomorrow at 12:30?",
  },
  {
    label: "🚨 Phishing Örneği",
    text: "URGENT: Your bank account has been compromised! Click here immediately to verify your identity and prevent unauthorized access.",
  },
];

const MAX_LENGTH = 5000;

export default function MessageForm({ onSubmit, isLoading }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  // Textarea otomatik yükseklik ayarı
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 300)}px`;
    }
  }, [text]);

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed);
  }

  function handleKeyDown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      handleSubmit(e);
    }
  }

  function handleExampleClick(exampleText) {
    setText(exampleText);
    textareaRef.current?.focus();
  }

  const charCount = text.length;
  const isOverLimit = charCount > MAX_LENGTH;
  const isEmpty = text.trim().length === 0;

  return (
    <form className="message-form" onSubmit={handleSubmit}>
      {/* Textarea */}
      <div className="textarea-wrapper">
        <textarea
          ref={textareaRef}
          id="message-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Spam kontrolü yapmak istediğiniz mesajı buraya yazın..."
          rows={4}
          maxLength={MAX_LENGTH}
          disabled={isLoading}
          aria-label="Mesaj girişi"
          autoFocus
        />
        <div className={`char-counter ${isOverLimit ? "over-limit" : ""}`}>
          <span>{charCount.toLocaleString("tr-TR")}</span>
          <span className="separator">/</span>
          <span>{MAX_LENGTH.toLocaleString("tr-TR")}</span>
        </div>
      </div>

      {/* Submit Butonu */}
      <button
        type="submit"
        id="submit-button"
        className="submit-btn"
        disabled={isEmpty || isOverLimit || isLoading}
      >
        {isLoading ? (
          <span className="btn-loading">
            <span className="spinner" />
            Analiz ediliyor...
          </span>
        ) : (
          <span className="btn-content">
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
              <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
              <path d="m9 12 2 2 4-4" />
            </svg>
            Mesajı Analiz Et
          </span>
        )}
      </button>

      <p className="shortcut-hint">
        <kbd>Ctrl</kbd> + <kbd>Enter</kbd> ile hızlıca gönderin
      </p>

      {/* Örnek Mesajlar */}
      <div className="examples-section">
        <p className="examples-title">Hızlı Test:</p>
        <div className="examples-grid">
          {EXAMPLE_MESSAGES.map((example, index) => (
            <button
              key={index}
              type="button"
              className="example-btn"
              onClick={() => handleExampleClick(example.text)}
              disabled={isLoading}
            >
              <span className="example-label">{example.label}</span>
              <span className="example-text">{example.text}</span>
            </button>
          ))}
        </div>
      </div>
    </form>
  );
}
