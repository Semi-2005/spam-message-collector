// src/hooks/useToast.js
/**
 * useToast — Toast Bildirim Yönetimi Custom Hook
 * =================================================
 * Toast state yönetimi, ekleme/kaldırma ve otomatik kapanma mantığı.
 *
 * Kullanım:
 *   const { toasts, addToast, removeToast } = useToast();
 *   addToast("Başarılı!", "success");
 */

import { useState, useCallback, useRef } from "react";

let toastIdCounter = 0;

export default function useToast(defaultDuration = 3500) {
  const [toasts, setToasts] = useState([]);
  const timersRef = useRef(new Map());

  const removeToast = useCallback((id) => {
    // Önce çıkış animasyonu için "exiting" state
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, exiting: true } : t))
    );

    // Animasyon bitince DOM'dan kaldır
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
      const timer = timersRef.current.get(id);
      if (timer) {
        clearTimeout(timer);
        timersRef.current.delete(id);
      }
    }, 300);
  }, []);

  const addToast = useCallback(
    (message, type = "info", duration = defaultDuration) => {
      const id = ++toastIdCounter;

      const toast = {
        id,
        message,
        type, // "success" | "error" | "warning" | "info"
        exiting: false,
        createdAt: Date.now(),
      };

      setToasts((prev) => [...prev, toast]);

      // Otomatik kapanma
      if (duration > 0) {
        const timer = setTimeout(() => removeToast(id), duration);
        timersRef.current.set(id, timer);
      }

      return id;
    },
    [defaultDuration, removeToast]
  );

  return { toasts, addToast, removeToast };
}
