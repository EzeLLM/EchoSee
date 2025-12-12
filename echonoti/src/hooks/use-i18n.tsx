"use client";

import { createContext, useContext, useMemo } from "react";
import { useSettingsContext } from "@/components/SettingsProvider";
import { messages, Locale } from "@/i18n/messages";

const I18nContext = createContext<(key: string, values?: Record<string, any>) => string>((key) => key);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const { settings } = useSettingsContext();

  const t = useMemo(() => {
    const current = messages[settings.language as Locale] || messages.en;
    return (key: string, values?: Record<string, any>) => {
      let text = current[key as keyof typeof current] as string | undefined;
      if (!text) return key;
      if (values) {
        for (const k of Object.keys(values)) {
          text = text.replace(new RegExp(`{${k}}`, "g"), String(values[k]));
        }
      }
      return text;
    };
  }, [settings.language]);

  return <I18nContext.Provider value={t}>{children}</I18nContext.Provider>;
}

export function useT() {
  return useContext(I18nContext);
}
