"use client";

import React, { createContext, useContext } from "react";
import { useSettings, Settings } from "@/hooks/use-settings";

interface Ctx {
  settings: Settings;
  setSettings: (s: Settings) => void;
}

const Context = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const { settings, setSettings } = useSettings();

  React.useEffect(() => {
    document.documentElement.lang = settings.language;
  }, [settings.language]);

  return (
    <Context.Provider value={{ settings, setSettings }}>{children}</Context.Provider>
  );
}

export function useSettingsContext() {
  const ctx = useContext(Context);
  if (!ctx) throw new Error("useSettingsContext must be used within SettingsProvider");
  return ctx;
}
