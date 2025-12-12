import { useState, useEffect } from 'react';

export interface Settings {
  name: string;
  info: string;
  language: string;
  maxQueries: number;
}

const defaultSettings: Settings = {
  name: '',
  info: '',
  language: 'en',
  maxQueries: 3,
};

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(defaultSettings);

  useEffect(() => {
    const stored = localStorage.getItem('settings');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSettings({ ...defaultSettings, ...parsed });
      } catch {
        setSettings(defaultSettings);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('settings', JSON.stringify(settings));
  }, [settings]);

  return { settings, setSettings } as const;
}
