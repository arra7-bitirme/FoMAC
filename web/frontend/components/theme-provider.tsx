"use client";

import { useEffect } from "react";
import { useAppStore } from "@/lib/store";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useAppStore((state) => state.theme);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
  }, [theme]);

  useEffect(() => {
    const storedTheme = localStorage.getItem("fomac-theme") as "light" | "dark" | null;
    if (storedTheme) {
      useAppStore.getState().setTheme(storedTheme);
    } else {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      useAppStore.getState().setTheme(prefersDark ? "dark" : "light");
    }
  }, []);

  return <>{children}</>;
}
