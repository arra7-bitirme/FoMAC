"use client";

import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Moon, Sun, Bug } from "lucide-react";

export function Navigation() {
  const theme = useAppStore((state) => state.theme);
  const debugMode = useAppStore((state) => state.debugMode);
  const toggleTheme = useAppStore((state) => state.toggleTheme);
  const setDebugMode = useAppStore((state) => state.setDebugMode);

  return (
    <nav className="border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-12 w-12 rounded-lg bg-brand flex items-center justify-center shadow-lg ring-2 ring-brand/20">
              <span className="text-white font-bold text-xl">F</span>
            </div>
            <div className="flex flex-col">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-brand to-brand-light bg-clip-text text-transparent">
                FOMAC
              </h1>
              <span className="text-xs text-muted-foreground hidden sm:inline">
                Football Match AI Commentary
              </span>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Label htmlFor="debug-mode" className="text-sm font-medium">
                Debug Mode
              </Label>
              <Switch
                id="debug-mode"
                checked={debugMode}
                onCheckedChange={setDebugMode}
              />
              {debugMode && (
                <Bug className="h-4 w-4 text-brand ml-1" />
              )}
            </div>

            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <Sun className="h-5 w-5" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}
