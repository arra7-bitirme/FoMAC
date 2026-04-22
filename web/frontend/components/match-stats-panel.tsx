"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";

interface MatchMeta {
  session_id: string;
  fps: number;
  total_seconds: number;
  team_names?: Record<string, string>;
  match_info?: { date?: string; teams?: string; competition?: string };
}

interface MatchEvent {
  frame: number;
  second: number;
  minute: number;
  label: string;
  confidence: number;
  model: string;
  team_id: number;
}

interface PlayerRecord {
  track_ids: number[];
  jersey_number: number | null;
  team_id: number;
  total_distance_m: number;
  avg_speed_kmh: number;
  player_name?: string | null;
}

interface MatchStats {
  meta: MatchMeta;
  events: MatchEvent[];
  possession: {
    by_second: number[];
    team_percentages: Record<string, number>;
  };
  team_event_counts: Record<string, Record<string, number>>;
  ball_position_by_second: ([number, number] | null)[];
  players: Record<string, PlayerRecord>;
}

const TEAM_COLORS: Record<string, { bg: string; text: string }> = {
  "0": { bg: "bg-blue-500", text: "text-blue-500" },
  "1": { bg: "bg-red-500", text: "text-red-500" },
};

function fmt(s: number) {
  return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}`;
}

export function MatchStatsPanel({ matchStatsUrl, currentTime }: { matchStatsUrl: string; currentTime: number }) {
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!matchStatsUrl) return;
    fetch(matchStatsUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setStats)
      .catch((e) => setError(e.message));
  }, [matchStatsUrl]);

  if (error) return (
    <Card className="p-4 text-sm text-destructive">Maç istatistikleri yüklenemedi: {error}</Card>
  );
  if (!stats) return (
    <Card className="p-4 text-sm text-muted-foreground">Maç istatistikleri yükleniyor...</Card>
  );

  const pct = stats.possession.team_percentages;
  const t0pct = pct["0"] ?? 0;
  const t1pct = pct["1"] ?? 0;

  const teamLabel = (tid: string) =>
    stats.meta.team_names?.[tid] ?? `Takım ${tid}`;

  const currentSec = Math.floor(currentTime);
  const windowEvents = stats.events.filter(
    (e) => e.second >= currentSec - 30 && e.second <= currentSec + 30
  );

  const players = Object.entries(stats.players)
    .filter(([, p]) => p.team_id !== 2)
    .sort(([, a], [, b]) => b.total_distance_m - a.total_distance_m)
    .slice(0, 10);

  const topEventsByTeam = (teamId: string) => {
    const counts = stats.team_event_counts[teamId] || {};
    return Object.entries(counts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
  };

  return (
    <div className="space-y-4 mt-4">
      {/* Possession Bar */}
      <Card className="p-4">
        <h3 className="font-semibold text-sm mb-3">Top Hakimiyeti</h3>
        <div className="flex items-center gap-2 text-sm mb-2">
          <span className="text-blue-500 font-medium w-16 text-right">{t0pct.toFixed(1)}%</span>
          <div className="flex-1 h-4 rounded-full overflow-hidden bg-muted flex">
            <div className="bg-blue-500 h-full transition-all" style={{ width: `${t0pct}%` }} />
            <div className="bg-red-500 h-full transition-all" style={{ width: `${t1pct}%` }} />
          </div>
          <span className="text-red-500 font-medium w-16">{t1pct.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{teamLabel("0")}</span>
          <span>{teamLabel("1")}</span>
        </div>
      </Card>

      {/* Team Event Counts — only shown when data is available */}
      {(["0", "1"] as const).some((tid) => topEventsByTeam(tid).length > 0) && (
        <div className="grid grid-cols-2 gap-3">
          {(["0", "1"] as const).map((tid) => (
            <Card key={tid} className="p-3">
              <h4 className={`text-xs font-semibold mb-2 ${TEAM_COLORS[tid].text}`}>
                {teamLabel(tid)}
              </h4>
              <div className="space-y-1">
                {topEventsByTeam(tid).map(([label, count]) => (
                  <div key={label} className="flex justify-between text-xs">
                    <span className="text-muted-foreground truncate">{label}</span>
                    <span className="font-medium ml-2">{count}</span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Events near current time */}
      <Card className="p-4">
        <h3 className="font-semibold text-sm mb-3">
          Yakın Olaylar ({fmt(Math.max(0, currentSec - 30))} – {fmt(currentSec + 30)})
        </h3>
        <div className="space-y-1 max-h-40 overflow-y-auto">
          {windowEvents.length === 0 && (
            <div className="text-xs text-muted-foreground">Bu aralıkta olay yok</div>
          )}
          {windowEvents.map((e, i) => (
            <div
              key={i}
              className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${
                Math.abs(e.second - currentSec) <= 2 ? "bg-muted" : ""
              }`}
            >
              <span className="text-muted-foreground w-10 shrink-0">{fmt(e.second)}</span>
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  e.team_id === 0 ? "bg-blue-500" : e.team_id === 1 ? "bg-red-500" : "bg-muted-foreground"
                }`}
              />
              <span className="font-medium">{e.label}</span>
              <span className="text-muted-foreground ml-auto">{(e.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Player Stats */}
      <Card className="p-4">
        <h3 className="font-semibold text-sm mb-3">Oyuncu İstatistikleri (Top 10 Mesafe)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-2 text-muted-foreground">Takım</th>
                <th className="text-left py-1 pr-2 text-muted-foreground">Oyuncu</th>
                <th className="text-right py-1 pr-2 text-muted-foreground">Mesafe (m)</th>
                <th className="text-right py-1 text-muted-foreground">Ort Hız (km/h)</th>
              </tr>
            </thead>
            <tbody>
              {players.map(([key, p]) => (
                <tr key={key} className="border-b border-muted/40">
                  <td className="py-1 pr-2">
                    <span
                      className={`inline-block w-2 h-2 rounded-full ${
                        p.team_id === 0 ? "bg-blue-500" : "bg-red-500"
                      }`}
                      title={teamLabel(String(p.team_id))}
                    />
                  </td>
                  <td className="py-1 pr-2 font-medium">
                    {p.player_name
                      ? p.player_name
                      : p.jersey_number !== null
                      ? `#${p.jersey_number}`
                      : "—"}
                  </td>
                  <td className="py-1 pr-2 text-right">{p.total_distance_m.toFixed(0)}</td>
                  <td className="py-1 text-right">{p.avg_speed_kmh.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
