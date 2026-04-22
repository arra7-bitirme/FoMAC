"""
analyze_match.py
----------------
Match statistics analyzer for the FoMAC football analysis pipeline.

Combines:
  - Calibration JSONL  → real-world (metres) player/ball positions, per frame
  - Tracking CSV       → team_id, jersey_number per player, per frame
  - SoccerNet_big JSON → primary action-spotting predictions
  - SoccerNetBall_challenge2 JSON → ball action-spotting predictions

Output: a single JSON file whose arrays are indexed by *second bucket*, so a
frontend can slice any [start_minute, end_minute] window with zero
reprocessing:
  - `possession.by_second[s:e]`              → possession % for window
  - `players["<id>"].distance_by_second[s:e]` → distance covered
  - filter `events` where `start_sec <= second < end_sec`

Usage
-----
python analyze_match.py \\
    --tracking   tracking_XXX_merged.csv \\
    --calibration calibration_frames_XXX.jsonl \\
    --big_model  tdeed_soccernetbig_XXX.json \\
    --ball_model tdeed_soccernetballchallenge2_XXX.json \\
    --fps        50.0 \\
    [--output    match_stats.json] \\
    [--session_id my_session]

Dependencies: pandas, numpy  (both standard in any ML/CV env)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALL_PROXIMITY_MAX_M = 5.0          # ignore possession if ball > 5 m from all players
REFEREE_TEAM_ID = 2                 # team_id value used for referees in CSV
ID_MATCH_FRAMES = 500               # number of calibration frames used for ID matching
ID_MATCH_MAX_DIST_PX = 80           # max bbox-centre pixel distance to accept a match
SPEED_OUTLIER_THRESHOLD_KMH = 50.0  # discard per-frame deltas implying > 50 km/h
BALL_SPEED_OUTLIER_THRESHOLD_KMH = 200.0  # discard ball position deltas implying impossibly fast movement

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FoMAC match statistics analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracking",    required=True, help="Path to tracking CSV (*_merged.csv)")
    p.add_argument("--calibration", required=True, help="Path to calibration JSONL")
    p.add_argument("--big_model",   required=True, help="Path to SoccerNet_big predictions JSON")
    p.add_argument("--ball_model",  required=True, help="Path to SoccerNetBall_challenge2 predictions JSON")
    p.add_argument("--fps",         required=True, type=float, help="Video FPS (from pipeline)")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <calibration_dir>/match_stats_<session_id>.json",
    )
    p.add_argument(
        "--session_id",
        default=None,
        help="Session identifier embedded in meta. Auto-derived from filenames if omitted.",
    )
    p.add_argument(
        "--roster",
        default=None,
        help="Path to roster JSON file with match_info and player names.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase 1 – Data loading helpers
# ---------------------------------------------------------------------------


def load_tracking_csv(path: str) -> pd.DataFrame:
    """Load tracking CSV; coerce types; return full DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # jersey_number may be empty / NaN when not yet detected
    if "jersey_number" in df.columns:
        df["jersey_number"] = pd.to_numeric(df["jersey_number"], errors="coerce")
    df["frame_id"]  = df["frame_id"].astype(int)
    df["track_id"]  = df["track_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    return df


def load_json_predictions(path: str, model_name: str) -> list[dict]:
    """Load a T-DEED predictions JSON and tag each entry with model_name."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    predictions = data.get("predictions", [])
    for p in predictions:
        p["model"] = model_name
    return predictions


# ---------------------------------------------------------------------------
# Phase 2 – Jersey & team mapping from tracking CSV
# ---------------------------------------------------------------------------


def build_track_maps(df: pd.DataFrame) -> tuple[dict[int, int | None], dict[int, int]]:
    """
    Returns:
        jersey_map   : {track_id → most-common jersey_number (int) or None}
        team_map     : {track_id → most-common team_id (0|1|2)}
    """
    jersey_map: dict[int, int | None] = {}
    team_map:   dict[int, int] = {}

    for track_id, grp in df.groupby("track_id"):
        # Jersey: drop NaN and -1; take mode
        valid_jerseys = grp["jersey_number"].dropna()
        valid_jerseys = valid_jerseys[valid_jerseys != -1]
        if len(valid_jerseys) > 0:
            jersey_map[track_id] = int(valid_jerseys.mode().iloc[0])
        else:
            jersey_map[track_id] = None

        # Team: drop -1; take mode
        valid_teams = grp["team_id"][grp["team_id"] != -1]
        if len(valid_teams) > 0:
            team_map[track_id] = int(valid_teams.mode().iloc[0])
        else:
            team_map[track_id] = -1

    return jersey_map, team_map


# ---------------------------------------------------------------------------
# Phase 3 – Calibration ↔ Tracking ID correspondence
# ---------------------------------------------------------------------------


def _bbox_centre(bbox_xyxy: list[float]) -> tuple[float, float]:
    return (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0


def _build_id_correspondence_with_offset(
    calib_path: str,
    tracking_lookup: dict[int, list[tuple[int, float, float]]],
    n_frames: int,
    frame_offset: int,
) -> dict[int, int]:
    """Inner helper — tries a fixed frame_offset (0 or 1) and returns the correspondence."""
    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    frames_read = 0
    with open(calib_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_idx = int(entry["frame_idx"])
            calib_players = entry.get("data", {}).get("players", [])
            if not calib_players:
                frames_read += 1
                if frames_read >= n_frames:
                    break
                continue
            t_players = tracking_lookup.get(frame_idx + frame_offset, [])
            if not t_players:
                frames_read += 1
                if frames_read >= n_frames:
                    break
                continue
            t_ids = np.array([tp[0] for tp in t_players], dtype=np.int32)
            t_cx  = np.array([tp[1] for tp in t_players], dtype=np.float64)
            t_cy  = np.array([tp[2] for tp in t_players], dtype=np.float64)
            for cp in calib_players:
                calib_id = int(cp["track_id"])
                ccx, ccy = _bbox_centre(cp["bbox_xyxy"])
                dists = np.sqrt((t_cx - ccx) ** 2 + (t_cy - ccy) ** 2)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < ID_MATCH_MAX_DIST_PX:
                    votes[calib_id][int(t_ids[min_idx])] += 1
            frames_read += 1
            if frames_read >= n_frames:
                break
    correspondence: dict[int, int] = {}
    for calib_id, vote_dict in votes.items():
        best = max(vote_dict, key=vote_dict.__getitem__)
        correspondence[calib_id] = best
    return correspondence


def build_id_correspondence(
    calib_path: str,
    tracking_df: pd.DataFrame,
    fps: float,
    n_frames: int = ID_MATCH_FRAMES,
) -> dict[int, int]:
    """
    Read the first `n_frames` calibration JSONL lines.
    For each frame, match each calibration player (calib_track_id) to the
    nearest tracking player bounding-box in pixel space.
    Majority-vote across all matched frames → {calib_track_id: tracking_track_id}.
    Tries both frame_offset=0 and frame_offset=1; keeps whichever yields more matches.
    """
    tracking_lookup: dict[int, list[tuple[int, float, float]]] = {}
    for row in tracking_df.itertuples(index=False):
        fid = int(row.frame_id)
        cx = (row.x1 + row.x2) / 2.0
        cy = (row.y1 + row.y2) / 2.0
        tracking_lookup.setdefault(fid, []).append((int(row.track_id), cx, cy))

    result_0 = _build_id_correspondence_with_offset(calib_path, tracking_lookup, n_frames, 0)
    result_1 = _build_id_correspondence_with_offset(calib_path, tracking_lookup, n_frames, 1)
    if len(result_0) >= len(result_1):
        print(f"       Frame offset=0 selected ({len(result_0)} IDs vs offset=1 {len(result_1)} IDs)")
        return result_0
    else:
        print(f"       Frame offset=1 selected ({len(result_1)} IDs vs offset=0 {len(result_0)} IDs)")
        return result_1


def build_possession_from_tracking(
    tracking_df: pd.DataFrame,
    fps: float,
    total_seconds: int,
) -> dict[int, dict[int, int]]:
    """
    Fallback possession computation using tracking CSV team_id column only.
    For each frame, finds the team with the most tracked players and
    increments that team's frame count for the corresponding second bucket.
    Not ball-proximity based, but allows team attribution when calibration
    correspondence is unavailable.
    """
    possession_frames: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    players_df = tracking_df[tracking_df["team_id"] != REFEREE_TEAM_ID]
    players_df = players_df[players_df["team_id"] != -1]
    for frame_id, grp in players_df.groupby("frame_id"):
        sec = int(int(frame_id) / fps)
        if sec >= total_seconds:
            continue
        team_counts = grp.groupby("team_id").size()
        if len(team_counts) > 0:
            dominant = int(team_counts.idxmax())
            possession_frames[sec][dominant] += 1
    return dict(possession_frames)


# ---------------------------------------------------------------------------
# Phase 4 – Single-pass calibration streaming
# ---------------------------------------------------------------------------


def _euclidean(xy1: list[float], xy2: list[float]) -> float:
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)


def stream_calibration(
    calib_path: str,
    fps: float,
    id_correspondence: dict[int, int],
    team_map: dict[int, int],
) -> tuple[
    dict[str, dict[int, float]],      # player_dist_by_sec[str(tid)][sec]
    dict[str, dict[int, float]],      # player_max_speed[str(tid)][sec]
    dict[int, dict[int, int]],        # possession_frames[sec][team_id]
    dict[int, list[float | None]],    # ball_pos_sum[sec] = [sum_x, sum_y, count]
    int,                              # total_frames seen
]:
    """
    Stream calibration JSONL once.
    Computes per-second-bucket statistics for:
      - player displacement (metres)
      - player max speed (km/h)
      - possession (proximity to ball)
      - ball world position (average per second)
    """
    # {calib_track_id → [x, y]} for previous frame displacement
    prev_world: dict[int, list[float]] = {}
    # {str(tracking_track_id) → {sec → metres}}
    player_dist: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    # {str(tracking_track_id) → {sec → max_speed_kmh}}
    player_max_speed: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    # {sec → {team_id → frame_count}}
    possession_frames: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # {sec → [sum_x, sum_y, count]}
    ball_sec_accum: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])

    total_frames = 0

    with open(calib_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not entry.get("calibration_ok", False):
                total_frames += 1
                continue

            frame_idx = int(entry["frame_idx"])
            sec = int(frame_idx / fps)
            data = entry.get("data", {})
            players = data.get("players", [])
            ball_data = data.get("ball")

            # ---- Per-player displacement & speed ---------------------------
            for cp in players:
                calib_id = int(cp["track_id"])
                tracking_id = id_correspondence.get(calib_id)
                if tracking_id is None:
                    continue

                team_id = team_map.get(tracking_id, -1)
                if team_id == REFEREE_TEAM_ID:
                    continue  # exclude referee from player stats

                wxy = cp["world_xy"]
                tid_key = str(tracking_id)

                if calib_id in prev_world:
                    delta_m = _euclidean(wxy, prev_world[calib_id])
                    # Sanity clamp: skip outlier frames (teleportation artefacts)
                    implied_speed_kmh = delta_m * fps * 3.6
                    if implied_speed_kmh < SPEED_OUTLIER_THRESHOLD_KMH:
                        player_dist[tid_key][sec] += delta_m
                        cur_max = player_max_speed[tid_key][sec]
                        if implied_speed_kmh > cur_max:
                            player_max_speed[tid_key][sec] = implied_speed_kmh

                prev_world[calib_id] = wxy

            # ---- Ball proximity possession ----------------------------------
            if ball_data and players:
                bxy = ball_data["world_xy"]
                # Accumulate ball position for average
                acc = ball_sec_accum[sec]
                acc[0] += bxy[0]
                acc[1] += bxy[1]
                acc[2] += 1

                best_dist = float("inf")
                best_team = -1
                for cp in players:
                    calib_id = int(cp["track_id"])
                    tracking_id = id_correspondence.get(calib_id)
                    if tracking_id is None:
                        continue
                    team_id = team_map.get(tracking_id, -1)
                    if team_id == REFEREE_TEAM_ID or team_id == -1:
                        continue
                    d = _euclidean(cp["world_xy"], bxy)
                    if d < best_dist:
                        best_dist = d
                        best_team = team_id

                if best_team != -1 and best_dist <= BALL_PROXIMITY_MAX_M:
                    possession_frames[sec][best_team] += 1

            total_frames += 1

    return player_dist, player_max_speed, possession_frames, ball_sec_accum, total_frames


# ---------------------------------------------------------------------------
# Phase 5 – Possession & ball position aggregation
# ---------------------------------------------------------------------------


def build_possession_array(
    possession_frames: dict[int, dict[int, int]],
    total_seconds: int,
) -> list[int]:
    """
    For each second bucket 0..total_seconds-1:
      dominant team = argmax of possession_frames[sec]; -1 if no data.
    """
    result: list[int] = []
    for sec in range(total_seconds):
        counts = possession_frames.get(sec)
        if not counts:
            result.append(-1)
        else:
            dominant = max(counts, key=counts.__getitem__)
            result.append(dominant)
    return result


def build_ball_position_array(
    ball_sec_accum: dict[int, list[float]],
    total_seconds: int,
) -> list[list[float] | None]:
    """Average X/Y per second bucket."""
    result: list[list[float] | None] = []
    for sec in range(total_seconds):
        acc = ball_sec_accum.get(sec)
        if acc and acc[2] > 0:
            result.append([round(acc[0] / acc[2], 3), round(acc[1] / acc[2], 3)])
        else:
            result.append(None)
    return result


def build_possession_percentages(
    possession_frames: dict[int, dict[int, int]],
    total_seconds: int,
) -> dict[str, float]:
    """
    Overall possession % for each team over the full video
    (excludes seconds where no ball was detected).
    """
    totals: dict[int, int] = defaultdict(int)
    for sec in range(total_seconds):
        counts = possession_frames.get(sec, {})
        for team_id, cnt in counts.items():
            totals[team_id] += cnt

    grand_total = sum(totals.values())
    if grand_total == 0:
        return {}
    return {
        str(tid): round(cnt / grand_total * 100, 2)
        for tid, cnt in sorted(totals.items())
    }


# ---------------------------------------------------------------------------
# Phase 6 – Event assembly + team assignment + team event counts
# ---------------------------------------------------------------------------

# Events that are team-attributable by the team in possession at event time
TEAM_ATTRIBUTABLE_LABELS = {
    # SoccerNet_big
    "Goal", "Shots on target", "Shots off target", "Clearance",
    "Throw-in", "Indirect free-kick", "Direct free-kick", "Corner",
    "Foul", "Kick-off",
    # SoccerNetBall_challenge2 (normalised to title case below)
    "Pass", "Drive", "Header", "High Pass", "Out", "Cross",
    "Throw In", "Shot", "Ball Player Block", "Player Successful Tackle",
    "Free Kick",
}


def assemble_events(
    big_predictions:  list[dict],
    ball_predictions: list[dict],
    fps: float,
    possession_by_second: list[int],
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """
    Merge both model predictions into one sorted list.
    Assign team_id from possession at the event's second bucket.
    Build team_event_counts.
    """
    all_preds = big_predictions + ball_predictions
    total_seconds = len(possession_by_second)

    team_event_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    events: list[dict] = []

    for pred in all_preds:
        frame = int(pred["frame"])
        second = round(frame / fps, 3)
        minute = round(second / 60.0, 4)
        sec_bucket = int(frame / fps)
        label = pred["label"]

        # Resolve team from possession
        if 0 <= sec_bucket < total_seconds:
            team_id = int(possession_by_second[sec_bucket])
        else:
            team_id = -1

        event = {
            "frame":      frame,
            "second":     second,
            "minute":     minute,
            "label":      label,
            "confidence": round(float(pred["confidence"]), 6),
            "model":      pred["model"],
            "team_id":    team_id,
        }
        events.append(event)

        if team_id != -1:
            team_event_counts[str(team_id)][label] += 1

    events.sort(key=lambda e: e["frame"])
    return events, {k: dict(v) for k, v in team_event_counts.items()}


# ---------------------------------------------------------------------------
# Roster helpers
# ---------------------------------------------------------------------------


def load_roster(path: str | None) -> dict:
    """Load roster JSON and return parsed dict, or {} on failure."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        warnings.warn(f"Could not load roster JSON {path!r}: {exc}", stacklevel=2)
        return {}


def build_roster_player_map(roster_data: dict) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    """
    Returns:
        player_name_map : {(team_id_0based, jersey_number) → player_name}
        team_name_map   : {team_id_0based → team_display_name}
    team_id_0based is 0 for first roster key, 1 for second, etc.
    """
    rosters = roster_data.get("rosters") or {}
    player_name_map: dict[tuple[int, int], str] = {}
    team_name_map: dict[int, str] = {}
    for idx, (team_key, players) in enumerate(rosters.items()):
        team_name_map[idx] = str(team_key).replace("_", " ").title()
        if not isinstance(players, list):
            continue
        for p in players:
            if not isinstance(p, dict):
                continue
            try:
                num = int(p["number"])
            except (KeyError, TypeError, ValueError):
                continue
            name = str(p.get("name") or "").strip()
            if name:
                player_name_map[(idx, num)] = name
    return player_name_map, team_name_map


# ---------------------------------------------------------------------------
# Phase 7 – Player record assembly
# ---------------------------------------------------------------------------


def build_player_records(
    player_dist:       dict[str, dict[int, float]],
    player_max_speed:  dict[str, dict[int, float]],
    jersey_map:        dict[int, int | None],
    team_map:          dict[int, int],
    total_seconds:     int,
    player_name_map:   dict[tuple[int, int], str] | None = None,
) -> dict[str, dict]:
    """
    Assemble player records keyed by (team_id, jersey_number).
    Multiple track_ids that share the same jersey+team are merged:
      - distances are summed per second bucket
      - max speeds are max'd per second bucket
    Tracks without a detected jersey remain as individual entries.
    Referees are excluded.
    """
    all_track_ids: set[int] = set()
    all_track_ids.update(jersey_map.keys())
    all_track_ids.update(team_map.keys())
    all_track_ids.update(int(k) for k in player_dist.keys())

    # group_key → accumulated group data
    groups: dict[str, dict] = {}

    for tid in sorted(all_track_ids):
        team_id = team_map.get(tid, -1)
        if team_id == REFEREE_TEAM_ID:
            continue

        jersey = jersey_map.get(tid)
        if jersey is not None:
            group_key = f"t{team_id}_j{jersey}"
        else:
            group_key = f"t{team_id}_unk{tid}"

        tid_key = str(tid)
        dist_by_sec = player_dist.get(tid_key, {})
        spd_by_sec  = player_max_speed.get(tid_key, {})

        if group_key not in groups:
            groups[group_key] = {
                "track_ids":     [],
                "team_id":       team_id,
                "jersey_number": jersey,
                "dist_acc":      defaultdict(float),
                "spd_acc":       defaultdict(float),
            }

        g = groups[group_key]
        g["track_ids"].append(tid)
        for s, v in dist_by_sec.items():
            g["dist_acc"][s] += v
        for s, v in spd_by_sec.items():
            if v > g["spd_acc"][s]:
                g["spd_acc"][s] = v

    records: dict[str, dict] = {}
    for gk, g in groups.items():
        dist_arr: list[float] = [
            round(g["dist_acc"].get(s, 0.0), 4) for s in range(total_seconds)
        ]
        max_spd_arr: list[float] = [
            round(g["spd_acc"].get(s, 0.0), 2) for s in range(total_seconds)
        ]

        total_dist = sum(dist_arr)
        observed_secs = sum(1 for v in dist_arr if v > 0.0)
        avg_speed_kmh = (
            round((total_dist / observed_secs) * 3.6, 2)
            if observed_secs > 0 else 0.0
        )

        team_id = g["team_id"]
        jersey = g["jersey_number"]
        pname: str | None = None
        if player_name_map and jersey is not None and team_id >= 0:
            pname = player_name_map.get((team_id, jersey))

        records[gk] = {
            "track_ids":              g["track_ids"],
            "jersey_number":          jersey,
            "team_id":                team_id,
            "player_name":            pname,
            "total_distance_m":       round(total_dist, 2),
            "avg_speed_kmh":          avg_speed_kmh,
            "distance_by_second":     dist_arr,
            "max_speed_kmh_by_second": max_spd_arr,
        }

    return records


# ---------------------------------------------------------------------------
# Output JSON helper
# ---------------------------------------------------------------------------


def derive_session_id(
    session_id_arg: str | None,
    calib_path: str,
    big_path:   str,
) -> str:
    """Derive a session_id from filenames if not supplied."""
    if session_id_arg:
        return session_id_arg
    # Try to extract the timestamp+hash segment from a filename like
    # calibration_frames_20260418204058_58334.jsonl
    stem = Path(calib_path).stem  # e.g. "calibration_frames_20260418204058_58334"
    parts = stem.split("_")
    # Take the last two numeric-ish parts as the session signature
    candidate = "_".join(parts[-2:]) if len(parts) >= 2 else stem
    return candidate


def default_output_path(calib_path: str, session_id: str) -> str:
    parent = Path(calib_path).parent
    return str(parent / f"match_stats_{session_id}.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.fps <= 0:
        sys.exit("ERROR: --fps must be > 0")

    fps = args.fps
    print(f"[INFO] FPS: {fps}")

    # -----------------------------------------------------------------------
    # Phase 0 – Load roster (optional)
    # -----------------------------------------------------------------------
    roster_data = load_roster(getattr(args, "roster", None))
    player_name_map, team_name_map = build_roster_player_map(roster_data)
    match_info = roster_data.get("match_info") or {}
    if team_name_map:
        print(f"[INFO] Roster loaded: {team_name_map}")

    # -----------------------------------------------------------------------
    # Phase 1 – Load data
    # -----------------------------------------------------------------------
    print("[INFO] Loading tracking CSV...")
    tracking_df = load_tracking_csv(args.tracking)
    print(f"       {len(tracking_df):,} rows, frames {tracking_df['frame_id'].min()}–{tracking_df['frame_id'].max()}")

    print("[INFO] Loading action-spotting JSONs...")
    big_preds  = load_json_predictions(args.big_model,  "SoccerNet_big")
    ball_preds = load_json_predictions(args.ball_model, "SoccerNetBall_challenge2")
    print(f"       SoccerNet_big: {len(big_preds)} predictions")
    print(f"       SoccerNetBall: {len(ball_preds)} predictions")

    # Validate JSONL fps vs CLI fps
    with open(args.calibration, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    if first_line:
        try:
            first_entry = json.loads(first_line)
            jsonl_fps = float(first_entry.get("fps", fps))
            if abs(jsonl_fps - fps) > 0.1:
                warnings.warn(
                    f"--fps {fps} differs from JSONL fps {jsonl_fps}. "
                    f"Using --fps={fps} as authoritative.",
                    stacklevel=1,
                )
        except (json.JSONDecodeError, ValueError):
            pass

    # -----------------------------------------------------------------------
    # Phase 2 – Build jersey / team maps
    # -----------------------------------------------------------------------
    print("[INFO] Building jersey/team maps from tracking CSV...")
    jersey_map, team_map = build_track_maps(tracking_df)
    print(f"       {len(jersey_map)} unique track IDs found")

    # -----------------------------------------------------------------------
    # Phase 3 – Calibration ↔ tracking ID correspondence
    # -----------------------------------------------------------------------
    print(f"[INFO] Building calib<->tracking ID correspondence (first {ID_MATCH_FRAMES} frames)...")
    id_correspondence = build_id_correspondence(
        args.calibration, tracking_df, fps, n_frames=ID_MATCH_FRAMES
    )
    print(f"       Resolved {len(id_correspondence)} calib IDs -> tracking IDs")

    # -----------------------------------------------------------------------
    # Phase 4 – Single-pass calibration streaming
    # -----------------------------------------------------------------------
    print("[INFO] Streaming calibration JSONL (single pass)...")
    player_dist, player_max_speed, possession_frames, ball_sec_accum, total_frames = (
        stream_calibration(args.calibration, fps, id_correspondence, team_map)
    )
    print(f"       Processed {total_frames:,} calibration frames")

    total_seconds = math.ceil(total_frames / fps)
    print(f"       Total seconds: {total_seconds}")

    # -----------------------------------------------------------------------
    # Phase 5 – Possession & ball position
    # -----------------------------------------------------------------------
    print("[INFO] Aggregating possession and ball position...")
    possession_by_second = build_possession_array(possession_frames, total_seconds)
    ball_position_by_second = build_ball_position_array(ball_sec_accum, total_seconds)
    possession_pct = build_possession_percentages(possession_frames, total_seconds)

    # If calibration-based possession is empty, fall back to tracking-CSV team counts
    calibration_possession_ok = any(v != -1 for v in possession_by_second)
    if not calibration_possession_ok:
        print("[INFO] Calibration possession empty — falling back to tracking-CSV team counts")
        fallback_frames = build_possession_from_tracking(tracking_df, fps, total_seconds)
        possession_by_second = build_possession_array(fallback_frames, total_seconds)
        possession_pct = build_possession_percentages(fallback_frames, total_seconds)
        print(f"       Fallback possession: {possession_pct}")

    # -----------------------------------------------------------------------
    # Phase 6 – Events + team counts
    # -----------------------------------------------------------------------
    print("[INFO] Assembling events...")
    events, team_event_counts = assemble_events(
        big_preds, ball_preds, fps, possession_by_second
    )

    for tid_key in ("0", "1"):
        if tid_key not in team_event_counts:
            team_event_counts[tid_key] = {}

    # -----------------------------------------------------------------------
    # Phase 7 – Player records
    # -----------------------------------------------------------------------
    print("[INFO] Building player records...")
    players = build_player_records(
        player_dist, player_max_speed, jersey_map, team_map, total_seconds,
        player_name_map=player_name_map or None,
    )

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    session_id = derive_session_id(args.session_id, args.calibration, args.big_model)
    output_path = args.output or default_output_path(args.calibration, session_id)

    meta: dict[str, Any] = {
        "session_id":     session_id,
        "fps":            fps,
        "total_frames":   total_frames,
        "total_seconds":  round(total_frames / fps, 3),
        "total_seconds_bucketed": total_seconds,
    }
    if match_info:
        meta["match_info"] = match_info
    if team_name_map:
        meta["team_names"] = {str(k): v for k, v in team_name_map.items()}

    output: dict[str, Any] = {
        "meta": meta,
        "events": events,
        "possession": {
            "by_second":       possession_by_second,
            "team_percentages": possession_pct,
        },
        "team_event_counts": team_event_counts,
        "ball_position_by_second": ball_position_by_second,
        "players": players,
    }

    print(f"[INFO] Writing output to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, separators=(",", ":"))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_team0 = sum(1 for v in possession_by_second if v == 0)
    n_team1 = sum(1 for v in possession_by_second if v == 1)
    n_no_ball = sum(1 for v in possession_by_second if v == -1)
    t0_name = team_name_map.get(0, "Team 0")
    t1_name = team_name_map.get(1, "Team 1")
    print("\n[DONE] Summary")
    print(f"  Events:             {len(events)}")
    print(f"  Players tracked:    {len(players)}")
    print(f"  Possession ({t0_name}): {n_team0}s  |  {t1_name}: {n_team1}s  |  no ball: {n_no_ball}s")
    print(f"  Possession %:        {possession_pct}")
    print(f"  Output:             {output_path}")
    # File size hint
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Output file size:   {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
