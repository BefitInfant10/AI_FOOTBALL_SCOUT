"""
YOLOv8 + DeepSORT football tracker with jersey-colour team classification,
team-aware pass / shot / goal / interception detection.
"""

import os
import math
import json
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ──────────────────── helpers ────────────────────────────

def _apply_nms(boxes_xywh, confidences, conf_thresh=0.3, iou_thresh=0.5):
    if not boxes_xywh:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_thresh, iou_thresh)
    return [int(i) for i in idxs.flatten()] if len(idxs) else []


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ──────────────────── jersey colour sampling ────────────

def sample_jersey_hsv(frame, bbox):
    """Return median HSV of the torso region of a player bbox."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 10:
        return None

    # Sample 3 horizontal strips across the torso (30%-60% of height)
    # to get jersey color while avoiding head and legs
    samples = []
    for frac_start, frac_end in [(0.25, 0.40), (0.35, 0.50), (0.45, 0.60)]:
        sx1 = max(0, x1 + int(0.15 * bw))
        sx2 = min(frame.shape[1], x1 + int(0.85 * bw))
        sy1 = max(0, y1 + int(frac_start * bh))
        sy2 = min(frame.shape[0], y1 + int(frac_end * bh))
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size > 0:
            samples.append(roi)

    if not samples:
        return None

    combined = np.vstack([s.reshape(-1, 3) for s in [cv2.cvtColor(s, cv2.COLOR_BGR2HSV) for s in samples]])

    # Remove pixels that are likely skin (H 0-25, S 40-200, V 60-220) or grass
    h, s, v = combined[:, 0], combined[:, 1], combined[:, 2]

    # Keep only saturated, non-skin, non-green pixels
    skin_mask = (h < 25) & (s > 40) & (s < 200) & (v > 60) & (v < 220)
    grass_mask = (h > 25) & (h < 80) & (s > 30) & (v > 40)
    jersey_mask = ~skin_mask & ~grass_mask & (s > 20)

    if jersey_mask.sum() < 10:
        # fallback to all pixels
        jersey_mask = np.ones(len(combined), dtype=bool)

    filtered = combined[jersey_mask]
    return tuple(np.median(filtered, axis=0))


def assign_teams(player_samples):
    """
    Given {player_id: [list of HSV samples across frames]}, assign teams.
    Uses median HSV per player, then splits by the dominant color channel
    that has the biggest separation.
    """
    player_medians = {}
    for tid, samples in player_samples.items():
        if samples:
            arr = np.array(samples)
            player_medians[tid] = np.median(arr, axis=0)

    if len(player_medians) < 2:
        return {tid: "unknown" for tid in player_samples}

    # Try splitting on each channel (H, S, V) and pick the one with best separation
    best_channel = 2  # default V
    best_score = 0

    for ch in range(3):
        vals = sorted([(tid, m[ch]) for tid, m in player_medians.items()], key=lambda x: x[1])
        max_gap = 0
        for i in range(len(vals) - 1):
            gap = vals[i + 1][1] - vals[i][1]
            if gap > max_gap:
                max_gap = gap
        if max_gap > best_score:
            best_score = max_gap
            best_channel = ch

    vals = sorted([(tid, player_medians[tid][best_channel]) for tid in player_medians], key=lambda x: x[1])
    split_idx = 0
    best_gap = 0
    for i in range(len(vals) - 1):
        gap = vals[i + 1][1] - vals[i][1]
        if gap > best_gap:
            best_gap = gap
            split_idx = i + 1

    if split_idx == 0:
        split_idx = len(vals) // 2

    threshold = (vals[split_idx - 1][1] + vals[split_idx][1]) / 2.0

    teams = {}
    for tid in player_samples:
        if tid in player_medians:
            val = player_medians[tid][best_channel]
            teams[tid] = "team_low" if val < threshold else "team_high"
        else:
            teams[tid] = "unknown"

    return teams


# ──────────────────── main pipeline ─────────────────────

def detect_video(input_video_path, output_video_path="outputs/processed_video.mp4"):
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = YOLO(os.path.join(base_dir, "yolov8s.pt"))
    tracker = DeepSort(max_age=20, n_init=2)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    # ── ball state ──
    ball_pos = None
    ball_prev = None
    ball_missing = 0
    MAX_MISSING = 15
    MAX_JUMP = 150

    # ── possession ──
    POSSESSION_DIST = 80
    MIN_CLOSE_FRAMES = 5
    MIN_HOLD_FRAMES = 8
    EVENT_COOLDOWN = 30

    current_owner = None
    pending_owner = None
    pending_frames = 0
    owner_hold_frames = 0

    # ── event tracking ──
    events = []
    event_text = ""
    event_text_frames = 0
    last_event_frame = -50

    # ── stats ──
    pass_count = 0
    shot_count = 0
    goal_count = 0
    interception_count = 0

    frame_id = 0

    # ── accumulate jersey samples per track_id ──
    jersey_samples = {}
    team_names = {"team_low": "T1", "team_high": "T2"}

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_id += 1
        xmin, xmax = int(0.05 * W), int(0.95 * W)
        ymin, ymax = int(0.10 * H), int(0.90 * H)

        results = model(frame, conf=0.25, imgsz=960, verbose=False)
        annotated = frame.copy()

        # ── YOLO detections ──
        ball_raw = None
        person_boxes, person_confs = [], []

        boxes = results[0].boxes
        if boxes is not None and boxes.xyxy is not None:
            ball_cands = []
            for box in boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if conf < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not (xmin < cx < xmax and ymin < cy < ymax):
                    continue
                if cls == 32:
                    ball_cands.append({"center": (cx, cy), "conf": conf})
                elif cls == 0:
                    bw, bh = x2 - x1, y2 - y1
                    person_boxes.append([float(x1), float(y1), float(bw), float(bh)])
                    person_confs.append(float(conf))

            if ball_cands:
                if ball_pos is not None:
                    selected = min(ball_cands, key=lambda b: _dist(b["center"], ball_pos))
                else:
                    selected = max(ball_cands, key=lambda b: b["conf"])
                raw = selected["center"]
                if ball_pos is None or _dist(raw, ball_pos) <= MAX_JUMP:
                    ball_raw = raw

        # ── ball update ──
        if ball_raw is not None:
            ball_prev = ball_pos
            ball_pos = ball_raw
            ball_missing = 0
        else:
            ball_missing += 1
            if ball_missing > MAX_MISSING:
                ball_prev = None
                ball_pos = None

        # ── player tracking ──
        kept = _apply_nms(person_boxes, person_confs)
        detections = [[person_boxes[i], person_confs[i], 0] for i in kept]
        tracks = tracker.update_tracks(detections, frame=frame)

        players = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not (xmin < cx < xmax and ymin < cy < ymax):
                continue

            hsv = sample_jersey_hsv(frame, (x1, y1, x2, y2))
            if hsv is not None:
                if tid not in jersey_samples:
                    jersey_samples[tid] = []
                jersey_samples[tid].append(hsv)

            players.append({
                "id": tid,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "jersey_hsv": hsv,
            })

        # ── team assignment (use accumulated data for robustness) ──
        frame_teams = assign_teams(jersey_samples)

        # ── ball speed ──
        ball_speed = _dist(ball_pos, ball_prev) if (ball_pos and ball_prev) else 0.0

        # ── find closest player to ball ──
        closest = None
        closest_d = POSSESSION_DIST
        if ball_pos is not None:
            for p in players:
                d = _dist(p["center"], ball_pos)
                if d < closest_d:
                    closest_d = d
                    closest = p

        new_owner = closest["id"] if closest else None
        new_team = frame_teams.get(new_owner, "unknown") if closest else None

        # ── stable ownership switch ──
        if new_owner is not None:
            if new_owner == current_owner:
                pending_owner = None
                pending_frames = 0
                owner_hold_frames += 1
            elif new_owner == pending_owner:
                pending_frames += 1
                if pending_frames >= MIN_CLOSE_FRAMES:
                    old_owner = current_owner
                    old_team = frame_teams.get(current_owner, "unknown") if current_owner else "unknown"
                    old_hold = owner_hold_frames

                    current_owner = new_owner
                    pending_owner = None
                    pending_frames = 0
                    owner_hold_frames = 0

                    if old_owner is None or old_hold < MIN_HOLD_FRAMES:
                        pass
                    elif new_team == old_team:
                        # SAME TEAM = PASS
                        if (frame_id - last_event_frame) > EVENT_COOLDOWN:
                            pass_count += 1
                            last_event_frame = frame_id
                            event_text = "PASS"
                            event_text_frames = 30
                            events.append({
                                "type": "pass",
                                "from": old_owner, "to": current_owner,
                                "from_team": old_team, "to_team": new_team,
                                "frame": frame_id,
                            })
                    else:
                        # DIFFERENT TEAM = interception
                        if (frame_id - last_event_frame) > EVENT_COOLDOWN:
                            interception_count += 1
                            last_event_frame = frame_id
                            event_text = "INTERCEPT"
                            event_text_frames = 30
                            events.append({
                                "type": "interception",
                                "from": old_owner, "to": current_owner,
                                "from_team": old_team, "to_team": new_team,
                                "frame": frame_id,
                            })
            else:
                pending_owner = new_owner
                pending_frames = 1

        # ── shot: very fast ball in goal zone with no nearby player ──
        if (ball_pos is not None
                and ball_speed > 40
                and ball_pos[0] > W * 0.80
                and closest_d > POSSESSION_DIST
                and (frame_id - last_event_frame) > EVENT_COOLDOWN):
            shot_count += 1
            last_event_frame = frame_id
            event_text = "SHOT!"
            event_text_frames = 30
            events.append({
                "type": "shot",
                "from": current_owner,
                "from_team": frame_teams.get(current_owner, "unknown"),
                "frame": frame_id,
            })

        # ── goal: ball in goal rectangle ──
        if ball_pos is not None:
            gx1, gy1 = int(W * 0.85), int(H * 0.25)
            gx2, gy2 = W, int(H * 0.75)
            if gx1 <= ball_pos[0] <= gx2 and gy1 <= ball_pos[1] <= gy2:
                if (frame_id - last_event_frame) > EVENT_COOLDOWN:
                    goal_count += 1
                    last_event_frame = frame_id
                    event_text = "GOAL!"
                    event_text_frames = int(fps * 3)
                    events.append({
                        "type": "goal",
                        "player": current_owner,
                        "team": frame_teams.get(current_owner, "unknown"),
                        "frame": frame_id,
                    })

        # ── draw players ──
        TEAM_COLORS = {"team_low": (0, 0, 255), "team_high": (255, 200, 0), "unknown": (0, 255, 0)}
        for p in players:
            x1, y1, x2, y2 = p["bbox"]
            tid = p["id"]
            team = frame_teams.get(tid, "unknown")

            if tid == current_owner:
                color = (0, 255, 255)
                lbl = f"{team_names.get(team, '?')} #{tid} (Ball)"
            else:
                color = TEAM_COLORS.get(team, (0, 255, 0))
                lbl = f"{team_names.get(team, '?')} #{tid}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, lbl, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # ── draw ball ──
        if ball_pos is not None:
            bx, by = ball_pos
            cv2.circle(annotated, (bx, by), 12, (0, 0, 255), 3)
            cv2.putText(annotated, "BALL", (bx + 15, by + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ── event flash ──
        if event_text_frames > 0:
            event_text_frames -= 1
            ty = H // 2 if "GOAL" in event_text else 50
            sz = 2.0 if "GOAL" in event_text else 1.0
            thk = 4 if "GOAL" in event_text else 3
            clr = (0, 0, 255) if "GOAL" in event_text or "SHOT" in event_text else (0, 255, 255)
            tx = W // 2 - 100 if "GOAL" in event_text else 30
            cv2.putText(annotated, event_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, sz, clr, thk)

        # ── stats bar ──
        cv2.rectangle(annotated, (5, 5), (280, 80), (0, 0, 0), -1)
        cv2.putText(annotated, f"Passes: {pass_count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Shots: {shot_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(annotated, f"Goals: {goal_count}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(annotated)

    cap.release()
    writer.release()

    # ── resolve team names ──
    low_count = sum(1 for t in jersey_samples if frame_teams.get(t) == "team_low")
    high_count = sum(1 for t in jersey_samples if frame_teams.get(t) == "team_high")

    # Print debug info
    print(f"\nTeam split: team_low={low_count} players, team_high={high_count} players")
    for tid in sorted(jersey_samples.keys()):
        team = frame_teams.get(tid, "?")
        samples = jersey_samples[tid]
        med = np.median(samples, axis=0)
        print(f"  Player {tid}: {team} median HSV=({med[0]:.0f},{med[1]:.0f},{med[2]:.0f}) n={len(samples)}")

    if low_count >= high_count:
        team_names_resolved = {"team_low": "Team 1", "team_high": "Team 2"}
    else:
        team_names_resolved = {"team_low": "Team 2", "team_high": "Team 1"}

    resolved_events = []
    for ev in events:
        rev = dict(ev)
        for key in ("from_team", "to_team", "team"):
            if key in rev and rev[key] in team_names_resolved:
                rev[key] = team_names_resolved[rev[key]]
        resolved_events.append(rev)

    stats = {
        "passes": pass_count,
        "shots": shot_count,
        "goals": goal_count,
        "interceptions": interception_count,
        "events": resolved_events,
    }

    stats_path = os.path.join(base_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    return output_video_path, stats
