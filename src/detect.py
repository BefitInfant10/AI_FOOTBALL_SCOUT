"""
Football event detector — teams, passes, shots on goal.

All event logic uses POSITION + JERSEY COLOR per frame.
DeepSORT is only used for visual labels, never for event decisions.
This makes the system robust against tracking ID changes.
"""

import os
import math
import json
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ──────────────────── utilities ──────────────────────────

def _convert_to_browser_mp4(input_path, output_path):
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
        if os.path.exists(output_path):
            os.replace(output_path, input_path)
    except Exception:
        pass


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ──────────────────── jersey colour ──────────────────────

def sample_jersey_hsv(frame, bbox):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 10:
        return None
    samples = []
    for fs, fe in [(0.25, 0.45), (0.35, 0.55), (0.45, 0.65)]:
        sx1 = max(0, x1 + int(0.15 * bw))
        sx2 = min(frame.shape[1], x1 + int(0.85 * bw))
        sy1 = max(0, y1 + int(fs * bh))
        sy2 = min(frame.shape[0], y1 + int(fe * bh))
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size > 0:
            samples.append(roi)
    if not samples:
        return None
    combined = np.vstack([
        s.reshape(-1, 3)
        for s in [cv2.cvtColor(s, cv2.COLOR_BGR2HSV) for s in samples]
    ])
    h, s, v = combined[:, 0], combined[:, 1], combined[:, 2]
    skin = (h < 25) & (s > 40) & (s < 200) & (v > 60) & (v < 220)
    grass = (h > 25) & (h < 80) & (s > 30) & (v > 40)
    jersey = ~skin & ~grass & (s > 20)
    if jersey.sum() < 10:
        jersey = np.ones(len(combined), dtype=bool)
    return tuple(np.median(combined[jersey], axis=0))


def _classify_frame_teams(player_hsvs):
    """
    Given list of (index, hsv_tuple), cluster into two groups.
    Returns dict {index: 'A' | 'B'}.
    Uses accumulated global samples for stability when available.
    """
    if len(player_hsvs) < 2:
        return {i: 'A' for i, _ in player_hsvs}

    best_ch, best_gap = 2, 0
    vals_all = [(i, hsv[ch]) for i, hsv in player_hsvs for ch in range(3)]
    for ch in range(3):
        items = sorted([(i, hsv[ch]) for i, hsv in player_hsvs], key=lambda x: x[1])
        gap = max(
            (items[j + 1][1] - items[j][1] for j in range(len(items) - 1)),
            default=0,
        )
        if gap > best_gap:
            best_gap = gap
            best_ch = ch

    items = sorted(
        [(i, hsv[best_ch]) for i, hsv in player_hsvs], key=lambda x: x[1],
    )

    best_split, split_idx = 0, len(items) // 2
    for j in range(len(items) - 1):
        g = items[j + 1][1] - items[j][1]
        if g > best_split:
            best_split = g
            split_idx = j + 1

    if split_idx == 0:
        split_idx = 1

    threshold = (items[split_idx - 1][1] + items[split_idx][1]) / 2.0

    return {
        i: ('A' if hsv[best_ch] < threshold else 'B')
        for i, hsv in player_hsvs
    }


# ──────────────────── global team accumulation ───────────

class TeamAccumulator:
    """
    Accumulates jersey samples per track ID across frames.
    Cluster splits are recomputed periodically for a stable global assignment.
    """

    def __init__(self):
        self.samples = {}
        self.assignment = {}

    def update(self, track_id, hsv):
        if hsv is None:
            return
        self.samples.setdefault(track_id, []).append(hsv)

    def reassign(self):
        if len(self.samples) < 2:
            return

        medians = {}
        for tid, slist in self.samples.items():
            if len(slist) >= 3:
                medians[tid] = np.median(np.array(slist[-30:]), axis=0)

        if len(medians) < 2:
            return

        best_ch, best_gap = 2, 0
        for ch in range(3):
            items = sorted(
                [(t, m[ch]) for t, m in medians.items()], key=lambda x: x[1],
            )
            gap = max(
                (items[j + 1][1] - items[j][1] for j in range(len(items) - 1)),
                default=0,
            )
            if gap > best_gap:
                best_gap = gap
                best_ch = ch

        items = sorted(
            [(t, medians[t][best_ch]) for t in medians], key=lambda x: x[1],
        )
        best_split, split_idx = 0, len(items) // 2
        for j in range(len(items) - 1):
            g = items[j + 1][1] - items[j][1]
            if g > best_split:
                best_split = g
                split_idx = j + 1
        if split_idx == 0:
            split_idx = 1

        thr = (items[split_idx - 1][1] + items[split_idx][1]) / 2.0
        self.assignment = {
            t: ('A' if medians[t][best_ch] < thr else 'B')
            for t in medians
        }

    def get(self, track_id):
        return self.assignment.get(track_id, None)


# ──────────────────── main pipeline ──────────────────────

def detect_video(input_video_path, output_video_path="outputs/processed_video.mp4"):
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = YOLO(os.path.join(base_dir, "yolov8s.pt"))
    tracker = DeepSort(max_age=45, n_init=5)

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

    # ── thresholds ──
    POSSESSION_DIST = 0.06 * W
    MIN_HOLD_FRAMES = max(2, int(fps * 0.08))
    SHOT_SPEED = 0.012 * W
    EVENT_COOLDOWN = max(25, int(fps * 1.5))
    GOAL_COOLDOWN = max(40, int(fps * 2.0))
    GOAL_CONFIRM_WINDOW = int(fps * 5)
    MAX_BALL_MISSING = 15
    MAX_BALL_JUMP = 0.10 * W
    MAX_TRANSIT_FRAMES = int(fps * 1.5)

    # ── ball state ──
    ball_pos = None
    ball_prev = None
    ball_missing = 0

    # ── possession state ──
    poss_player_pos = None
    poss_player_idx = None
    poss_team = None
    poss_hold_frames = 0

    # ── transit state (ball between players) ──
    in_transit = False
    transit_frames = 0
    transit_sender_pos = None
    transit_sender_idx = None
    transit_sender_team = None

    ball_free_frames = 0

    # ── event tracking ──
    last_event_frame = -999
    last_shot_frame = -999
    last_shot_team = None
    event_text = ""
    event_text_frames = 0

    pass_count = 0
    shot_count = 0
    goal_count = 0
    shot_saved_count = 0
    events = []

    team_accum = TeamAccumulator()
    frame_id = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_id += 1
        results = model(frame, conf=0.25, imgsz=960, verbose=False)
        annotated = frame.copy()

        # ── YOLO detections ──
        ball_raw = None
        person_boxes, person_confs = [], []

        boxes = results[0].boxes
        if boxes is not None and boxes.xyxy is not None:
            for box in boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if conf < 0.25:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cls == 32 and conf > 0.15:
                    ball_raw = (cx, cy)
                elif cls == 0:
                    bw, bh = x2 - x1, y2 - y1
                    person_boxes.append(
                        [float(x1), float(y1), float(bw), float(bh)]
                    )
                    person_confs.append(float(conf))

        # ── ball update ──
        if ball_raw is not None:
            if ball_pos is None or _dist(ball_raw, ball_pos) <= MAX_BALL_JUMP:
                ball_prev = ball_pos
                ball_pos = ball_raw
                ball_missing = 0
            else:
                ball_missing += 1
                if ball_missing > MAX_BALL_MISSING:
                    ball_prev = ball_pos
                    ball_pos = None
        else:
            ball_missing += 1
            if ball_missing > MAX_BALL_MISSING:
                ball_prev = ball_pos
                ball_pos = None

        # ── DeepSORT (visual labels only) ──
        if person_boxes:
            idxs = cv2.dnn.NMSBoxes(
                person_boxes, person_confs, 0.25, 0.5
            )
            kept = [int(i) for i in idxs.flatten()] if len(idxs) else []
            detections = [[person_boxes[i], person_confs[i], 0] for i in kept]
        else:
            detections = []
        tracks = tracker.update_tracks(detections, frame=frame)

        # ── collect per-frame player data ──
        frame_players = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            hsv = sample_jersey_hsv(frame, (x1, y1, x2, y2))
            team_accum.update(tid, hsv)
            frame_players.append({
                'id': tid,
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'hsv': hsv,
            })

        # ── reassign global teams periodically ──
        if frame_id % 30 == 0 or frame_id == 1:
            team_accum.reassign()

        # ── classify this frame's players into teams ──
        frame_hsvs = [
            (idx, p['hsv'])
            for idx, p in enumerate(frame_players)
            if p['hsv'] is not None
        ]
        frame_teams = _classify_frame_teams(frame_hsvs)

        for idx, p in enumerate(frame_players):
            global_team = team_accum.get(p['id'])
            frame_team = frame_teams.get(idx, None)
            p['team'] = global_team or frame_team or 'A'

        # ── find closest player to ball ──
        closest = None
        closest_d = float('inf')
        if ball_pos is not None:
            for idx, p in enumerate(frame_players):
                d = _dist(p['center'], ball_pos)
                if d < closest_d:
                    closest_d = d
                    closest = p

        # ── current frame's nearest player info ──
        new_poss_idx = None
        new_poss_pos = None
        new_poss_team = None
        if closest is not None and closest_d <= POSSESSION_DIST:
            new_poss_idx = closest['id']
            new_poss_pos = closest['center']
            new_poss_team = closest['team']

        # ── ball speed ──
        ball_speed = (
            _dist(ball_pos, ball_prev)
            if (ball_pos and ball_prev)
            else 0.0
        )
        ball_dx = (
            ball_pos[0] - ball_prev[0]
            if (ball_pos and ball_prev)
            else 0
        )

        # ── ball free tracking ──
        if ball_pos is not None:
            if new_poss_idx is not None:
                ball_free_frames = 0
            else:
                ball_free_frames += 1
        else:
            ball_free_frames += 1

        ball_is_free = ball_free_frames >= max(3, int(fps * 0.15))

        # ═══════════════════════════════════════════════════
        #  POSSESSION + TRANSIT LOGIC
        # ═══════════════════════════════════════════════════

        if new_poss_idx is not None:

            if in_transit:
                # ball was travelling between players — check pass/save
                if (transit_sender_team is not None
                        and new_poss_team == transit_sender_team
                        and transit_sender_pos is not None
                        and new_poss_pos is not None
                        and transit_frames >= 1):

                    ball_travelled = _dist(new_poss_pos, transit_sender_pos)
                    if (ball_travelled >= 0.03 * W
                            and (frame_id - last_event_frame) > EVENT_COOLDOWN):
                        pass_count += 1
                        last_event_frame = frame_id
                        old_pid = str(transit_sender_idx) if transit_sender_idx else "?"
                        new_pid = str(closest['id'])
                        desc = f"PASS: {transit_sender_team} #{old_pid} \u2192 {transit_sender_team} #{new_pid}"
                        event_text = "PASS"
                        event_text_frames = int(fps * 1.0)
                        ev = {
                            "type": "pass",
                            "from_player": old_pid,
                            "to_player": new_pid,
                            "team": transit_sender_team,
                            "frame": frame_id,
                            "time": round(frame_id / fps, 1),
                            "distance": round(ball_travelled, 1),
                            "description": desc,
                        }
                        events.append(ev)
                        print(f"  {desc}")

                if (transit_sender_team is not None
                        and new_poss_team != transit_sender_team
                        and (frame_id - last_shot_frame) < GOAL_CONFIRM_WINDOW
                        and transit_sender_team == last_shot_team):

                    near_goal = (new_poss_pos is not None
                                 and (new_poss_pos[0] < W * 0.20
                                       or new_poss_pos[0] > W * 0.80))
                    ball_near_goal = (ball_pos is not None
                                      and (ball_pos[0] < W * 0.20
                                            or ball_pos[0] > W * 0.80))
                    if (near_goal or ball_near_goal
                            and (frame_id - last_event_frame) > EVENT_COOLDOWN):
                        shot_saved_count += 1
                        last_event_frame = frame_id
                        pid = str(transit_sender_idx) if transit_sender_idx else "?"
                        desc = f"SHOT ON TARGET: {transit_sender_team} #{pid} \u2192 Goalkeeper"
                        event_text = "SAVED!"
                        event_text_frames = int(fps * 1.5)
                        ev = {
                            "type": "shot_saved",
                            "player": pid,
                            "team": transit_sender_team,
                            "frame": frame_id,
                            "time": round(frame_id / fps, 1),
                            "description": desc,
                        }
                        events.append(ev)
                        print(f"  {desc}")

                in_transit = False
                transit_frames = 0

            if (poss_team is not None
                    and new_poss_team == poss_team
                    and poss_player_pos is not None
                    and _dist(new_poss_pos, poss_player_pos) < 0.10 * W):
                poss_hold_frames += 1
                poss_player_pos = new_poss_pos
            else:
                poss_player_pos = new_poss_pos
                poss_player_idx = new_poss_idx
                poss_team = new_poss_team
                poss_hold_frames = 0

        elif new_poss_idx is None and poss_team is not None:

            if not in_transit:
                in_transit = True
                transit_frames = 0
                transit_sender_pos = poss_player_pos
                transit_sender_idx = poss_player_idx
                transit_sender_team = poss_team

            transit_frames += 1

            if (poss_hold_frames >= MIN_HOLD_FRAMES
                    and ball_pos is not None
                    and ball_speed > SHOT_SPEED
                    and abs(ball_dx) > SHOT_SPEED * 0.3
                    and poss_player_pos is not None
                    and transit_frames <= int(fps * 0.5)):

                moving_right = ball_dx > 0
                moving_left = ball_dx < 0
                shot_right = (
                    moving_right
                    and poss_player_pos[0] < W * 0.75
                    and ball_pos[0] > W * 0.45
                )
                shot_left = (
                    moving_left
                    and poss_player_pos[0] > W * 0.25
                    and ball_pos[0] < W * 0.55
                )

                if shot_right or shot_left:
                    if (frame_id - last_event_frame) > EVENT_COOLDOWN:
                        direction = 'right' if shot_right else 'left'
                        shot_count += 1
                        last_event_frame = frame_id
                        last_shot_frame = frame_id
                        last_shot_team = poss_team
                        pid = str(poss_player_idx) if poss_player_idx else "?"
                        desc = f"SHOT: {poss_team} #{pid} \u2192 {direction.title()} Goal"
                        event_text = "SHOT!"
                        event_text_frames = int(fps * 1.5)
                        ev = {
                            "type": "shot",
                            "player": pid,
                            "team": poss_team,
                            "direction": direction,
                            "frame": frame_id,
                            "time": round(frame_id / fps, 1),
                            "description": desc,
                        }
                        events.append(ev)
                        print(f"  {desc}")

            if transit_frames > MAX_TRANSIT_FRAMES:
                in_transit = False
                transit_frames = 0
                poss_player_pos = None
                poss_player_idx = None
                poss_team = None
                poss_hold_frames = 0

        elif new_poss_idx is None and poss_team is None and in_transit:
            transit_frames += 1
            if transit_frames > MAX_TRANSIT_FRAMES:
                in_transit = False
                transit_frames = 0

        # ═══════════════════════════════════════════════════
        #  GOAL DETECTION
        # ═══════════════════════════════════════════════════
        if ball_pos is not None:
            in_goal_right = (
                ball_pos[0] > W * 0.85
                and H * 0.15 < ball_pos[1] < H * 0.85
            )
            in_goal_left = (
                ball_pos[0] < W * 0.15
                and H * 0.15 < ball_pos[1] < H * 0.85
            )
            if (in_goal_right or in_goal_left):
                recent_shot = (frame_id - last_shot_frame) < GOAL_CONFIRM_WINDOW
                if recent_shot and (frame_id - last_event_frame) > GOAL_COOLDOWN:
                    goal_count += 1
                    last_event_frame = frame_id
                    direction = 'right' if in_goal_right else 'left'
                    team_label = last_shot_team or 'Unknown'
                    event_text = "GOAL!"
                    event_text_frames = int(fps * 3)
                    desc = f"GOAL: {team_label} scored ({direction})"
                    ev = {
                        "type": "goal",
                        "team": team_label,
                        "direction": direction,
                        "frame": frame_id,
                        "time": round(frame_id / fps, 1),
                        "description": desc,
                    }
                    events.append(ev)
                    print(f"  {desc}")

        # ═══════════════════════════════════════════════════
        #  DRAWING
        # ═══════════════════════════════════════════════════
        TEAM_COLORS = {
            'A': (0, 200, 255),
            'B': (255, 150, 0),
        }

        for p in frame_players:
            x1, y1, x2, y2 = p['bbox']
            tid = p['id']
            team = p['team']

            if tid == poss_player_idx:
                color = (0, 255, 255)
                lbl = f"{team} #{tid} (BALL)"
            else:
                color = TEAM_COLORS.get(team, (0, 255, 0))
                lbl = f"{team} #{tid}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, lbl, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        if ball_pos is not None:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(annotated, (bx, by), 12, (0, 0, 255), 3)
            cv2.putText(
                annotated, "BALL", (bx + 15, by + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )

        # ── event flash ──
        if event_text_frames > 0:
            event_text_frames -= 1
            if "GOAL" in event_text:
                ty, sz, thk = H // 2, 2.5, 5
                clr = (0, 0, 255)
                tx = W // 2 - 120
            elif "SHOT" in event_text:
                ty, sz, thk = 60, 1.5, 3
                clr = (0, 165, 255)
                tx = 30
            elif "SAVED" in event_text:
                ty, sz, thk = 60, 1.2, 3
                clr = (180, 0, 180)
                tx = 30
            else:
                ty, sz, thk = 60, 1.0, 3
                clr = (0, 255, 255)
                tx = 30
            cv2.putText(
                annotated, event_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, sz, clr, thk,
            )

        # ── stats bar ──
        cv2.rectangle(annotated, (5, 5), (340, 100), (0, 0, 0), -1)
        cv2.putText(annotated, f"Passes: {pass_count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Shots: {shot_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(annotated, f"Goals: {goal_count}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(annotated, f"Saved: {shot_saved_count}", (10, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 180), 2)

        writer.write(annotated)

    cap.release()
    writer.release()

    _convert_to_browser_mp4(output_video_path, output_video_path + ".h264.mp4")

    a_count = sum(1 for t in team_accum.assignment.values() if t == 'A')
    b_count = sum(1 for t in team_accum.assignment.values() if t == 'B')
    print(f"\n{'='*40}")
    print(f"  Team A players: {a_count}   Team B players: {b_count}")
    print(f"  Passes: {pass_count}   Shots: {shot_count}   "
          f"Goals: {goal_count}   Saved: {shot_saved_count}")
    print(f"{'='*40}\n")

    stats = {
        "passes": pass_count,
        "shots": shot_count,
        "goals": goal_count,
        "interceptions": 0,
        "shot_saved": shot_saved_count,
        "events": events,
    }

    stats_dir = os.path.dirname(output_video_path) or "."
    stats_path = os.path.join(stats_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    return output_video_path, stats
