"""
Mini AI CCTV project: people counting + restricted-zone alert + dwell time.

Run (inside your venv):
    python cctv_demo.py --source 0            # webcam
    python cctv_demo.py --source test.mp4     # video file
    python cctv_demo.py --source test.mp4 --no-show   # no window, just save output

Press q in the video window to stop.

Outputs (saved in the folder you run the command from, unless you pass a path):
    output.mp4   annotated video
    events.csv   zone ENTER / EXIT log with dwell time (written as events happen)
"""
import argparse
import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Restricted zone as fractions of the frame (x, y), top-left origin.
# Edit these numbers to move or resize the zone.
ZONE = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]

PERSON_CLASS = 0   # COCO class 0 = person. Swap for your own classes later (e.g. helmet).
CONF = 0.4         # minimum detection confidence
EXIT_GRACE_S = 1.0 # someone must be out of the zone (or undetected) this long before we log EXIT
MIN_TRACK_FRAMES = 5  # an ID must be seen this many frames before it counts toward "Total seen"

GREEN = (0, 200, 0)
RED = (0, 0, 255)
GREY = (180, 180, 180)


def is_live(source):
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam, or a video file path / stream URL")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--csv", default="events.csv")
    ap.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    live = is_live(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0
    model = YOLO(args.model)

    # Video files: time comes from the frame number (exact, independent of processing speed).
    # Live sources: time comes from the wall clock, since YOLO may run slower than the camera.
    start_wall = time.time()

    def now(frame_idx):
        return time.time() - start_wall if live else frame_idx / fps

    csv_file = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_s", "event", "track_id", "dwell_s"])
    n_events = 0

    def log(t, event, tid, dwell):
        nonlocal n_events
        csv_writer.writerow([round(t, 2), event, tid, round(dwell, 2)])
        csv_file.flush()   # so nothing is lost if the script crashes
        n_events += 1

    writer = None
    zone_px = None
    frames_seen = {}         # track_id -> number of frames this ID has been seen
    counted_ids = set()      # IDs seen for at least MIN_TRACK_FRAMES ("Total seen")
    entered_at = {}          # track_id -> time they entered the zone
    last_in_zone = {}        # track_id -> last time they were seen inside the zone
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            t = now(frame_idx)

            if writer is None:
                zone_px = np.array([(int(x * w), int(y * h)) for x, y in ZONE], np.int32)
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Detect + track people (ByteTrack keeps the same ID across frames)
            result = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS],
                conf=CONF,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            in_zone_now = set()
            people_now = 0
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                # The tracker may not have assigned IDs yet; still count and draw those people.
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    people_now += 1

                    # Use the feet point (bottom centre of the box) to test the zone
                    foot = (float((x1 + x2) // 2), float(y2))
                    inside = cv2.pointPolygonTest(zone_px, foot, False) >= 0

                    if tid is None:
                        label = "person"
                        color = RED if inside else GREY
                    else:
                        frames_seen[tid] = frames_seen.get(tid, 0) + 1
                        if frames_seen[tid] >= MIN_TRACK_FRAMES:
                            counted_ids.add(tid)
                        label = f"ID {tid}"
                        if inside:
                            in_zone_now.add(tid)
                            last_in_zone[tid] = t
                            if tid not in entered_at:
                                entered_at[tid] = t
                                log(t, "ENTER", tid, 0.0)
                            label += f"  {t - entered_at[tid]:.1f}s"
                        color = RED if inside else GREEN

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Only log EXIT once someone has been out of the zone (or lost) for EXIT_GRACE_S,
            # so a single missed detection doesn't split one visit into two.
            for tid in list(entered_at):
                if tid not in in_zone_now and t - last_in_zone[tid] > EXIT_GRACE_S:
                    left = last_in_zone.pop(tid)
                    log(left, "EXIT", tid, left - entered_at.pop(tid))

            # Draw the zone and the on-screen summary
            zone_color = RED if in_zone_now else GREEN
            cv2.polylines(frame, [zone_px], True, zone_color, 2)
            summary = (f"People now: {people_now} | Total seen: {len(counted_ids)} "
                       f"| In zone: {len(in_zone_now)}")
            cv2.putText(frame, summary, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if in_zone_now:
                cv2.putText(frame, "ALERT: zone intrusion", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)

            writer.write(frame)
            if not args.no_show:
                cv2.imshow("AI CCTV demo (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        # Close out anyone still inside the zone when the video ends (or the script stops)
        for tid, start in entered_at.items():
            left = last_in_zone[tid]
            log(left, "EXIT", tid, left - start)

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        csv_file.close()

    print(f"Done. {len(counted_ids)} unique people, {n_events} zone events.")
    print(f"Saved video to {args.out} and log to {args.csv}")


if __name__ == "__main__":
    main()
