"""
Mini AI CCTV project: people counting + restricted-zone alert + dwell time.

Run (inside your venv):
    python cctv_demo.py --source 0            # webcam
    python cctv_demo.py --source test.mp4     # video file
    python cctv_demo.py --source test.mp4 --no-show   # no window, just save output

Press q in the video window to stop.

Outputs (saved next to this script):
    output.mp4   annotated video
    events.csv   zone ENTER / EXIT log with dwell time
"""
import argparse
import csv

import cv2
import numpy as np
from ultralytics import YOLO

# Restricted zone as fractions of the frame (x, y), top-left origin.
# Edit these numbers to move or resize the zone.
ZONE = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]

PERSON_CLASS = 0   # COCO class 0 = person. Swap for your own classes later (e.g. helmet).
CONF = 0.4         # minimum detection confidence

GREEN = (0, 200, 0)
RED = (0, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam, or a video file path")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--csv", default="events.csv")
    ap.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(args.model)

    writer = None
    zone_px = None
    seen_ids = set()         # every unique person ID seen so far
    in_zone_since = {}       # track_id -> frame index when they entered the zone
    events = []              # rows for the CSV
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

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
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                people_now += 1
                seen_ids.add(tid)

                # Use the feet point (bottom centre of the box) to test the zone
                foot = (float((x1 + x2) // 2), float(y2))
                inside = cv2.pointPolygonTest(zone_px, foot, False) >= 0
                label = f"ID {tid}"

                if inside:
                    in_zone_now.add(tid)
                    if tid not in in_zone_since:
                        in_zone_since[tid] = frame_idx
                        events.append((round(frame_idx / fps, 2), "ENTER", tid, 0.0))
                    dwell = (frame_idx - in_zone_since[tid]) / fps
                    label += f"  {dwell:.1f}s"

                color = RED if inside else GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Anyone who was in the zone but is not now has left (or been lost)
        for tid in list(in_zone_since):
            if tid not in in_zone_now:
                dwell = (frame_idx - in_zone_since.pop(tid)) / fps
                events.append((round(frame_idx / fps, 2), "EXIT", tid, round(dwell, 2)))

        # Draw the zone and the on-screen summary
        zone_color = RED if in_zone_now else GREEN
        cv2.polylines(frame, [zone_px], True, zone_color, 2)
        summary = (f"People now: {people_now} | Total seen: {len(seen_ids)} "
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

    # Close out anyone still inside the zone when the video ends
    for tid, start in in_zone_since.items():
        events.append((round(frame_idx / fps, 2), "EXIT", tid, round((frame_idx - start) / fps, 2)))

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time_s", "event", "track_id", "dwell_s"])
        wr.writerows(events)

    print(f"Done. {len(seen_ids)} unique people, {len(events)} zone events.")
    print(f"Saved video to {args.out} and log to {args.csv}")


if __name__ == "__main__":
    main()
