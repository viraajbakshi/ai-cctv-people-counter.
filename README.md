# ai-cctv-people-counter.
Real-time people counting, restricted-zone alerts and dwell-time logging using YOLOv8, ByteTrack and OpenCV
A small computer vision demo that counts people in a video or webcam feed, raises an alert when someone enters a restricted zone, and logs how long each person stays inside it.

Built as a learning project using YOLOv8, ByteTrack and OpenCV.

## Features
- Detects and tracks people, with a stable ID for each person
- Shows people currently in view and the total unique people seen
- Restricted zone that turns red with an on-screen alert when someone enters
- Dwell time shown for each person inside the zone
- Saves an annotated video (`output.mp4`) and an event log (`events.csv`)

## How it works
1. OpenCV reads the video one frame at a time.
2. A pretrained YOLOv8n model detects people in each frame (COCO class 0).
3. ByteTrack, built into Ultralytics, assigns a persistent ID to each person.
4. The bottom-centre of each person's box is checked against the zone polygon.
5. Enter and exit events with dwell time are written to a CSV file.
