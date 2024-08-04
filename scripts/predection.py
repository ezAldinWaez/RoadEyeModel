import cv2 as cv
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from scipy.signal import savgol_filter
from math import atan2, degrees

def predectOnVideo(videoPath, modelPath):
    track_history = defaultdict(lambda: [])

    capture = cv.VideoCapture(videoPath)
    capFps = capture.get(cv.CAP_PROP_FPS)
    capFrameWidth = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    capFrameHight = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    capTimeStep = 1 / capFps

    output_path = videoPath.rsplit('.', 1)[0] + '_predected.mp4'
    writer = cv.VideoWriter(
        filename=output_path,
        fourcc=cv.VideoWriter_fourcc(*'XVID'),
        fps=capFps,
        frameSize=(capFrameWidth, capFrameHight))

    model = YOLO(modelPath)

    count = 0
    while capture.isOpened():

        count += 1
        success, frame = capture.read()
        if not success:
            break
        
        timestamp = count * capTimeStep
        
        roi_x, roi_y = 0, 200
        roi_w, roi_h = 1100, 1080
        roi = frame[roi_y:roi_h, roi_x:roi_w]

        cv.rectangle(
            img=frame,
            pt1=(int(roi_x), int(roi_y)),
            pt2=(int(roi_w), int(roi_h)),
            color=(95, 150, 124),
            thickness=2)

        results = model.track(source=roi, persist=True)

        if (len(results) > 0) & (results[0].boxes.id != None):
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xywh.cpu()

            speeds, angles = calculate_current_motions(track_history, num_points=5)

            for track_id, (x, y, w, h) in zip(track_ids, boxes):
                track_history[track_id].append((timestamp, float(x), float(y)))

                x1 = int(x) - int(w / 2)
                x2 = x1 + w
                y1 = int(y) - int(h / 2)
                y2 = y1 + h

                cv.rectangle(
                    img=roi,
                    pt1=(int(x1), int(y1)),
                    pt2=(int(x2), int(y2)),
                    color=(0, 0, 255),
                    thickness=2)

                text = f"ID: {track_id}"
                fontFace = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                text_width = cv.getTextSize(
                    text, fontFace, fontScale, thickness=2)[0][0]
                cv.putText(
                    img=roi,
                    text=text,
                    org=(int((x1 + w / 2 - text_width / 2)), int(y1 - 10)),
                    fontFace=fontFace,
                    fontScale=fontScale,
                    color=(0, 0, 255),
                    thickness=2)

                # Draw the tracking lines
                track_history_points = np.array([(x, y) for _, x, y in track_history[track_id]])
                history_points = np.hstack(track_history_points).astype(np.int32).reshape((-1, 1, 2))
                cv.polylines(
                    img=roi,
                    pts=[history_points],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2)

                if len(track_history[track_id]) >= 5:
                    # Draw the smoothed tracking lines
                    track_history_points_smoothed = np.array([(x, y) for _, x, y in smooth_path(track_history[track_id])])
                    history_points_smoothed = np.hstack(track_history_points_smoothed).astype(np.int32).reshape((-1, 1, 2))
                    cv.polylines(
                        img=roi,
                        pts=[history_points_smoothed],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=2)


                speed = speeds.get(track_id, 0)
                direction = angles.get(track_id, 0)

                # Draw the speed vector
                speed_v = (int(x + speed * np.cos(direction * np.pi / 180)),
                           int(y + speed * np.sin(direction * np.pi / 180)))
                cv.arrowedLine(
                    img=roi,
                    pt1=(int(x), int(y)),
                    pt2=speed_v,
                    color=(0, 0, 0),
                    thickness=2)

        writer.write(frame)

    capture.release()
    writer.release()

def calculate_current_motions(track_history, num_points=5):
    speeds = {}
    angles = {}
    for track_id, data in track_history.items():
        if len(data) <= num_points:
            continue
        smoothed_data = smooth_path(data, window_length=5, poly_order=2)
        speeds[track_id], angles[track_id] = calculate_current_motion(smoothed_data, num_points)
    return speeds, angles


def calculate_current_motion(data, num_points):
    num_points = min(num_points, len(data))
    recent_data = data[-num_points:]

    timestamps = np.array([point[0] for point in recent_data])
    positions = np.array([(point[1], point[2]) for point in recent_data])

    time_intervals = np.diff(timestamps)
    displacements = np.diff(positions, axis=0)

    velocities = displacements / time_intervals[:, np.newaxis]

    weights = np.arange(1, len(velocities) + 1)
    weighted_velocity = np.average(velocities, axis=0, weights=weights)

    speed = np.linalg.norm(weighted_velocity)

    direction = degrees(atan2(weighted_velocity[1], weighted_velocity[0]))

    return speed, direction

def smooth_path(data, window_length=5, poly_order=2):
    
    # spl = UnivariateSpline(x, y, s=s)

    timestamps = np.array([point[0] for point in data])
    positions = np.array([(point[1], point[2]) for point in data])
    
    smoothed_x = savgol_filter(positions[:, 0], window_length, poly_order)
    smoothed_y = savgol_filter(positions[:, 1], window_length, poly_order)
    
    return list(zip(timestamps, smoothed_x, smoothed_y))

# * To predict on a video, run:
# VIDEO_PATH = '/path/to/video.mp4'
# MODEL_PATH = '../models/pretrained_e50.pt'

# predectOnVideo(VIDEO_PATH, MODEL_PATH)
