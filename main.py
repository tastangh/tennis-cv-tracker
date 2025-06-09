#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2 
import math
import os

LOWER_COURT_LINES = np.array([0, 0, 180])
UPPER_COURT_LINES = np.array([180, 50, 255])

def detect_court_in_roi(roi_frame):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    _, line_mask = cv2.threshold(enhanced_gray, 210, 255, cv2.THRESH_BINARY)

    h, w = roi_frame.shape[:2]
    min_line_length = int(w * 0.10)
    max_line_gap = int(w * 0.05)
    hough_threshold = 30

    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180,
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    if lines is None: return None

    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45 or angle > 135:
            horizontal.append(line)
        else:
            vertical.append(line)

    if len(horizontal) < 2 or len(vertical) < 2: return None

    horizontal.sort(key=lambda l: (l[0][1] + l[0][3]) / 2)
    vertical.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)

    top_line, bottom_line = horizontal[0], horizontal[-1]
    left_line, right_line = vertical[0], vertical[-1]

    def get_line_params(line):
        (x1, y1, x2, y2) = line[0]
        m = float('inf') if x1 == x2 else (y2 - y1) / (x2 - x1)
        c = x1 if m == float('inf') else y1 - m * x1
        return m, c

    def get_intersection(p1, p2):
        m1, c1 = p1
        m2, c2 = p2
        if m1 == m2: return None
        if m1 == float('inf'): return (int(c1), int(m2 * c1 + c2))
        if m2 == float('inf'): return (int(c2), int(m1 * c2 + c1))
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        return (int(x), int(y))

    p_top = get_line_params(top_line)
    p_bottom = get_line_params(bottom_line)
    p_left = get_line_params(left_line)
    p_right = get_line_params(right_line)

    tl = get_intersection(p_top, p_left)
    tr = get_intersection(p_top, p_right)
    bl = get_intersection(p_bottom, p_left)
    br = get_intersection(p_bottom, p_right)

    if not all([tl, tr, bl, br]): return None

    return np.array([tl, tr, br, bl], dtype=np.int32)

def merge_overlapping_boxes(boxes, proximity_thresh=75):
    if len(boxes) == 0:
        return []

    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                box1 = boxes[i]
                box2 = boxes[j]

                dist_x = max(0, max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2]))
                dist_y = max(0, max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3]))

                if dist_x < proximity_thresh and dist_y < proximity_thresh:
                    min_x = min(box1[0], box2[0])
                    min_y = min(box1[1], box2[1])
                    max_x = max(box1[0] + box1[2], box2[0] + box2[2])
                    max_y = max(box1[1] + box1[3], box2[1] + box2[3])

                    boxes[i] = (min_x, min_y, max_x - min_x, max_y - min_y)
                    boxes.pop(j)
                    merged = True
                    j = i + 1
                else:
                    j += 1
            i += 1
            if merged:
                break
    return boxes

def detect_players_motion_only(fg_mask, court_polygon, min_area=300):
    if court_polygon is None: return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    processed_mask = cv2.erode(processed_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if cv2.pointPolygonTest(court_polygon, (x + w // 2, y + h // 2), False) < 0:
            continue
        if cv2.contourArea(c) < min_area:
            continue
        if w > h * 1.5 or h > w * 8:
            continue

        candidate_boxes.append((x, y, w, h))

    players = merge_overlapping_boxes(candidate_boxes, proximity_thresh=50)
    final_players = [box for box in players if (box[2] * box[3]) > 500]

    return final_players
def detect_ball(frame, hsv_frame, fg_mask, court_polygon):
    if court_polygon is None:
        return None

    LOWER_BALL = np.array([25, 80, 100])
    UPPER_BALL = np.array([40, 255, 255])
    color_mask = cv2.inRange(hsv_frame, LOWER_BALL, UPPER_BALL)

    contours_color, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_motion, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = contours_color + contours_motion

    best_candidate = None
    highest_score = -1

    # 🔐 Skorbord bölgesi (sol alt)
    SKORBOARD_REGION = (0, frame.shape[0] - 100, 200, 100)

    for c in all_contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 150:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        # 🎯 Skorbord kutusunu tamamen dışla
        if (SKORBOARD_REGION[0] <= x <= SKORBOARD_REGION[0] + SKORBOARD_REGION[2] and
            SKORBOARD_REGION[1] <= y <= SKORBOARD_REGION[1] + SKORBOARD_REGION[3]):
            continue

        if cv2.pointPolygonTest(court_polygon, (x + w // 2, y + h // 2), False) < 0:
            continue

        roi_motion_mask = fg_mask[y:y + h, x:x + w]
        motion_ratio = cv2.countNonZero(roi_motion_mask) / (w * h + 1e-6)
        if motion_ratio < 0.05:
            continue

        aspect_ratio = w / float(h) if h > 0 else 0
        shape_score = 1.0 - abs(1.0 - aspect_ratio)

        roi_color_mask = color_mask[y:y + h, x:x + w]
        color_ratio = cv2.countNonZero(roi_color_mask) / (w * h + 1e-6)
        if color_ratio < (np.mean(color_mask) / 255) * 0.5:
            continue

        final_score = shape_score * color_ratio * (motion_ratio + 0.1)

        if final_score > highest_score:
            highest_score = final_score
            best_candidate = ((x, y, w, h), (x + w // 2, y + h // 2))

    if highest_score > 0.15:
        return best_candidate
    else:
        return None


def main(debug=False):
    video_path = 'tennis.mp4'
    if not os.path.exists(video_path):
        print(f"Hata: Video dosyası bulunamadı -> {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Hata: Video dosyası açılamadı.")
        return

    print("Video başarıyla açıldı, işlem başlıyor...")

    target_width = 800
    ret, test_frame = cap.read()
    if not ret:
        print("İlk kare alınamadı.")
        return
    scale = target_width / test_frame.shape[1]
    target_height = int(test_frame.shape[0] * scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_width = target_width
    frame_height = target_height
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)

    last_known_corners = None
    prev_ball_center = None
    prev_time = None
    ball_tracker = None
    tracking_ball = False

    frame_counter = 0
    COURT_DETECTION_INTERVAL = 30
    window_name = 'Tenis Analizi'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break

        frame = cv2.resize(frame, (target_width, target_height))

        if frame_counter % COURT_DETECTION_INTERVAL == 0 or last_known_corners is None:
            h, w, _ = frame.shape
            roi_y_start, roi_y_end = int(h * 0.18), int(h * 0.96)
            roi_x_start, roi_x_end = int(w * 0.25), int(w * 0.90)
            court_roi_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            corners_in_roi = detect_court_in_roi(court_roi_frame)
            if corners_in_roi is not None:
                adjusted_corners = corners_in_roi.copy()
                adjusted_corners[:, 0] += roi_x_start
                adjusted_corners[:, 1] += roi_y_start
                last_known_corners = adjusted_corners

        frame_counter += 1

        try:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            fg_mask_raw = bg_subtractor.apply(frame, learningRate=0.01)
            _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)

            players = detect_players_motion_only(fg_mask, last_known_corners)
            ball = detect_ball(frame, hsv_frame, fg_mask, last_known_corners)

            if ball is not None:
                box, center = ball
                (x, y, w, h) = box
                ball_tracker = cv2.TrackerCSRT_create()
                ball_tracker.init(frame, (x, y, w, h))
                tracking_ball = True
            elif tracking_ball and ball_tracker is not None:
                ok, tracked_box = ball_tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, tracked_box)
                    center = (x + w // 2, y + h // 2)
                    ball = ((x, y, w, h), center)
                else:
                    tracking_ball = False
                    ball_tracker = None

            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
            final_display_frame = frame.copy()
            if last_known_corners is not None:
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [last_known_corners], (255, 255, 255))
                final_display_frame = np.where(mask == 255, frame, blurred_frame)

            if last_known_corners is not None:
                cv2.polylines(final_display_frame, [last_known_corners], True, (0, 255, 255), 3)

                # Perspektif düzeltme
                src_pts = np.float32(last_known_corners)
                dst_pts = np.float32([[0, 0], [400, 0], [400, 800], [0, 800]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (400, 800))
                if debug:
                    cv2.imshow("Kort Perspektif", warped)

            for i, p_box in enumerate(players):
                x, y, w, h = p_box
                cv2.rectangle(final_display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(final_display_frame, f'Oyuncu {i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ball is not None:
                box, center = ball
                cv2.circle(final_display_frame, center, 8, (255, 0, 255), -1)
                cv2.putText(final_display_frame, 'Top', (center[0] - 15, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Top hızı
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if prev_ball_center is not None and prev_time is not None:
                    dt = current_time - prev_time
                    dx = center[0] - prev_ball_center[0]
                    dy = center[1] - prev_ball_center[1]
                    speed = math.sqrt(dx ** 2 + dy ** 2) / dt
                    cv2.putText(final_display_frame, f"Hiz: {speed:.1f} px/s", (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                prev_ball_center = center
                prev_time = current_time

            if debug:
                cv2.imshow("Fg Mask", fg_mask)
                ball_mask = cv2.inRange(hsv_frame, np.array([25, 80, 100]), np.array([40, 255, 255]))
                cv2.imshow("Color Mask", ball_mask)

            out.write(final_display_frame)
            cv2.imshow(window_name, final_display_frame)

        except Exception as e:
            print(f"İşleme sırasında hata: {e}")
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")

if __name__ == "__main__":
    main(debug=True)
