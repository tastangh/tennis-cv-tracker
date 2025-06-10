#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import os

# KalmanTracker ve diğer yardımcı fonksiyonlar önceki kod bloğundan aynen alınır.
class KalmanTracker:
    def __init__(self, initial_box):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        x, y, w, h = initial_box
        self.kf.statePost = np.array([x + w/2, y + h/2, 0, 0], np.float32).T
        self.box = initial_box
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0

    def predict(self):
        predicted_state = self.kf.predict()
        self.age += 1
        if self.consecutive_invisible_count > 0:
            self.total_visible_count = 0
        self.consecutive_invisible_count += 1
        predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
        w, h = self.box[2], self.box[3]
        self.box = (predicted_x - w//2, predicted_y - h//2, w, h)
        return self.box

    def update(self, box):
        x, y, w, h = box
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        self.kf.correct(measurement)
        self.box = box
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0

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

def merge_overlapping_boxes(boxes, proximity_thresh=50):
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

def is_camera_moving(prev_gray, current_gray, motion_threshold=0.6):
    if prev_gray is None: return False, 0.0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitude)
    return mean_magnitude > motion_threshold, mean_magnitude

def get_detections(processed_mask):
    # Artık kort poligonuna gerek yok, çünkü maske zaten odaklanmış.
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_candidates, ball_candidates = [], []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0: continue
        aspect_ratio = w/h
        # Filtreleri biraz daha genel tutabiliriz, çünkü gürültü azaldı.
        if 800 < area < 20000 and aspect_ratio < 1.5 and h > w:
            player_candidates.append((x, y, w, h))
        elif 15 < area < 400 and 0.6 < aspect_ratio < 1.5:
            ball_candidates.append((x, y, w, h))
    return merge_overlapping_boxes(player_candidates), ball_candidates

def draw_grid(image, grid_shape=(12, 6)): # Daha geniş bir grid
    h, w, _ = image.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols
    # Dikey çizgiler
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(image, (x, 0), (x, h), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    # Yatay çizgiler
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(image, (0, y), (w, y), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return image

def main(debug=False):
    video_path = 'tennis.mp4'
    cap = cv2.VideoCapture(video_path)
    target_width = 800
    ret, test_frame = cap.read()
    if not ret: return
    scale = target_width / test_frame.shape[1]
    target_height = int(test_frame.shape[0] * scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)

    last_known_corners = None
    player_trackers = []
    homography_matrix = None
    frame_counter = 0

    top_down_width, top_down_height = 400, 800
    dst_points = np.array([[0, 0], [top_down_width - 1, 0], [top_down_width - 1, top_down_height - 1], [0, top_down_height - 1]], dtype=np.float32)
    
    print("İşlem başlıyor...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (target_width, target_height))

        # Adım 1: Kort Tespiti ve Homografi
        # (Kamera hareketi kontrolü basitlik için kaldırıldı, statik kamera varsayılıyor)
        if last_known_corners is None or frame_counter % 60 == 0:
            h, w, _ = frame.shape
            roi = frame[int(h*0.2):, :]
            corners = detect_court_in_roi(roi)
            if corners is not None:
                corners[:, 1] += int(h*0.2)
                last_known_corners = corners
                corners = sorted(corners, key=lambda p: p[1])
                top_corners = sorted(corners[:2], key=lambda p: p[0])
                bottom_corners = sorted(corners[2:], key=lambda p: p[0])
                src_points = np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
                homography_matrix, _ = cv2.findHomography(src_points, dst_points)

        if last_known_corners is None:
            cv2.imshow("Tenis Analizi", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # --- YENİ ADIM 2: ODAKLANMIŞ MASKELEME ---
        # 2a. Genişletilmiş bir "Oyun Alanı" maskesi oluştur
        play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # Kort poligonunu biraz genişlet (dilate)
        dilated_corners = cv2.convexHull(last_known_corners) # Dış bükey bir alan oluştur
        cv2.drawContours(play_area_mask, [dilated_corners], -1, 255, -1)
        # Morfolojik olarak daha da genişletelim
        play_area_mask = cv2.dilate(play_area_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)), iterations=1)

        # 2b. Orijinal karenin kort dışı alanlarını bulanıklaştır
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        # Oyun alanını orijinalden, geri kalanını bulanık olandan al
        focused_frame = np.where(play_area_mask[:, :, None] == 255, frame, blurred_frame)

        # 2c. Arka Plan Çıkarıcıyı bu odaklanmış kareye uygula
        fg_mask = bg_subtractor.apply(focused_frame, learningRate=0.005)
        
        # Adım 3: Hareket Maskesini İşle
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        processed_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # Adım 4 & 5: Tespit ve Takip (Kalman)
        player_detections, _ = get_detections(processed_mask) # Şimdilik sadece oyuncular
        
        if player_trackers:
            for t in player_trackers: t.predict()
        unmatched_detections_indices = list(range(len(player_detections)))
        for tracker in player_trackers:
            min_dist, best_match_idx = float('inf'), -1
            for j in unmatched_detections_indices:
                dist = np.linalg.norm(np.array(player_detections[j][:2]) - np.array(tracker.box[:2]))
                if dist < 120 and dist < min_dist: # Arama penceresini biraz genişlet
                    min_dist, best_match_idx = dist, j
            if best_match_idx != -1:
                tracker.update(player_detections[best_match_idx])
                unmatched_detections_indices.remove(best_match_idx)
        for idx in unmatched_detections_indices:
             if len(player_trackers) < 2: player_trackers.append(KalmanTracker(player_detections[idx]))
        player_trackers = [t for t in player_trackers if t.consecutive_invisible_count < 15]

        # Adım 6: Görselleştirme
        display_frame = frame.copy()
        cv2.addWeighted(cv2.cvtColor(play_area_mask, cv2.COLOR_GRAY2BGR), 0.3, display_frame, 0.7, 0, display_frame) # Oyun alanını göster

        top_down_court_grid = np.zeros((top_down_height, top_down_width, 3), dtype=np.uint8)
        top_down_court_grid = draw_grid(top_down_court_grid)

        visible_players = [t for t in player_trackers if t.total_visible_count > 3]
        visible_players.sort(key=lambda t: t.box[1])
        for i, tracker in enumerate(visible_players):
            x, y, w, h = map(int, tracker.box)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"P{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if homography_matrix is not None:
                player_foot_point = np.array([[[x + w/2, y + h]]], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(player_foot_point, homography_matrix)
                if transformed_point is not None:
                    tx, ty = int(transformed_point[0][0][0]), int(transformed_point[0][0][1])
                    cv2.circle(top_down_court_grid, (tx, ty), 12, (0, 255, 0), -1)
                    cv2.putText(top_down_court_grid, f"P{i+1}", (tx + 15, ty + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Tenis Analizi", display_frame)
        cv2.imshow("Kuş Bakışı Kort (Gridli)", top_down_court_grid)
        if debug:
            cv2.imshow("Odaklanmış Kare", focused_frame)
            cv2.imshow("Hareket Maskesi", processed_mask)
        
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")

if __name__ == "__main__":
    main(debug=True)