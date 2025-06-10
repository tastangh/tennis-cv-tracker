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

# --- Diğer yardımcı fonksiyonlar (detect_court_in_roi, merge_overlapping_boxes, detect_players_motion_only, draw_grid) ---
# ... (Bu fonksiyonlar bir önceki cevapla aynı olduğu için yer kaplamaması adına kesilmiştir. Kodunuza aynen ekleyin.)
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
    return [box for box in players if (box[2] * box[3]) > 500]

def draw_grid(image, grid_shape=(12, 6)):
    h, w, _ = image.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(image, (x, 0), (x, h), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(image, (0, y), (w, y), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return image
# --- Diğer yardımcı fonksiyonların sonu ---


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

        # Adım 1 & 2: Kort Tespiti, Homografi ve Odaklanmış Maske
        # (Bu kısımlarda değişiklik yok)
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
            cv2.imshow("Tenis Analizi - Kort Aranıyor...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        dilated_corners = cv2.convexHull(last_known_corners)
        cv2.drawContours(play_area_mask, [dilated_corners], -1, 255, -1)
        play_area_mask = cv2.dilate(play_area_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)), iterations=1)
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        focused_frame = np.where(play_area_mask[:, :, None] == 255, frame, blurred_frame)

        # Adım 3: Arka Plan Çıkarma ve Oyuncu Tespiti
        fg_mask = bg_subtractor.apply(focused_frame, learningRate=0.005)
        player_detections = detect_players_motion_only(fg_mask, last_known_corners)
        
        # --- YENİ VE GELİŞTİRİLMİŞ TAKİP MANTIĞI ---

        # 1. Başlangıç Aşaması: Henüz 2 oyuncu izleyicimiz yoksa, yenilerini ekle.
        if len(player_trackers) < 2:
            for det in player_detections:
                if len(player_trackers) < 2:
                    player_trackers.append(KalmanTracker(det))
        
        # 2. Ana Takip Aşaması: Tam olarak 2 izleyicimiz varsa
        if len(player_trackers) == 2:
            # Önce her iki izleyici için de bir sonraki adımı TAHMİN ET.
            # Atamayı bu tahmin edilen konumlara göre yapacağız.
            for t in player_trackers:
                t.predict()

            # Eğer hiç tespit yoksa, bu karede kimseyi güncelleyemeyiz.
            # `predict` adımı zaten çalıştı, bu yüzden konumları hala güncel.
            if len(player_detections) > 0:
                # Olası tüm (izleyici, tespit) çiftlerinin mesafelerini hesapla
                pairings = []
                for i, tracker in enumerate(player_trackers):
                    for j, detection in enumerate(player_detections):
                        # Tahmin edilen kutu merkezi ile tespit edilen kutu merkezi arasındaki mesafe
                        tracker_pos = np.array(tracker.box[:2]) + np.array(tracker.box[2:]) / 2
                        detection_pos = np.array(detection[:2]) + np.array(detection[2:]) / 2
                        dist = np.linalg.norm(tracker_pos - detection_pos)
                        pairings.append((i, j, dist))
                
                # Çiftleri mesafeye göre sırala (en iyi eşleşmeler en başta olacak)
                pairings.sort(key=lambda x: x[2])
                
                # Atanan izleyicileri ve tespitleri takip etmek için setler kullan
                assigned_trackers_idx = set()
                assigned_detections_idx = set()
                
                # En iyi eşleşmeleri ata (greedy assignment)
                for tracker_idx, detection_idx, dist in pairings:
                    # Eğer bu izleyici ve tespit daha önce atanmadıysa
                    if tracker_idx not in assigned_trackers_idx and detection_idx not in assigned_detections_idx:
                        # Atamayı yap: izleyiciyi yeni tespit ile GÜNCELLE
                        player_trackers[tracker_idx].update(player_detections[detection_idx])
                        # Atandılar olarak işaretle
                        assigned_trackers_idx.add(tracker_idx)
                        assigned_detections_idx.add(detection_idx)
        
        # 3. Temizlik Aşaması: Uzun süredir görünmeyen izleyicileri kaldır.
        # Bu, bir oyuncu kaybolduğunda sistemin yeni bir oyuncu bulmasına olanak tanır.
        player_trackers = [t for t in player_trackers if t.consecutive_invisible_count < 25]

        # --- GÖRSELLEŞTİRME (Değişiklik yok) ---
        display_frame = frame.copy()
        cv2.addWeighted(cv2.cvtColor(play_area_mask, cv2.COLOR_GRAY2BGR), 0.3, display_frame, 0.7, 0, display_frame)

        top_down_court_grid = np.zeros((top_down_height, top_down_width, 3), dtype=np.uint8)
        top_down_court_grid = draw_grid(top_down_court_grid)

        # Oyuncuları y-koordinatına göre sıralayarak P1 ve P2 atamasını kararlı hale getir
        player_trackers.sort(key=lambda t: t.box[1]) 
        
        for i, tracker in enumerate(player_trackers):
            x, y, w, h = map(int, tracker.box)
            color = (0, 255, 0) if i == 0 else (255, 100, 0) # Oyunculara farklı renkler
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_frame, f"P{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if homography_matrix is not None:
                player_foot_point = np.array([[[x + w/2, y + h]]], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(player_foot_point, homography_matrix)
                if transformed_point is not None:
                    tx, ty = int(transformed_point[0][0][0]), int(transformed_point[0][0][1])
                    cv2.circle(top_down_court_grid, (tx, ty), 12, color, -1)
                    cv2.putText(top_down_court_grid, f"P{i+1}", (tx + 15, ty + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Tenis Analizi", display_frame)
        cv2.imshow("Kuş Bakışı Kort (Gridli)", top_down_court_grid)
        if debug:
            cv2.imshow("Odaklanmış Kare", focused_frame)
            cv2.imshow("Hareket Maskesi (Ham)", fg_mask)
        
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")

if __name__ == "__main__":
    # Önceki kodunuzdaki gibi, kesilen yardımcı fonksiyonları buraya veya yukarıya eklemeyi unutmayın.
    main(debug=True)