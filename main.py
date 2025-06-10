#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import os

# --- KalmanTracker Sınıfı ---
class KalmanTracker:
    """
    Kalman Filtresi kullanarak bir nesneyi takip eden sınıf.
    Konum ve hızı içeren 4 durumlu bir model (x, y, dx, dy) kullanır.
    """
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
        self.court_side = None

    def predict(self):
        """Nesnenin bir sonraki konumunu tahmin eder."""
        predicted_state = self.kf.predict()
        self.age += 1
        self.consecutive_invisible_count += 1
        predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
        w, h = self.box[2], self.box[3]
        self.box = (predicted_x - w//2, predicted_y - h//2, w, h)
        return self.box

    def update(self, box):
        """Yeni bir tespit ile Kalman filtresini günceller."""
        x, y, w, h = box
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        self.kf.correct(measurement)
        self.box = box
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0

# --- Yardımcı Fonksiyonlar ---

def detect_court_in_roi(roi_frame):
    """Verilen bir ROI içinde kort çizgilerinden yola çıkarak kort köşelerini bulur."""
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    _, line_mask = cv2.threshold(enhanced_gray, 210, 255, cv2.THRESH_BINARY)
    h, w = roi_frame.shape[:2]
    min_line_length, max_line_gap, hough_threshold = int(w * 0.10), int(w * 0.05), 30
    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None: return None
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45 or angle > 135: horizontal.append(line)
        else: vertical.append(line)
    if len(horizontal) < 2 or len(vertical) < 2: return None
    horizontal.sort(key=lambda l: (l[0][1] + l[0][3]) / 2)
    vertical.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)
    top_line, bottom_line, left_line, right_line = horizontal[0], horizontal[-1], vertical[0], vertical[-1]
    def get_line_params(line):
        (x1, y1, x2, y2) = line[0]; m = float('inf') if x1 == x2 else (y2 - y1) / (x2 - x1); c = x1 if m == float('inf') else y1 - m * x1; return m, c
    def get_intersection(p1, p2):
        m1, c1 = p1; m2, c2 = p2
        if m1 == m2: return None
        if m1 == float('inf'): return (int(c1), int(m2 * c1 + c2))
        if m2 == float('inf'): return (int(c2), int(m1 * c2 + c1))
        x = (c2 - c1) / (m1 - m2); y = m1 * x + c1; return (int(x), int(y))
    p_top, p_bottom, p_left, p_right = get_line_params(top_line), get_line_params(bottom_line), get_line_params(left_line), get_line_params(right_line)
    tl, tr = get_intersection(p_top, p_left), get_intersection(p_top, p_right)
    bl, br = get_intersection(p_bottom, p_left), get_intersection(p_bottom, p_right)
    if not all([tl, tr, bl, br]): return None
    return np.array([tl, tr, br, bl], dtype=np.int32)

def merge_overlapping_boxes(boxes, proximity_thresh=50):
    """Birbirine yakın veya üst üste binen kutuları birleştirir."""
    if len(boxes) == 0: return []
    merged = True
    while merged:
        merged = False; i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                box1, box2 = boxes[i], boxes[j]
                dist_x = max(0, max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2])); dist_y = max(0, max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3]))
                if dist_x < proximity_thresh and dist_y < proximity_thresh:
                    min_x, min_y = min(box1[0], box2[0]), min(box1[1], box2[1]); max_x, max_y = max(box1[0] + box1[2], box2[0] + box2[2]), max(box1[1] + box1[3], box2[1] + box2[3])
                    boxes[i] = (min_x, min_y, max_x - min_x, max_y - min_y); boxes.pop(j); merged = True; j = i + 1
                else: j += 1
            i += 1
            if merged: break
    return boxes

def detect_players_motion_only(fg_mask, court_polygon, min_area=300):
    """Verilen hareket maskesi üzerinde kontur analizi yaparak oyuncuları tespit eder."""
    if court_polygon is None: return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_mask = cv2.dilate(fg_mask, kernel, iterations=2); processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if cv2.pointPolygonTest(court_polygon, (x + w // 2, y + h // 2), False) < 0: continue
        if cv2.contourArea(c) < min_area or w > h * 1.5 or h > w * 8: continue
        candidate_boxes.append((x, y, w, h))
    if len(candidate_boxes) > 2:
        candidate_boxes.sort(key=lambda box: box[2] * box[3], reverse=True); candidate_boxes = candidate_boxes[:2]
    players = merge_overlapping_boxes(candidate_boxes, proximity_thresh=50)
    return [box for box in players if (box[2] * box[3]) > 500]

def draw_grid(image, grid_shape=(12, 6)):
    """Kuş bakışı görünüme grid çizer."""
    h, w, _ = image.shape; rows, cols = grid_shape; dy, dx = h / rows, w / cols
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1): cv2.line(image, (int(round(x)), 0), (int(round(x)), h), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1): cv2.line(image, (0, int(round(y))), (w, int(round(y))), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return image

def get_court_side(box, homography_matrix, court_center_y):
    """Bir oyuncunun kuş bakışı görünümde hangi sahada olduğunu belirler."""
    if homography_matrix is None: return None
    x, y, w, h = box; foot_point = np.array([[[x + w / 2, y + h]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(foot_point, homography_matrix)
    if transformed_point is None or transformed_point.size == 0: return None
    return "top" if transformed_point[0][0][1] < court_center_y else "bottom"

def get_top_down_coords(box, homography_matrix):
    """Bir oyuncunun kuş bakışı haritadaki koordinatlarını hesaplar."""
    if homography_matrix is None: return None, None
    x, y, w, h = map(int, box); foot_point = np.array([[[x + w/2, y + h]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(foot_point, homography_matrix)
    if transformed_point is not None and transformed_point.size > 0: return int(transformed_point[0][0][0]), int(transformed_point[0][0][1])
    return None, None
    
# --- Ana Fonksiyon ---
def main(debug=False):
    video_path = 'tennis.mp4'
    if not os.path.exists(video_path): print(f"Hata: '{video_path}' bulunamadı."); return
    cap = cv2.VideoCapture(video_path)
    target_width = 800
    ret, test_frame = cap.read()
    if not ret: print("Hata: Video dosyası okunamadı."); return
    scale = target_width / test_frame.shape[1]; target_height = int(test_frame.shape[0] * scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Ayarlanabilir Dikey Genişletme Miktarı (Oyuncuların servis alanı için)
    COURT_TOP_PADDING_PIXELS = 40

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    last_known_corners, player_trackers, homography_matrix, frame_counter = None, [], None, 0
    top_down_width, top_down_height = 400, 800
    dst_points = np.array([[0, 0], [top_down_width - 1, 0], [top_down_width - 1, top_down_height - 1], [0, top_down_height - 1]], dtype=np.float32)
    court_center_y_top_down = top_down_height / 2
    
    print("İşlem başlıyor... Çıkmak için 'q' tuşuna basın.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (target_width, target_height))

        # 1. KORT TESpitİ VE HOMOGRAFİ (Periyodik olarak güncellenir)
        if last_known_corners is None or frame_counter % 60 == 0:
            h, w, _ = frame.shape; roi = frame[int(h*0.2):, :]
            corners = detect_court_in_roi(roi)
            if corners is not None:
                corners[:, 1] += int(h*0.2); last_known_corners = corners
                # Homografi için her zaman hassas, orijinal köşeleri kullan
                sorted_corners = sorted(corners, key=lambda p: p[1])
                top_corners = sorted(sorted_corners[:2], key=lambda p: p[0]); bottom_corners = sorted(sorted_corners[2:], key=lambda p: p[0])
                src_points = np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
                homography_matrix, _ = cv2.findHomography(src_points, dst_points)

        if last_known_corners is None:
            cv2.putText(frame, "Kort Aranıyor...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2); cv2.imshow("Tenis Analizi", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 2. HAREKET TESPİT ALANI MASKESİ OLUŞTURMA
        # Tespit alanı için köşeleri dikey olarak yukarı doğru genişlet
        padded_corners = np.copy(last_known_corners)
        indices = np.argsort(padded_corners[:, 1])
        padded_corners[indices[:2], 1] -= COURT_TOP_PADDING_PIXELS
        
        play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(play_area_mask, [cv2.convexHull(padded_corners)], -1, 255, -1)

        # 3. HAREKET TESPİTİ
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask_raw = bg_subtractor.apply(blurred_frame, learningRate=0.005)
        fg_mask_court_only = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=play_area_mask)
        player_detections = detect_players_motion_only(fg_mask_court_only, cv2.convexHull(padded_corners))
        
        # 4. TAKİP MANTIĞI
        for t in player_trackers: t.predict()
        
        if homography_matrix is not None:
            # Oyuncuları ve tespitleri sahalarına göre grupla
            top_side_trackers, bottom_side_trackers = [], []; top_side_detections, bottom_side_detections = [], []
            for i, trk in enumerate(player_trackers):
                side = get_court_side(trk.box, homography_matrix, court_center_y_top_down)
                if side is not None: trk.court_side = side
                if trk.court_side == 'top': top_side_trackers.append((i, trk))
                elif trk.court_side == 'bottom': bottom_side_trackers.append((i, trk))
            for i, det in enumerate(player_detections):
                side = get_court_side(det, homography_matrix, court_center_y_top_down)
                if side == 'top': top_side_detections.append((i, det))
                elif side == 'bottom': bottom_side_detections.append((i, det))

            updated_trackers_indices, assigned_detections_indices = set(), set()
            # Her saha için ayrı ayrı eşleştirme yap
            def match_in_side(trackers, detections):
                if not trackers or not detections: return
                pairings = sorted([(np.linalg.norm(np.array(trk.box[:2]) - np.array(det[:2])), trk_idx, det_idx) for trk_idx, trk in trackers for det_idx, det in detections])
                for _, trk_idx, det_idx in pairings:
                    if trk_idx not in updated_trackers_indices and det_idx not in assigned_detections_indices:
                        player_trackers[trk_idx].update(player_detections[det_idx]); updated_trackers_indices.add(trk_idx); assigned_detections_indices.add(det_idx)
            
            match_in_side(top_side_trackers, top_side_detections); match_in_side(bottom_side_trackers, bottom_side_detections)

            # Eşleşmemiş tespitlerden yeni oyuncu ekle
            existing_sides = {t.court_side for t in player_trackers}
            unassigned_dets = [player_detections[i] for i in range(len(player_detections)) if i not in assigned_detections_indices]
            for det in unassigned_dets:
                if len(player_trackers) >= 2: break
                det_side = get_court_side(det, homography_matrix, court_center_y_top_down)
                if det_side and det_side not in existing_sides:
                    new_tracker = KalmanTracker(det); new_tracker.court_side = det_side; player_trackers.append(new_tracker); existing_sides.add(det_side)
        
        # Uzun süre görünmeyen takipçileri sil
        player_trackers = [t for t in player_trackers if t.consecutive_invisible_count < 25]

        # 5. GÖRSELLEŞTİRME
        display_frame = frame.copy()
        viz_mask = cv2.cvtColor(play_area_mask, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(viz_mask, 0.3, display_frame, 0.7, 0, display_frame)
        
        top_down_court_grid = np.zeros((top_down_height, top_down_width, 3), dtype=np.uint8); top_down_court_grid = draw_grid(top_down_court_grid)
        
        for tracker in player_trackers:
            x, y, w, h = map(int, tracker.box)
            color, label = ((0, 255, 0), "P_Top") if tracker.court_side == 'top' else (((255, 100, 0), "P_Bottom") if tracker.court_side == 'bottom' else ((0, 0, 255), "P_Unknown"))
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2); cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            tx, ty = get_top_down_coords(tracker.box, homography_matrix)
            if tx is not None: cv2.circle(top_down_court_grid, (tx, ty), 12, color, -1); cv2.putText(top_down_court_grid, label, (tx - 50, ty + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Tenis Analizi", display_frame); cv2.imshow("Kuş Bakışı Kort", top_down_court_grid)
        if debug: cv2.imshow("Hareket Maskesi (Genişletilmiş)", fg_mask_court_only)
        
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); print("İşlem tamamlandı.")

if __name__ == "__main__":
    main(debug=True)