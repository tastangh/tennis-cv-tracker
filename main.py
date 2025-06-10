#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import os
from collections import deque

# --- KalmanTracker Sınıfı (Değişiklik Yok) ---
class KalmanTracker:
    def __init__(self, initial_box):
        self.kf = cv2.KalmanFilter(4, 2); self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03; self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        x, y, w, h = initial_box; self.kf.statePost = np.array([x + w/2, y + h/2, 0, 0], np.float32).T
        self.box = initial_box; self.age = 0; self.consecutive_invisible_count = 0; self.court_side = None
    def predict(self):
        predicted_state = self.kf.predict(); self.age += 1; self.consecutive_invisible_count += 1
        predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
        w, h = self.box[2], self.box[3]; self.box = (predicted_x - w//2, predicted_y - h//2, w, h); return self.box
    def update(self, box):
        x, y, w, h = box; measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        self.kf.correct(measurement); self.box = box; self.consecutive_invisible_count = 0

# --- Yardımcı Fonksiyonlar ---
def detect_court_in_roi(roi_frame):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)); enhanced_gray = clahe.apply(gray)
    _, line_mask = cv2.threshold(enhanced_gray, 210, 255, cv2.THRESH_BINARY)
    h, w = roi_frame.shape[:2]
    min_line_length, max_line_gap, hough_threshold = int(w * 0.10), int(w * 0.05), 30
    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None: return None
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]; angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45 or angle > 135: horizontal.append(line)
        else: vertical.append(line)
    if len(horizontal) < 2 or len(vertical) < 2: return None
    horizontal.sort(key=lambda l: (l[0][1] + l[0][3]) / 2); vertical.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)
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
    tl, tr, bl, br = get_intersection(p_top, p_left), get_intersection(p_top, p_right), get_intersection(p_bottom, p_left), get_intersection(p_bottom, p_right)
    if not all([tl, tr, bl, br]): return None
    return np.array([tl, tr, br, bl], dtype=np.int32)

# --- EN GÜÇLÜ GEOMETRİ KONTROL FONKSİYONU ---
def is_court_geometry_valid(corners, top_bottom_ratio_range, aspect_ratio_range, vertical_line_angle_range, side_height_ratio_range):
    if corners is None or len(corners) != 4: return False
    
    sorted_y = sorted(corners, key=lambda p: p[1])
    top_corners = sorted(sorted_y[:2], key=lambda p: p[0]); bottom_corners = sorted(sorted_y[2:], key=lambda p: p[0])
    tl, tr = top_corners[0], top_corners[1]
    bl, br = (bottom_corners[0], bottom_corners[1]) if bottom_corners[0][0] < bottom_corners[1][0] else (bottom_corners[1], bottom_corners[0])

    top_width = np.linalg.norm(tl - tr); bottom_width = np.linalg.norm(bl - br)
    left_height = np.linalg.norm(tl - bl); right_height = np.linalg.norm(tr - br)
    
    if bottom_width < 1 or left_height < 1 or right_height < 1: return False
    
    # 1. Oran Kontrolü
    top_bottom_ratio = top_width / bottom_width
    if not (top_bottom_ratio_range[0] < top_bottom_ratio < top_bottom_ratio_range[1]): return False
        
    # 2. En-Boy Oranı Kontrolü
    avg_height = (left_height + right_height) / 2
    aspect_ratio = bottom_width / avg_height
    if not (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]): return False
    
    # 3. Dikey Çizgi Açı Kontrolü
    def get_angle(p1, p2): return abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    angle_left, angle_right = get_angle(bl, tl), get_angle(br, tr)
    min_angle, max_angle = vertical_line_angle_range
    if not (min_angle < angle_left < max_angle and min_angle < angle_right < max_angle): return False
        
    # 4. YENİ KONTROL: Dikey Kenar Simetri Kontrolü
    height_ratio = left_height / right_height
    min_h_ratio, max_h_ratio = side_height_ratio_range
    if not (min_h_ratio < height_ratio < max_h_ratio): return False
        
    return True

# --- Diğer Yardımcı Fonksiyonlar (Değişiklik Yok) ---
def merge_overlapping_boxes(boxes, proximity_thresh=50):
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
def get_court_side(box, homography_matrix, court_center_y):
    if homography_matrix is None: return None
    x, y, w, h = box; foot_point = np.array([[[x + w / 2, y + h]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(foot_point, homography_matrix)
    if transformed_point is None or transformed_point.size == 0: return None
    return "top" if transformed_point[0][0][1] < court_center_y else "bottom"
def get_valid_contours(fg_mask, min_area, max_area):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
def is_point_inside_bbox(point, bbox):
    x_p, y_p = point; x_b, y_b, w_b, h_b = bbox
    return x_b < x_p < x_b + w_b and y_b < y_p < y_b + h_b
def detect_ball(fg_mask, last_known_ball_center, ball_lost_counter, court_boundary_points, player_bboxes, params):
    if last_known_ball_center and ball_lost_counter < params['MAX_BALL_LOST_FRAMES']:
        search_mask = np.zeros_like(fg_mask)
        sx = int(last_known_ball_center[0] - params['SEARCH_REGION_PADDING']); sy = int(last_known_ball_center[1] - params['SEARCH_REGION_PADDING'])
        sw = int(params['SEARCH_REGION_PADDING'] * 2); sh = int(params['SEARCH_REGION_PADDING'] * 2)
        cv2.rectangle(search_mask, (sx, sy), (sx + sw, sy + sh), 255, -1); search_area = cv2.bitwise_and(fg_mask, fg_mask, mask=search_mask)
    else: search_area = fg_mask
    contours = get_valid_contours(search_area, params['MIN_BALL_CONTOUR_AREA'], params['MAX_BALL_CONTOUR_AREA'])
    potential_balls = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (params['BALL_MIN_WIDTH_HEIGHT'] <= w <= params['BALL_MAX_WIDTH_HEIGHT'] and params['BALL_MIN_WIDTH_HEIGHT'] <= h <= params['BALL_MAX_WIDTH_HEIGHT']): continue
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (params['BALL_MIN_ASPECT_RATIO'] <= aspect_ratio <= params['BALL_MAX_ASPECT_RATIO']): continue
        area = cv2.contourArea(cnt)
        if len(cnt) >= 5:
            if (float(area) / cv2.contourArea(cv2.convexHull(cnt))) < params['BALL_MIN_SOLIDITY']: continue
        center_candidate = (x + w // 2, y + h // 2)
        if cv2.pointPolygonTest(court_boundary_points, center_candidate, False) < 0: continue
        if any(is_point_inside_bbox(center_candidate, p_bbox) for p_bbox in player_bboxes): continue
        potential_balls.append({'bbox': (x, y, w, h), 'center': center_candidate, 'area': area})
    if not potential_balls: return None
    best_ball = None
    if last_known_ball_center and ball_lost_counter < params['MAX_BALL_LOST_FRAMES']:
        min_dist = float('inf')
        for ball in potential_balls:
            dist = np.linalg.norm(np.array(last_known_ball_center) - np.array(ball['center']))
            if dist < params['MAX_BALL_JUMP_DISTANCE'] and dist < min_dist: min_dist = dist; best_ball = ball
    if best_ball is None:
        potential_balls.sort(key=lambda b: b['area'], reverse=True); best_ball = potential_balls[0]
    return (best_ball['bbox'], best_ball['center'])
def transform_points_for_sketch(points_list, H_matrix):
    if not points_list or H_matrix is None: return []
    np_points = np.array([[p[0], p[1]] for p in points_list], dtype=np.float32)
    if np_points.shape[0] == 0: return []
    transformed_points = cv2.perspectiveTransform(np.array([np_points]), H_matrix)
    if transformed_points is None: return []
    return transformed_points[0]
def draw_court_sketch(base_sketch_img, homography_matrix, player_trackers, ball_trail_video, current_frame_count, viz_params):
    sketch_display = base_sketch_img.copy()
    if homography_matrix is None: return sketch_display
    if ball_trail_video:
        ball_points_to_transform = [pos for pos, fc in ball_trail_video]
        transformed_ball_trail = transform_points_for_sketch(ball_points_to_transform, homography_matrix)
        for i, (original_pos, fc) in enumerate(ball_trail_video):
            if i < len(transformed_ball_trail):
                t_point_ball = tuple(map(int, transformed_ball_trail[i]))
                age = current_frame_count - fc
                fade_factor = max(0.1, 1.0 - (age / float(viz_params['BALL_FADE_DURATION_SKETCH'])))
                faded_color_np = (np.array(viz_params['ball_viz_color'], dtype=np.float32) * fade_factor).astype(np.uint8)
                faded_color_tuple = tuple(faded_color_np.tolist())
                radius = 5 if age < 2 else (4 if age < viz_params['SKETCH_BALL_HISTORY_LEN'] // 2 else 3)
                cv2.circle(sketch_display, t_point_ball, radius, faded_color_tuple, -1)
    player_points_to_transform, player_colors, player_radii = [], [], []
    for tracker in player_trackers:
        x, y, w, h = tracker.box; player_points_to_transform.append((x + w/2, y + h))
        is_lost = tracker.consecutive_invisible_count > 0; pid = 0 if tracker.court_side == 'top' else 1
        player_colors.append(viz_params['player_viz_colors_lost'][pid] if is_lost else viz_params['player_viz_colors'][pid])
        player_radii.append(6 if is_lost else 7)
    if player_points_to_transform:
        transformed_player_points = transform_points_for_sketch(player_points_to_transform, homography_matrix)
        for i, t_point_player in enumerate(transformed_player_points):
            t_point_player_int = tuple(map(int, t_point_player))
            cv2.circle(sketch_display, t_point_player_int, player_radii[i], player_colors[i], -1)
    return sketch_display
def create_combined_view(main_frame, motion_mask, sketch_view, debug_mode, is_active):
    h, w, _ = main_frame.shape
    if debug_mode and is_active and motion_mask is not None: bottom_panel = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    else:
        bottom_panel = np.zeros_like(main_frame)
        status = "Analiz Aktif" if is_active else "Analiz Duraklatildi"; cv2.putText(bottom_panel, status, (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bottom_panel, "Hareket Maskesi", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    left_column = cv2.vconcat([main_frame, bottom_panel])
    if sketch_view is not None:
        target_h = left_column.shape[0]; target_w = int(sketch_view.shape[1] * (target_h / sketch_view.shape[0]))
        right_column = cv2.resize(sketch_view, (target_w, target_h))
    else:
        target_h = left_column.shape[0]; target_w = int(target_h / 2.2)
        right_column = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cv2.putText(right_column, "Kroki Yok", (target_w//2 - 50, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.hconcat([left_column, right_column])

# --- Ana Fonksiyon ---
def main(debug=False):
    video_path = 'tennis.mp4'
    if not os.path.exists(video_path): print(f"Hata: '{video_path}' bulunamadı."); return
    cap = cv2.VideoCapture(video_path)
    target_width = 800; ret, test_frame = cap.read()
    if not ret: print("Hata: Video dosyası okunamadı."); return
    scale = target_width / test_frame.shape[1]; target_height = int(test_frame.shape[0] * scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- AYARLANABİLİR PARAMETRELER ---
    TOP_BOTTOM_RATIO_RANGE = (0.3, 0.98)
    ASPECT_RATIO_RANGE = (1.0, 2.5)
    VERTICAL_LINE_ANGLE_RANGE = (60, 120)
    SIDE_HEIGHT_RATIO_RANGE = (0.75, 1.25) # YENİ PARAMETRE: Sol/Sağ yükseklik oranı
    INVALID_VIEW_THRESHOLD = 5
    COURT_TOP_PADDING_PIXELS = 30; COURT_HORIZONTAL_PADDING_PIXELS = 20
    ball_params = { 'MIN_BALL_CONTOUR_AREA': 8, 'MAX_BALL_CONTOUR_AREA': 100, 'BALL_MIN_WIDTH_HEIGHT': 3, 'BALL_MAX_WIDTH_HEIGHT': 25, 'BALL_MIN_ASPECT_RATIO': 0.7, 'BALL_MAX_ASPECT_RATIO': 1.4, 'BALL_MIN_SOLIDITY': 0.75, 'MAX_BALL_LOST_FRAMES': 10, 'MAX_BALL_JUMP_DISTANCE': 80, 'SEARCH_REGION_PADDING': 50 }
    viz_params = { 'player_viz_colors': {0:(0,100,255), 1:(255,100,0)}, 'player_viz_colors_lost': {0:(0,60,150), 1:(150,60,0)}, 'ball_viz_color':(0,255,255), 'SKETCH_BALL_HISTORY_LEN': 7, 'BALL_FADE_DURATION_SKETCH': 12 }
    
    KROKI_IMAGE_PATH = "kort.png"; TARGET_SKETCH_WIDTH, TARGET_SKETCH_HEIGHT = 250, 430
    base_sketch_resized = None
    if os.path.exists(KROKI_IMAGE_PATH): base_sketch_resized = cv2.resize(cv2.imread(KROKI_IMAGE_PATH), (TARGET_SKETCH_WIDTH, TARGET_SKETCH_HEIGHT))
    else: print(f"UYARI: Kroki resmi '{KROKI_IMAGE_PATH}' bulunamadı.")
    
    margin_x_sketch, margin_y_sketch = 25, 35
    DST_POINTS_FOR_SKETCH = np.array([[margin_x_sketch, margin_y_sketch], [TARGET_SKETCH_WIDTH - margin_x_sketch, margin_y_sketch], [TARGET_SKETCH_WIDTH - margin_x_sketch, TARGET_SKETCH_HEIGHT - margin_y_sketch], [margin_x_sketch, TARGET_SKETCH_HEIGHT - margin_y_sketch]], dtype=np.float32)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    
    last_known_corners, is_court_view_active, invalid_view_counter, frame_counter = None, True, 0, 0
    player_trackers, homography_matrix, homography_matrix_for_sketch = [], None, None
    last_known_ball_center, ball_lost_counter = None, 0
    ball_trail_video = deque(maxlen=viz_params['SKETCH_BALL_HISTORY_LEN'])

    top_down_width_logic, top_down_height_logic = 400, 800
    dst_points_for_logic = np.array([[0, 0], [top_down_width_logic-1, 0], [top_down_width_logic-1, top_down_height_logic-1], [0, top_down_height_logic-1]], dtype=np.float32)
    court_center_y_top_down = top_down_height_logic / 2
    
    status_text, status_color = "", (0, 0, 0)
    
    print("İşlem başlıyor... Çıkmak için 'q' tuşuna basın.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (target_width, target_height)); frame_counter += 1

        h, w, _ = frame.shape; roi_for_check = frame[int(h*0.2):, :]
        current_corners_relative = detect_court_in_roi(roi_for_check)
        is_geometry_ok = False
        if current_corners_relative is not None:
            current_corners_absolute = current_corners_relative + [0, int(h*0.2)]
            # --- GÜÇLENDİRİLMİŞ KONTROL ÇAĞRISI ---
            if is_court_geometry_valid(current_corners_absolute, TOP_BOTTOM_RATIO_RANGE, ASPECT_RATIO_RANGE, VERTICAL_LINE_ANGLE_RANGE, SIDE_HEIGHT_RATIO_RANGE):
                is_geometry_ok = True
        if is_geometry_ok:
            invalid_view_counter = 0; is_court_view_active = True
            last_known_corners = current_corners_absolute
            sorted_corners = sorted(last_known_corners, key=lambda p: p[1])
            top_corners = sorted(sorted_corners[:2], key=lambda p: p[0]); bottom_corners = sorted(sorted_corners[2:], key=lambda p: p[0])
            src_points = np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
            homography_matrix, _ = cv2.findHomography(src_points, dst_points_for_logic)
            homography_matrix_for_sketch, _ = cv2.findHomography(src_points, DST_POINTS_FOR_SKETCH)
        else: invalid_view_counter += 1
        if invalid_view_counter > INVALID_VIEW_THRESHOLD: is_court_view_active = False

        if is_court_view_active: status_text, status_color = "Kort Tam Istenen Acida", (0, 255, 0)
        else: status_text, status_color = "Kort Aci Kontrolu Basarisiz (Replay?)", (0, 0, 255)

        detected_ball_info, fg_mask_court_only = None, None
        if is_court_view_active and last_known_corners is not None:
            padded_corners = np.copy(last_known_corners); y_indices = np.argsort(padded_corners[:, 1]); x_indices = np.argsort(padded_corners[:, 0])
            padded_corners[y_indices[:2], 1] -= COURT_TOP_PADDING_PIXELS; padded_corners[x_indices[:2], 0] -= COURT_HORIZONTAL_PADDING_PIXELS; padded_corners[x_indices[2:], 0] += COURT_HORIZONTAL_PADDING_PIXELS
            court_polygon = cv2.convexHull(padded_corners)
            play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8); cv2.drawContours(play_area_mask, [court_polygon], -1, 255, -1)
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            fg_mask_raw = bg_subtractor.apply(blurred_frame, learningRate=0.005)
            fg_mask_court_only = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=play_area_mask)
            player_detections = detect_players_motion_only(fg_mask_court_only, court_polygon)
            for t in player_trackers: t.predict()
            if homography_matrix is not None:
                top_side_trackers, bottom_side_trackers, top_side_detections, bottom_side_detections = [],[],[],[]
                for i, trk in enumerate(player_trackers):
                    side = get_court_side(trk.box, homography_matrix, court_center_y_top_down)
                    if side: trk.court_side = side
                    if trk.court_side == 'top': top_side_trackers.append((i, trk))
                    else: bottom_side_trackers.append((i, trk))
                for i, det in enumerate(player_detections):
                    side = get_court_side(det, homography_matrix, court_center_y_top_down)
                    if side == 'top': top_side_detections.append((i, det))
                    else: bottom_side_detections.append((i, det))
                updated_trackers_indices, assigned_detections_indices = set(), set()
                def match_in_side(trackers, detections):
                    if not trackers or not detections: return
                    pairings = sorted([(np.linalg.norm(np.array(trk.box[:2]) - np.array(det[:2])), trk_idx, det_idx) for trk_idx, trk in trackers for det_idx, det in detections])
                    for _, trk_idx, det_idx in pairings:
                        if trk_idx not in updated_trackers_indices and det_idx not in assigned_detections_indices:
                            player_trackers[trk_idx].update(player_detections[det_idx]); updated_trackers_indices.add(trk_idx); assigned_detections_indices.add(det_idx)
                match_in_side(top_side_trackers, top_side_detections); match_in_side(bottom_side_trackers, bottom_side_detections)
                existing_sides = {t.court_side for t in player_trackers}
                unassigned_dets = [player_detections[i] for i in range(len(player_detections)) if i not in assigned_detections_indices]
                for det in unassigned_dets:
                    if len(player_trackers) >= 2: break
                    det_side = get_court_side(det, homography_matrix, court_center_y_top_down)
                    if det_side and det_side not in existing_sides: new_tracker = KalmanTracker(det); new_tracker.court_side = det_side; player_trackers.append(new_tracker); existing_sides.add(det_side)
            player_bboxes = [t.box for t in player_trackers]
            detected_ball_info = detect_ball(fg_mask_court_only, last_known_ball_center, ball_lost_counter, cv2.convexHull(last_known_corners), player_bboxes, ball_params)
            if detected_ball_info: last_known_ball_center = detected_ball_info[1]; ball_lost_counter = 0; ball_trail_video.append((last_known_ball_center, frame_counter))
            else:
                ball_lost_counter += 1
                if ball_lost_counter > ball_params['MAX_BALL_LOST_FRAMES']: last_known_ball_center = None; ball_trail_video.clear()
        else:
            ball_lost_counter += 1
            if ball_lost_counter > ball_params['MAX_BALL_LOST_FRAMES']: last_known_ball_center = None; ball_trail_video.clear()
        player_trackers = [t for t in player_trackers if t.consecutive_invisible_count < 50]

        display_frame = frame.copy()
        sketch_display = None
        if base_sketch_resized is not None:
             sketch_display = draw_court_sketch(base_sketch_resized, homography_matrix_for_sketch, player_trackers if is_court_view_active else [], ball_trail_video, frame_counter, viz_params)
        if is_court_view_active and last_known_corners is not None:
            padded_corners_viz = np.copy(last_known_corners); y_indices_viz = np.argsort(padded_corners_viz[:, 1]); x_indices_viz = np.argsort(padded_corners_viz[:, 0])
            padded_corners_viz[y_indices_viz[:2], 1] -= COURT_TOP_PADDING_PIXELS; padded_corners_viz[x_indices_viz[:2], 0] -= COURT_HORIZONTAL_PADDING_PIXELS; padded_corners_viz[x_indices_viz[2:], 0] += COURT_HORIZONTAL_PADDING_PIXELS
            viz_mask = np.zeros(frame.shape[:2], dtype=np.uint8); cv2.drawContours(viz_mask, [cv2.convexHull(padded_corners_viz)], -1, 255, -1)
            viz_mask_bgr = cv2.cvtColor(viz_mask, cv2.COLOR_GRAY2BGR); cv2.addWeighted(viz_mask_bgr, 0.3, display_frame, 0.7, 0, display_frame)
            
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
        if is_court_view_active:
            player_trackers.sort(key=lambda t: 0 if t.court_side == 'top' else 1)
            for i, tracker in enumerate(player_trackers):
                x, y, w_box, h_box = map(int, tracker.box); pid = 0 if tracker.court_side == 'top' else 1
                color = viz_params['player_viz_colors'][pid]; label = "P_Top" if tracker.court_side == 'top' else "P_Bottom"
                cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), color, 2); cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if detected_ball_info:
                _, ball_center = detected_ball_info; cv2.circle(display_frame, ball_center, 8, viz_params['ball_viz_color'], -1)

        combined_view = create_combined_view(display_frame, fg_mask_court_only, sketch_display, debug, is_court_view_active)
        cv2.imshow("Tenis Analizi - Birlesik Gorunum", combined_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); print("İşlem tamamlandı.")

if __name__ == "__main__":
    main(debug=True)