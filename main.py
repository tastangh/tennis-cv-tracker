#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Bu betik, bir tenis videosunu analiz ederek kortu, oyuncuları ve topu tespit eder.
- Kort tespiti: Görüntü işleme teknikleri ve Hough Çizgi Dönüşümü kullanılarak yapılır.
- Oyuncu tespiti: Arka plan çıkarma yöntemiyle elde edilen hareket maskesi üzerinden yapılır.
- Top tespiti: Hareket ve renk bilgilerini birleştiren bir puanlama sistemi ile yapılır.
- Görselleştirme: Tespit edilen kort alanı net bırakılırken, dışarısı bulanıklaştırılır.
"""

import numpy as np
import cv2 
import math
import os

# --- Sabitler ---

# Kort çizgileri için HSV renk aralığı (Yüksek parlaklık, düşük doygunluk)
# Not: Bu değişkenler mevcut kodda kullanılmıyor, ancak gelecekteki geliştirmeler için bırakılmıştır.
LOWER_COURT_LINES = np.array([0, 0, 180])
UPPER_COURT_LINES = np.array([180, 50, 255])

# --- Tespit Fonksiyonları ---

def detect_court_in_roi(roi_frame):
    """
    Verilen bir Görüntü Alanı (ROI) içinde tenis kortunun köşelerini tespit eder.
    Kontrast artırma, eşikleme ve Hough Çizgi Dönüşümü kullanır.
    """
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    # Kontrastı yerel olarak artıran CLAHE algoritması
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    # Beyaz çizgileri ayırmak için eşikleme
    _, line_mask = cv2.threshold(enhanced_gray, 210, 255, cv2.THRESH_BINARY)

    # Parametreleri ROI boyutuna göre dinamik olarak ayarla
    h, w = roi_frame.shape[:2]
    min_line_length = int(w * 0.10)  # Min. çizgi uzunluğu, ROI genişliğinin %10'u
    max_line_gap = int(w * 0.05)     # Maks. çizgi boşluğu, ROI genişliğinin %5'i
    hough_threshold = 30             # Hough dönüşümü için eşik değeri

    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, 
                            threshold=hough_threshold, 
                            minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)
    
    if lines is None: return None
    
    # Çizgileri yatay ve dikey olarak ayır
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45 or angle > 135: 
            horizontal.append(line)
        else: 
            vertical.append(line)
            
    if len(horizontal) < 2 or len(vertical) < 2: return None
    
    # Kortun sınırlarını oluşturacak en dış çizgileri bul
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

    # Köşe noktalarını bulmak için çizgilerin kesişimlerini hesapla
    p_top = get_line_params(top_line)
    p_bottom = get_line_params(bottom_line)
    p_left = get_line_params(left_line)
    p_right = get_line_params(right_line)
    
    tl = get_intersection(p_top, p_left)   # Sol üst
    tr = get_intersection(p_top, p_right)  # Sağ üst
    bl = get_intersection(p_bottom, p_left) # Sol alt
    br = get_intersection(p_bottom, p_right) # Sağ alt
    
    if not all([tl, tr, bl, br]): return None
    
    return np.array([tl, tr, br, bl], dtype=np.int32)

def merge_overlapping_boxes(boxes, proximity_thresh=75):
    """
    Birbirine yakın veya iç içe geçmiş sınırlayıcı kutuları birleştirir.
    """
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
                    # Birleştirme yapıldığında iç döngüyü yeniden başlat
                    j = i + 1 
                else:
                    j += 1
            i += 1
            if merged: # Eğer birleştirme olduysa, dış döngüyü en baştan başlat
                break
    return boxes

def detect_players_motion_only(fg_mask, court_polygon, min_area=300):
    """
    Sadece hareket maskesini kullanarak oyuncuları tespit eder. Renk bilgisi kullanmaz.
    """
    if court_polygon is None: return []

    # Gürültüyü temizle ve hareketli bölgeleri birleştir
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    processed_mask = cv2.erode(processed_mask, kernel, iterations=1)

    # Tüm hareketli adayların konturlarını bul
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Filtreleme: Kort içinde mi, yeterli alana sahip mi, şekli uygun mu?
        if cv2.pointPolygonTest(court_polygon, (x + w//2, y + h//2), False) < 0:
            continue
        if cv2.contourArea(c) < min_area:
            continue
        if w > h * 1.5 or h > w * 8: # Çok geniş veya çok ince nesneleri ele
            continue
            
        candidate_boxes.append((x, y, w, h))

    # Parçalı kutuları birleştir
    players = merge_overlapping_boxes(candidate_boxes, proximity_thresh=50)
    
    # Son alan filtresi
    final_players = [box for box in players if (box[2] * box[3]) > 500]
    
    return final_players

def detect_ball(frame, hsv_frame, fg_mask, court_polygon):
    """
    Hareket ve renk adaylarını birleştirip, ardından bir puanlama sistemi
    ile en olası topu seçen gelişmiş tespit fonksiyonu.
    """
    if court_polygon is None:
        return None

    # Renk maskesini oluştur (Tenis topu için sarı-yeşil tonları)
    LOWER_BALL = np.array([25, 80, 100])
    UPPER_BALL = np.array([40, 255, 255])
    color_mask = cv2.inRange(hsv_frame, LOWER_BALL, UPPER_BALL)
    
    # Hareket ve Renk konturlarını birleştir
    contours_color, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_motion, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = contours_color + contours_motion

    best_candidate = None
    highest_score = -1

    for c in all_contours:
        area = cv2.contourArea(c)
        if area < 15 or area > 350:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Kort dışında ise atla
        if cv2.pointPolygonTest(court_polygon, (x + w//2, y + h//2), False) < 0:
            continue
        
        # Puanlama için metrikleri hesapla
        aspect_ratio = w / float(h) if h > 0 else 0
        shape_score = 1.0 - abs(1.0 - aspect_ratio) # 1'e ne kadar yakınsa o kadar iyi

        roi_color_mask = color_mask[y:y+h, x:x+w]
        color_ratio = cv2.countNonZero(roi_color_mask) / (w * h + 1e-6)
        
        # Sert renk filtresi: Eğer adayda yeterince top rengi yoksa direkt atla
        if color_ratio < 0.1:
            continue

        roi_motion_mask = fg_mask[y:y+h, x:x+w]
        motion_ratio = cv2.countNonZero(roi_motion_mask) / (w * h + 1e-6)

        # Çarpımsal puanlama: Tüm kriterlerin iyi olması gerekir.
        final_score = shape_score * color_ratio * (motion_ratio + 0.1)

        if final_score > highest_score:
            highest_score = final_score
            best_candidate = ((x, y, w, h), (x + w // 2, y + h // 2))

    # Güven eşiği: Sadece yeterince yüksek puan alan adayı top olarak kabul et
    if highest_score > 0.15: 
        return best_candidate
    else:
        return None

# --- Ana İşlem Fonksiyonu ---

def main():
    """
    Ana fonksiyon: Videoyu açar, kare kare işler ve sonuçları ekranda gösterir.
    """
    # === DEĞİŞTİRİLECEK ALAN ===
    video_path = 'tennis.mp4' # VİDEO DOSYANIZIN YOLU
    # ==========================
    
    if not os.path.exists(video_path):
        print(f"Hata: Video dosyası bulunamadı -> {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Hata: Video dosyası açılamadı.")
        return
        
    print("Video başarıyla açıldı, işlem başlıyor...")
    
    # Arka plan çıkarıcıyı başlat
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
    
    last_known_corners = None
    window_name = 'Tenis Analizi'
    
    frame_counter = 0
    COURT_DETECTION_INTERVAL = 30 # Kortu her 30 karede bir yeniden tespit et (performans için)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("Video bitti veya kare okunamadı.")
            break

        # Kareyi yeniden boyutlandırarak işlem hızını artır
        target_width = 800
        scale = target_width / frame.shape[1]
        target_height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (target_width, target_height))
        
        # Kort tespiti periyodik olarak yapılır
        if frame_counter % COURT_DETECTION_INTERVAL == 0 or last_known_corners is None:
            h, w, _ = frame.shape
            # Kortun olabileceği bölgeyi (ROI) tanımla
            roi_y_start, roi_y_end = int(h * 0.18), int(h * 0.96)
            roi_x_start, roi_x_end = int(w * 0.25), int(w * 0.90)
            court_roi_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
            corners_in_roi = detect_court_in_roi(court_roi_frame)
            if corners_in_roi is not None:
                # ROI koordinatlarını ana çerçeve koordinatlarına dönüştür
                adjusted_corners = corners_in_roi.copy()
                adjusted_corners[:, 0] += roi_x_start
                adjusted_corners[:, 1] += roi_y_start
                last_known_corners = adjusted_corners
                
        frame_counter += 1

        # Görüntü işleme ve tespitler
        try:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            fg_mask_raw = bg_subtractor.apply(frame)
            _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)

            players = detect_players_motion_only(fg_mask, last_known_corners)
            ball = detect_ball(frame, hsv_frame, fg_mask, last_known_corners)
            
            # Kort dışını bulanıklaştırma efekti
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
            
            final_display_frame = frame.copy()
            if last_known_corners is not None:
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [last_known_corners], (255, 255, 255))
                # Maskeyi kullanarak orijinal ve bulanık çerçeveyi birleştir
                final_display_frame = np.where(mask == 255, frame, blurred_frame)

            # Tespitleri çizdirme
            if last_known_corners is not None:
                cv2.polylines(final_display_frame, [last_known_corners], isClosed=True, color=(0, 255, 255), thickness=3)

            for i, p_box in enumerate(players):
                x, y, w, h = p_box
                cv2.rectangle(final_display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(final_display_frame, f'Oyuncu {i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ball is not None:
                box, center = ball
                cv2.circle(final_display_frame, center, 8, (255, 0, 255), -1) 
                cv2.putText(final_display_frame, 'Top', (center[0] - 15, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow(window_name, final_display_frame)

        except Exception as e:
            print(f"İşleme sırasında bir hata oluştu: {e}")
            break
        
        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")


if __name__ == "__main__":
    main()