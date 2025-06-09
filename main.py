import cv2
import numpy as np
import math

# --- ANA AYARLAR ---
INPUT_VIDEO_PATH = 'tennis.mp4'
BLUR_INTENSITY = 35  # Bulanıklık yoğunluğu. Tek sayı olmalı (örn: 25, 35, 45).

# --- OTOMATİK KORT TESPİTİ PARAMETRELERİ ---
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 80
HOUGH_MAX_LINE_GAP = 25

# --- OYUNCU VE TOP TESPİTİ PARAMETRELERİ ---
TOP_BG_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
BOTTOM_BG_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
MIN_PLAYER_AREA = 300
MAX_PLAYER_AREA = 8000
MIN_PLAYER_ASPECT_RATIO = 1.2

MIN_BALL_AREA = 35
LOWER_BALL_COLOR_HSV = np.array([28, 80, 130])
UPPER_BALL_COLOR_HSV = np.array([45, 255, 255])

# --- YARDIMCI FONKSİYONLAR ---
# (Bu fonksiyonlar önceki versiyonla aynı, değişiklik yok)
def find_line_intersection(line1, line2):
    x1, y1, x2, y2 = line1; x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if 0 < t < 1 and u > 0: return int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
    return None

def average_lines(lines, frame_height):
    if lines is None or len(lines) == 0: return None
    vx, vy, x, y = cv2.fitLine(np.reshape(lines, (-1, 2)), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = vx[0], vy[0], x[0], y[0]
    return [int(x - (y - 0) * vx / vy), 0, int(x - (y - frame_height) * vx / vy), frame_height]

def detect_court_automatically(frame):
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESHOLD, None, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    if lines is None: return None
    top_lines, bottom_lines, left_lines, right_lines = [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:
            if y1 < height * 0.65 and y2 < height * 0.65: top_lines.append(line)
            else: bottom_lines.append(line)
        elif abs(angle) > 40 and abs(angle) < 85:
            if angle > 0: right_lines.append(line)
            else: left_lines.append(line)
    avg_top = average_lines(top_lines, height)
    avg_bottom = average_lines(bottom_lines, height)
    avg_left = average_lines(left_lines, height)
    avg_right = average_lines(right_lines, height)
    if not all([avg_top, avg_bottom, avg_left, avg_right]): return None
    top_left = find_line_intersection(avg_top, avg_left)
    top_right = find_line_intersection(avg_top, avg_right)
    bottom_left = find_line_intersection(avg_bottom, avg_left)
    bottom_right = find_line_intersection(avg_bottom, avg_right)
    if not all([top_left, top_right, bottom_left, bottom_right]): return None
    return np.float32([top_left, top_right, bottom_right, bottom_left])

def create_court_schematic(width, height):
    schematic = np.zeros((height, width, 3), dtype=np.uint8)
    schematic[:] = (34, 98, 30); cv2.rectangle(schematic, (0, 0), (width-1, height-1), (255, 255, 255), 2)
    cv2.line(schematic, (0, int(height/2)), (width-1, int(height/2)), (255, 255, 255), 2)
    return schematic

def find_player_in_zone(frame, zone_mask, bg_subtractor):
    masked_frame = cv2.bitwise_and(frame, frame, mask=zone_mask)
    fg_mask = bg_subtractor.apply(masked_frame)
    fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)[1]
    fg_mask = cv2.dilate(fg_mask, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_PLAYER_AREA < area < MAX_PLAYER_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0 and h > 0:
                if h / float(w) > MIN_PLAYER_ASPECT_RATIO:
                    valid_candidates.append((area, (x, y, w, h)))
    if not valid_candidates: return None
    valid_candidates.sort(key=lambda item: item[0], reverse=True)
    return valid_candidates[0][1]

# --- ANA ÇALIŞMA FONKSİYONU ---
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened(): return
    ret, first_frame = cap.read()
    if not ret: return
        
    source_points = detect_court_automatically(first_frame)
    if source_points is None:
        print("Otomatik kort tespiti başarısız oldu.")
        return
    
    frame_height, frame_width, _ = first_frame.shape
    mid_left = ((source_points[0][0] + source_points[3][0]) / 2, (source_points[0][1] + source_points[3][1]) / 2)
    mid_right = ((source_points[1][0] + source_points[2][0]) / 2, (source_points[1][1] + source_points[2][1]) / 2)
    top_half_poly = np.array([source_points[0], source_points[1], mid_right, mid_left], dtype=np.int32)
    top_half_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(top_half_mask, [top_half_poly], 255)
    bottom_half_poly = np.array([mid_left, mid_right, source_points[2], source_points[3]], dtype=np.int32)
    bottom_half_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(bottom_half_mask, [bottom_half_poly], 255)
    full_court_mask = cv2.bitwise_or(top_half_mask, bottom_half_mask)
    
    schematic_width, schematic_height = 411, 894
    dest_points = np.float32([[0, 0], [schematic_width, 0], [schematic_width, schematic_height], [0, schematic_height]])
    homography_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_frame_gray = None
    
    while cap.isOpened():
        ret, frame_orig = cap.read() # Orijinal kareyi ayrı bir değişkende tut
        if not ret: break

        # --- YENİ: KORT DIŞINI BULANIKLAŞTIRMA ---
        # 1. Görüntünün tamamını bulanıklaştır
        blurred_frame = cv2.GaussianBlur(frame_orig, (BLUR_INTENSITY, BLUR_INTENSITY), 0)
        # 2. Kort maskesinin tersini al (kort dışı beyaz, içi siyah)
        inverse_court_mask = cv2.bitwise_not(full_court_mask)
        # 3. Sadece bulanık arka planı izole et
        background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inverse_court_mask)
        # 4. Sadece keskin kort alanını izole et
        foreground = cv2.bitwise_and(frame_orig, frame_orig, mask=full_court_mask)
        # 5. İkisini birleştirerek son görüntü karesini oluştur
        frame = cv2.add(background, foreground)
        
        # --- Bundan sonraki tüm işlemler bu yeni `frame` üzerinde yapılacak ---

        schematic = create_court_schematic(schematic_width, schematic_height)
        
        # Tespitler orijinal (bulanıklaştırılmamış) kare üzerinde yapılmalı
        top_player_bbox = find_player_in_zone(frame_orig, top_half_mask, TOP_BG_SUBTRACTOR)
        bottom_player_bbox = find_player_in_zone(frame_orig, bottom_half_mask, BOTTOM_BG_SUBTRACTOR)
        
        player_positions = []
        if top_player_bbox:
            x, y, w, h = top_player_bbox; cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            player_positions.append((x + w // 2, y + h))
        if bottom_player_bbox:
            x, y, w, h = bottom_player_bbox; cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            player_positions.append((x + w // 2, y + h))

        frame_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=full_court_mask)
        ball_position = None
        if prev_frame_gray is not None:
            frame_diff = cv2.absdiff(prev_frame_gray, masked_gray)
            _, diff_thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
            hsv = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, LOWER_BALL_COLOR_HSV, UPPER_BALL_COLOR_HSV)
            combined_mask = cv2.bitwise_and(diff_thresh, color_mask)
            contours_ball, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_ball:
                best_ball = max(contours_ball, key=cv2.contourArea)
                if cv2.contourArea(best_ball) > MIN_BALL_AREA:
                    (x, y), _ = cv2.minEnclosingCircle(best_ball)
                    ball_position = (int(x), int(y))
                    cv2.circle(frame, ball_position, 10, (0, 255, 255), 2)
        prev_frame_gray = masked_gray.copy()
        
        if player_positions:
            trans_points = cv2.perspectiveTransform(np.array([player_positions], dtype='float32'), homography_matrix)
            if trans_points is not None: [cv2.circle(schematic, (int(p[0]), int(p[1])), 12, (255, 255, 0), -1) for p in trans_points[0]]
        if ball_position:
            trans_point = cv2.perspectiveTransform(np.array([[ball_position]], dtype='float32'), homography_matrix)
            if trans_point is not None: cv2.circle(schematic, (int(trans_point[0][0][0]), int(trans_point[0][0][1])), 8, (0, 0, 255), -1)
        
        h, w, _ = frame.shape
        schematic_resized = cv2.resize(schematic, (int(w * 0.4), h))
        combined_view = np.hstack((frame, schematic_resized))
        cv2.imshow('Otomatik Tenis Takip Sistemi', combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()