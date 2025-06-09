import cv2
import numpy as np
import sys
import math

# --- 1. AYARLAR VE GLOBAL DEĞİŞKENLER ---

# Video dosyasının yolu
VIDEO_PATH = "tennis_video.mp4" 
DEBUG = False # Otomatik kort tespiti adımlarını görmek için True yapın

# Genişletilmiş odak alanı için görsel dolgu miktarı (piksel)
VISUAL_PADDING = 135

# --- Otomatik Kort Tespiti Parametreleri ---
# Videodaki kortun rengine göre bu HSV aralığını ayarlayın (şu an mavi tonlar için)
# Yeşil kortlar için: (35, 50, 50) - (85, 255, 255)a
COURT_HSV_LOWER = np.array([95, 100, 50])
COURT_HSV_UPPER = np.array([125, 255, 255])
MORPH_KERNEL_SIZE = 15 # Kort maskesindeki delikleri kapatmak için kullanılacak kernel boyutu

# --- Nesne Tespiti Parametreleri ---
# Dinamik alan eşiği için parametreler
PLAYER_MIN_AREA_FAR = 300      # Uzaktaki (üstteki) oyuncu için minimum alan
PLAYER_MIN_AREA_CLOSE = 700    # Yakındaki (alttaki) oyuncu için minimum alan
PLAYER_MAX_AREA = 6000         # Maksimum alan sabit kalabilir

# Top için parametreler
BALL_MIN_AREA = 40
BALL_MAX_AREA = 300
BALL_HSV_LOWER = (25, 80, 100)
BALL_HSV_UPPER = (45, 255, 255)

# --- Replay Tespiti ---
# Kort alanının % kaçının aniden hareketlenmesi durumunda sahne değişimi kabul edilecek
SCENE_CHANGE_THRESHOLD = 0.60

# --- Perspektif Dönüşümü için Global Değişkenler ---
width, height = 450, 780
destination_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
homography_matrix = None

# --- 2. YARDIMCI FONKSİYONLAR ---

def find_court_corners_auto(frame):
    """Renk segmentasyonu ve kontur analizi kullanarak kort köşelerini otomatik olarak bulur."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COURT_HSV_LOWER, COURT_HSV_UPPER)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    if DEBUG: cv2.imshow("Debug - Kort Renk Maskesi", mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Hata: Kort konturu bulunamadı. HSV renk aralığını kontrol edin.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    sum_ = largest_contour.sum(axis=2)
    diff_ = np.diff(largest_contour, axis=2)
    
    top_left = largest_contour[np.argmin(sum_)]
    bottom_right = largest_contour[np.argmax(sum_)]
    top_right = largest_contour[np.argmax(diff_)]
    bottom_left = largest_contour[np.argmin(diff_)]
    
    court_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32).reshape(4, 2)
    return court_points

def expand_court_polygon(points, padding):
    """Verilen 4 köşe noktasını dışa doğru 'padding' miktarı kadar ittirerek yeni bir poligon oluşturur."""
    tl, tr, br, bl = points[0], points[1], points[2], points[3]
    vec_tl_tr = tl - tr; vec_tl_bl = tl - bl
    vec_tr_tl = tr - tl; vec_tr_br = tr - br
    vec_bl_tl = bl - tl; vec_bl_br = bl - br
    vec_br_tr = br - tr; vec_br_bl = br - bl

    def normalize(v): return v / (np.linalg.norm(v) + 1e-6)

    new_tl = tl + (normalize(vec_tl_tr) + normalize(vec_tl_bl)) * padding
    new_tr = tr + (normalize(vec_tr_tl) + normalize(vec_tr_br)) * padding
    new_bl = bl + (normalize(vec_bl_tl) + normalize(vec_bl_br)) * padding
    new_br = br + (normalize(vec_br_tr) + normalize(vec_br_bl)) * padding

    return np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)

def create_court_mask(frame, court_points):
    """Verilen köşe noktalarından bir poligon maskesi oluşturur."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(court_points)], (255))
    return mask

def apply_blur_outside_mask(frame, mask):
    """Maskenin dışındaki alanı blurlar."""
    blurred_frame = cv2.GaussianBlur(frame, (31, 31), 0)
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inv_mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    return cv2.add(background, foreground)

def get_dynamic_player_min_area(y_pos, y_top, y_bottom):
    """Bir konturun dikey konumuna göre olması gereken min alanı lineer olarak hesaplar."""
    if y_bottom == y_top: return PLAYER_MIN_AREA_CLOSE
    relative_pos = np.clip((y_pos - y_top) / (y_bottom - y_top), 0, 1)
    return PLAYER_MIN_AREA_FAR + relative_pos * (PLAYER_MIN_AREA_CLOSE - PLAYER_MIN_AREA_FAR)

def get_best_candidates(contours, frame, tight_court_points):
    """En iyi oyuncu ve top adaylarını seçer. Oyuncular konum ve dinamik alana göre filtrelenir."""
    player_candidates, ball_candidates = [], []
    
    y_top_court = min(tight_court_points[0][1], tight_court_points[1][1])
    y_bottom_court = max(tight_court_points[2][1], tight_court_points[3][1])
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # --- Oyuncu Tespiti ---
        dynamic_min_area = get_dynamic_player_min_area(y + h, y_top_court, y_bottom_court)
        player_pos = (x + w // 2, y + h)
        is_on_court = cv2.pointPolygonTest(tight_court_points, player_pos, False) >= 0

        if dynamic_min_area < area < PLAYER_MAX_AREA and is_on_court:
            player_candidates.append({'area': area, 'bbox': (x, y, w, h), 'pos': player_pos})

        # --- Top Tespiti ---
        if BALL_MIN_AREA < area < BALL_MAX_AREA:
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_roi, BALL_HSV_LOWER, BALL_HSV_UPPER)
            color_ratio = cv2.countNonZero(color_mask) / ((w * h) + 1e-6)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * math.pi * area / (perimeter**2)
            if color_ratio > 0.4 and circularity > 0.5:
                score = (color_ratio * 0.6) + (circularity * 0.4)
                ball_pos = (x + w // 2, y + h // 2)
                ball_candidates.append({'score': score, 'bbox': (x, y, w, h), 'pos': ball_pos})
    
    player_candidates.sort(key=lambda p: p['area'], reverse=True)
    ball_candidates.sort(key=lambda b: b['score'], reverse=True)
    
    return player_candidates[:2], ball_candidates[0] if ball_candidates else None

def draw_2d_court(players_coords, ball_coord):
    """Kuş bakışı 2D kort krokisini ve nesnelerin konumlarını çizer."""
    court_2d = np.zeros((height + 50, width + 50, 3), dtype=np.uint8)
    court_2d[:, :] = (34, 139, 34) # Orman yeşili
    court_area = (25, 25)

    cv2.rectangle(court_2d, court_area, (width + court_area[0], height + court_area[1]), (255, 255, 255), 2)
    cv2.line(court_2d, (court_area[0], court_area[1] + height // 2), (width + court_area[0], court_area[1] + height // 2), (255, 255, 255), 2)
    
    if homography_matrix is not None:
        if players_coords:
            player_points_2d = cv2.perspectiveTransform(np.float32(players_coords).reshape(-1, 1, 2), homography_matrix)
            for point in player_points_2d:
                x, y = int(point[0][0]), int(point[0][1])
                cv2.circle(court_2d, (x + court_area[0], y + court_area[1]), 10, (0, 0, 255), -1)
        if ball_coord:
            ball_point_2d = cv2.perspectiveTransform(np.float32([ball_coord]).reshape(-1, 1, 2), homography_matrix)
            x, y = int(ball_point_2d[0][0][0]), int(ball_point_2d[0][0][1])
            cv2.circle(court_2d, (x + court_area[0], y + court_area[1]), 7, (0, 255, 255), -1)
    return court_2d


# --- 3. ANA İŞLEM DÖNGÜSÜ ---

def main():
    global homography_matrix
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"Hata: Video dosyası açılamadı -> {VIDEO_PATH}"); return
    ret, first_frame = cap.read()
    if not ret: print("Hata: Videodan ilk kare okunamadı."); return
    
    # 1. Adım: Kortun KESİN sınırlarını otomatik bul
    tight_court_points = find_court_corners_auto(first_frame)
    if tight_court_points is None: sys.exit()
    
    # 2. Adım: Görsel blurlama için GENİŞLETİLMİŞ poligonu oluştur
    expanded_court_points = expand_court_polygon(tight_court_points, VISUAL_PADDING)
    
    # 3. Adım: Hazırlıklar
    homography_matrix, _ = cv2.findHomography(tight_court_points, destination_points)
    blur_mask = create_court_mask(first_frame, expanded_court_points)
    logic_mask = create_court_mask(first_frame, tight_court_points)
    logic_mask_area = cv2.countNonZero(logic_mask)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break

        # Blurlama her zaman geniş maskeye göre yapılır
        display_frame = apply_blur_outside_mask(frame, blur_mask)

        # Hareket tespiti dar (mantıksal) maskeye göre yapılır
        fg_mask = bg_subtractor.apply(frame)
        fg_mask_court_only = cv2.bitwise_and(fg_mask, fg_mask, mask=logic_mask)
        motion_ratio = cv2.countNonZero(fg_mask_court_only) / (logic_mask_area + 1e-6)

        # Boş bir 2D kroki oluştur, tespit olursa üzerine çizilecek
        court_2d_view = draw_2d_court([], None)

        if motion_ratio > SCENE_CHANGE_THRESHOLD:
            cv2.putText(display_frame, "REPLAY / YAKIN CEKIM", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)
        else:
            kernel = np.ones((5, 5), np.uint8)
            fg_mask_clean = cv2.morphologyEx(fg_mask_court_only, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(fg_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_players, detected_ball = get_best_candidates(contours, frame, tight_court_points)

            player_positions_for_2d = []
            for player in detected_players:
                x, y, w, h = player['bbox']
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                player_positions_for_2d.append(player['pos'])
            
            ball_position_for_2d = None
            if detected_ball:
                x, y, w, h = detected_ball['bbox']
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                ball_position_for_2d = detected_ball['pos']
            
            # 2D krokiyi tespit edilen nesnelerle güncelle
            court_2d_view = draw_2d_court(player_positions_for_2d, ball_position_for_2d)

        # Pencereleri göster
        cv2.imshow("Tam Otomatik Tenis Analizi", display_frame)
        cv2.imshow("2D Kroki", court_2d_view)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()