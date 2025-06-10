import numpy as np
import cv2
import math
import os
from collections import deque

# --- Nesne Takibi için Kalman Filtresi Sınıfı ---
class KalmanTracker:
    """
    Oyuncuları zaman içinde takip etmek için basit bir Kalman Filtresi uygular.
    Durum uzayı [x, y, vx, vy] şeklindedir: merkez koordinatları ve hızları.
    Sadece pozisyonu ölçer, hızları modelden çıkarır.
    """
    def __init__(self, initial_box):
        # 4 durum değişkeni (x, y, vx, vy) ve 2 ölçüm değişkeni (x, y) ile filtreyi başlat.
        self.kf = cv2.KalmanFilter(4, 2)
        # Ölçüm matrisi (H): Durum uzayından sadece pozisyonu (x, y) alırız.
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # Geçiş matrisi (A): Sabit hız modelini varsayar. x_yeni = x_eski + vx, y_yeni = y_eski + vy.
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # Modelin belirsizliği (Q) ve ölçüm gürültüsü (R) için kovaryans matrisleri. Bunlar ayar parametreleridir.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Filtreyi ilk kutunun merkeziyle başlat. Hız başlangıçta sıfır.
        x, y, w, h = initial_box
        self.kf.statePost = np.array([x + w/2, y + h/2, 0, 0], np.float32).T
        
        # Takipçinin dahili durumu
        self.box = initial_box  # En son bilinen kutu
        self.age = 0  # Takipçinin kaç karedir var olduğu
        self.consecutive_invisible_count = 0  # Kaç karedir eşleşmediği
        self.court_side = None  # Oyuncunun korttaki tarafı ('top' veya 'bottom')

    def predict(self):
        """Bir sonraki durumu tahmin eder. Her kare çağrılır."""
        predicted_state = self.kf.predict()
        self.age += 1
        self.consecutive_invisible_count += 1  # Henüz bir eşleşme olmadığı için artırılır
        
        # Tahmin edilen merkezden yeni bir kutu oluştur. Boyut sabit kalır.
        predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
        w, h = self.box[2], self.box[3]
        self.box = (predicted_x - w//2, predicted_y - h//2, w, h)
        return self.box

    def update(self, box):
        """Filtreyi yeni bir ölçümle (tespit edilen kutu) günceller."""
        x, y, w, h = box
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        self.kf.correct(measurement)
        self.box = box  # Kutuyu en son ölçümle güncelle
        self.consecutive_invisible_count = 0  # Görünür olduğu için sayacı sıfırla

# --- Kort Tespiti ve Geometri Analizi ---
def detect_court_in_roi(roi_frame):
    """
    Verilen bir ROI (Region of Interest) içinde Hough Dönüşümü kullanarak kort çizgilerini bulur
    ve kortun köşe noktalarını döndürür.
    """
    # Ön işleme: Gri tonlama ve kontrast artırma (CLAHE) ile çizgileri belirginleştir.
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Yüksek bir eşik değeriyle çizgileri ikili bir maskeye dönüştür.
    _, line_mask = cv2.threshold(enhanced_gray, 210, 255, cv2.THRESH_BINARY)
    
    # Hough Transform parametrelerini ROI boyutuna göre ayarla. Bu, ölçeklenebilirliği artırır.
    h, w = roi_frame.shape[:2]
    min_line_length, max_line_gap, hough_threshold = int(w * 0.10), int(w * 0.05), 30
    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None: return None
    
    # Çizgileri açılarına göre yatay ve dikey olarak sınıflandır.
    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 45 or angle > 135: horizontal.append(line)
        else: vertical.append(line)

    # Yeterli çizgi yoksa kort oluşturulamaz.
    if len(horizontal) < 2 or len(vertical) < 2: return None
    
    # Çizgileri pozisyonlarına göre sıralayarak en dıştaki çizgileri bul.
    horizontal.sort(key=lambda l: (l[0][1] + l[0][3]) / 2) # y-pozisyonuna göre
    vertical.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)   # x-pozisyonuna göre
    top_line, bottom_line, left_line, right_line = horizontal[0], horizontal[-1], vertical[0], vertical[-1]
    
    # İki çizginin kesişim noktasını bulan yardımcı fonksiyonlar.
    def get_line_params(line):
        (x1, y1, x2, y2) = line[0]; m = float('inf') if x1 == x2 else (y2 - y1) / (x2 - x1); c = x1 if m == float('inf') else y1 - m * x1; return m, c
    def get_intersection(p1, p2):
        m1, c1 = p1; m2, c2 = p2;
        if m1 == m2: return None # Paralel çizgiler
        if m1 == float('inf'): return (int(c1), int(m2 * c1 + c2))
        if m2 == float('inf'): return (int(c2), int(m1 * c2 + c1))
        x = (c2 - c1) / (m1 - m2); y = m1 * x + c1; return (int(x), int(y))

    p_top, p_bottom, p_left, p_right = get_line_params(top_line), get_line_params(bottom_line), get_line_params(left_line), get_line_params(right_line)
    
    # Dört köşe noktasını hesapla.
    tl, tr, bl, br = get_intersection(p_top, p_left), get_intersection(p_top, p_right), get_intersection(p_bottom, p_left), get_intersection(p_bottom, p_right)
    
    # Tüm köşeler bulunamazsa, bu geçersiz bir tespittir.
    if not all([tl, tr, bl, br]): return None
    return np.array([tl, tr, br, bl], dtype=np.int32)

def is_court_geometry_valid(corners, top_bottom_ratio_range, aspect_ratio_range, vertical_line_angle_range, side_height_ratio_range):
    """
    Tespit edilen kort köşelerinin geometrik olarak mantıklı olup olmadığını kontrol eder.
    Bu, yanlış pozitifleri (örneğin, tribünler) elemek için kritik bir adımdır.
    """
    if corners is None or len(corners) != 4: return False
    
    # Köşeleri y ve x koordinatlarına göre sıralayarak tl, tr, bl, br'yi güvenilir bir şekilde bul.
    sorted_y = sorted(corners, key=lambda p: p[1])
    top_corners = sorted(sorted_y[:2], key=lambda p: p[0]); bottom_corners = sorted(sorted_y[2:], key=lambda p: p[0])
    tl, tr = top_corners[0], top_corners[1]
    bl, br = (bottom_corners[0], bottom_corners[1]) if bottom_corners[0][0] < bottom_corners[1][0] else (bottom_corners[1], bottom_corners[0])
    
    # Geometrik ölçümleri hesapla.
    top_width = np.linalg.norm(tl - tr); bottom_width = np.linalg.norm(bl - br)
    left_height = np.linalg.norm(tl - bl); right_height = np.linalg.norm(tr - br)
    
    if bottom_width < 1 or left_height < 1 or right_height < 1: return False

    # 1. Perspektif kontrolü: Üst çizgi, alt çizgiden daha kısa olmalıdır.
    top_bottom_ratio = top_width / bottom_width
    if not (top_bottom_ratio_range[0] < top_bottom_ratio < top_bottom_ratio_range[1]): return False

    # 2. En-boy oranı kontrolü: Kortun genel en-boy oranı makul olmalı.
    avg_height = (left_height + right_height) / 2
    aspect_ratio = bottom_width / avg_height
    if not (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]): return False

    # 3. Dikey çizgi açısı kontrolü: Dikey çizgiler çok fazla eğik olmamalı.
    def get_angle(p1, p2): return abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    angle_left, angle_right = get_angle(bl, tl), get_angle(br, tr)
    min_angle, max_angle = vertical_line_angle_range
    if not (min_angle < angle_left < max_angle and min_angle < angle_right < max_angle): return False

    # 4. Yan yükseklik oranı kontrolü: Sol ve sağ kenarların uzunlukları birbirine yakın olmalı.
    height_ratio = left_height / right_height
    min_h_ratio, max_h_ratio = side_height_ratio_range
    if not (min_h_ratio < height_ratio < max_h_ratio): return False
    
    return True

# --- Oyuncu ve Top Tespiti için Yardımcı Fonksiyonlar ---
def merge_overlapping_boxes(boxes, proximity_thresh=50):
    """
    Bir oyuncunun hareketinden kaynaklanan parçalı konturları tek bir kutuda birleştirir.
    Yakın kutuları, daha fazla birleştirme mümkün olmayana kadar yinelemeli olarak birleştirir.
    """
    if len(boxes) == 0: return []
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                box1, box2 = boxes[i], boxes[j]
                # İki kutu arasındaki mesafe, belirlenen eşikten küçükse birleştir.
                dist_x = max(0, max(box1[0], box2[0]) - min(box1[0] + box1[2], box2[0] + box2[2]))
                dist_y = max(0, max(box1[1], box2[1]) - min(box1[1] + box1[3], box2[1] + box2[3]))
                if dist_x < proximity_thresh and dist_y < proximity_thresh:
                    min_x, min_y = min(box1[0], box2[0]), min(box1[1], box2[1])
                    max_x, max_y = max(box1[0] + box1[2], box2[0] + box2[2]), max(box1[1] + box1[3], box2[1] + box2[3])
                    boxes[i] = (min_x, min_y, max_x - min_x, max_y - min_y)
                    boxes.pop(j)
                    merged = True
                    j = i + 1  # Listeden bir eleman silindi, j'yi sıfırla.
                else:
                    j += 1
            i += 1
            if merged: break  # Birleştirme olduysa, döngüye baştan başla.
    return boxes

def detect_players_motion_only(fg_mask, court_polygon, court_y_bounds, player_size_params, min_area=300):
    """
    Sadece hareket maskesi (foreground) kullanarak oyuncuları tespit eder.
    Tespitleri kort alanı, boyut ve şekil gibi sezgisel kurallarla filtreler.
    """
    if court_polygon is None: return []

    # Perspektife bağlı olarak oyuncu boyutu beklentisini ayarla.
    # Kortun üstündeki oyuncular daha küçük, altındakiler daha büyük görünür.
    court_top_y, court_bottom_y = court_y_bounds
    max_h_top = player_size_params['max_height_top']
    max_h_bottom = player_size_params['max_height_bottom']
    court_height = court_bottom_y - court_top_y
    if court_height <= 0: court_height = 1 

    # Morfolojik işlemlerle hareket maskesini temizle: gürültüyü azalt ve boşlukları doldur.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Filtreleme adımları:
        # 1. Konturun merkezi kortun içinde mi?
        if cv2.pointPolygonTest(court_polygon, (x + w // 2, y + h // 2), False) < 0: continue
        # 2. Alan ve şekil (en-boy oranı) oyuncuya benziyor mu?
        if cv2.contourArea(c) < min_area or w > h * 1.5 or h > w * 8: continue

        # 3. Perspektif farkındalıklı yükseklik kontrolü.
        box_center_y = y + h / 2
        y_ratio = (box_center_y - court_top_y) / court_height
        y_ratio = np.clip(y_ratio, 0, 1) # Oranı [0, 1] aralığında tut.
        # İzin verilen maksimum yüksekliği lineer interpolasyonla hesapla.
        allowed_max_height = max_h_top + y_ratio * (max_h_bottom - max_h_top)
        if h > allowed_max_height:
            continue

        candidate_boxes.append((x, y, w, h))

    # Parçalı tespitleri birleştir ve çok küçük olanları ele.
    players = merge_overlapping_boxes(candidate_boxes, proximity_thresh=50)
    return [box for box in players if (box[2] * box[3]) > 500]

def get_court_side(box, homography_matrix, court_center_y):
    """
    Bir oyuncunun kortun hangi tarafında ('top' veya 'bottom') olduğunu belirler.
    Bunu, oyuncunun ayak noktasını homografi ile kuşbakışı görünüme dönüştürerek yapar.
    """
    if homography_matrix is None: return None
    x, y, w, h = box
    # Oyuncunun ayaklarının olduğu varsayılan noktayı al (kutunun alt ortası).
    foot_point = np.array([[[x + w / 2, y + h]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(foot_point, homography_matrix)
    
    if transformed_point is None or transformed_point.size == 0: return None
    # Dönüştürülmüş noktanın y-koordinatını kuşbakışı kort merkezinin y'si ile karşılaştır.
    return "top" if transformed_point[0][0][1] < court_center_y else "bottom"

def get_valid_contours(fg_mask, min_area, max_area):
    """ Hareket maskesindeki belirli alan aralığına sahip konturları bulur. """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

def is_point_inside_bbox(point, bbox):
    """ Bir noktanın bir sınırlayıcı kutunun içinde olup olmadığını kontrol eder. """
    x_p, y_p = point; x_b, y_b, w_b, h_b = bbox
    return x_b < x_p < x_b + w_b and y_b < y_p < y_b + h_b

def detect_ball(fg_mask, last_known_ball_center, ball_lost_counter, court_boundary_points, player_bboxes, params):
    """
    Hareket maskesinde topu tespit eder. Boyut, şekil, doluluk ve konum gibi
    birçok kısıtlamayı kullanarak yanlış pozitifleri en aza indirmeye çalışır.
    """
    # Top yakın zamanda görüldüyse, arama alanını son bilinen konumun etrafıyla sınırla.
    # Bu, performansı artırır ve yanlış pozitifleri azaltır.
    if last_known_ball_center and ball_lost_counter < params['MAX_BALL_LOST_FRAMES']:
        search_mask = np.zeros_like(fg_mask)
        sx = int(last_known_ball_center[0] - params['SEARCH_REGION_PADDING']); sy = int(last_known_ball_center[1] - params['SEARCH_REGION_PADDING'])
        sw = int(params['SEARCH_REGION_PADDING'] * 2); sh = int(params['SEARCH_REGION_PADDING'] * 2)
        cv2.rectangle(search_mask, (sx, sy), (sx + sw, sy + sh), 255, -1)
        search_area = cv2.bitwise_and(fg_mask, fg_mask, mask=search_mask)
    else:
        search_area = fg_mask
        
    contours = get_valid_contours(search_area, params['MIN_BALL_CONTOUR_AREA'], params['MAX_BALL_CONTOUR_AREA'])
    
    potential_balls = []
    active_player_bboxes = [t.box for t in player_bboxes if t is not None]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Top benzeri özellikler için katı filtreleme:
        # 1. Boyut: Çok büyük veya çok küçük olamaz.
        if not (params['BALL_MIN_WIDTH_HEIGHT'] <= w <= params['BALL_MAX_WIDTH_HEIGHT'] and params['BALL_MIN_WIDTH_HEIGHT'] <= h <= params['BALL_MAX_WIDTH_HEIGHT']): continue
        # 2. En-boy oranı: Neredeyse kare olmalı.
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (params['BALL_MIN_ASPECT_RATIO'] <= aspect_ratio <= params['BALL_MAX_ASPECT_RATIO']): continue
        # 3. Doluluk (Solidity): Kontur, dairesel/dolu bir şekil mi? (Alan / Dışbükey Zarf Alanı)
        area = cv2.contourArea(cnt)
        if len(cnt) >= 5: # Dışbükey zarf için en az 5 nokta gerekir.
            if (float(area) / cv2.contourArea(cv2.convexHull(cnt))) < params['BALL_MIN_SOLIDITY']: continue
        
        center_candidate = (x + w // 2, y + h // 2)
        # 4. Konum: Kortun içinde ve bir oyuncunun kutusunun içinde olmamalı.
        if cv2.pointPolygonTest(court_boundary_points, center_candidate, False) < 0: continue
        if any(is_point_inside_bbox(center_candidate, p_bbox) for p_bbox in active_player_bboxes): continue
        
        potential_balls.append({'bbox': (x, y, w, h), 'center': center_candidate, 'area': area})

    if not potential_balls: return None

    best_ball = None
    # Top yakın zamanda görüldüyse, en yakın adayı en iyi eşleşme olarak seç.
    if last_known_ball_center and ball_lost_counter < params['MAX_BALL_LOST_FRAMES']:
        min_dist = float('inf')
        for ball in potential_balls:
            dist = np.linalg.norm(np.array(last_known_ball_center) - np.array(ball['center']))
            # Topun bir kareden diğerine "zıplayabileceği" maksimum mesafeyi kontrol et.
            if dist < params['MAX_BALL_JUMP_DISTANCE'] and dist < min_dist:
                min_dist = dist
                best_ball = ball
    
    # Eğer bir önceki kareden bir eşleşme bulunamadıysa (veya top kayıpsa), en büyük potansiyel topu seç.
    if best_ball is None:
        potential_balls.sort(key=lambda b: b['area'], reverse=True)
        best_ball = potential_balls[0]
        
    return (best_ball['bbox'], best_ball['center'])


# --- Görselleştirme Fonksiyonları ---
def transform_points_for_sketch(points_list, H_matrix):
    """Noktaları video koordinat sisteminden kroki koordinat sistemine dönüştürür."""
    if not points_list or H_matrix is None: return []
    np_points = np.array([[p[0], p[1]] for p in points_list], dtype=np.float32)
    if np_points.shape[0] == 0: return []
    # cv2.perspectiveTransform, [1, N, 2] şeklinde bir dizi bekler.
    transformed_points = cv2.perspectiveTransform(np.array([np_points]), H_matrix)
    if transformed_points is None: return []
    return transformed_points[0]

def draw_court_sketch(base_sketch_img, homography_matrix, player_trackers, ball_trail_video, current_frame_count, viz_params):
    """
    Kortun 2D kuşbakışı krokisini çizer. Oyuncuları ve topun izini gösterir.
    """
    sketch_display = base_sketch_img.copy()
    if homography_matrix is None: return sketch_display
    
    # Topun geçmiş pozisyonlarını (izini) krokiye çiz.
    if ball_trail_video:
        ball_points_to_transform = [pos for pos, fc in ball_trail_video]
        transformed_ball_trail = transform_points_for_sketch(ball_points_to_transform, homography_matrix)
        
        for i, (original_pos, fc) in enumerate(ball_trail_video):
            if i < len(transformed_ball_trail):
                t_point_ball = tuple(map(int, transformed_ball_trail[i]))
                # Top izinin eski noktalarını soluklaştırarak göster.
                age = current_frame_count - fc
                fade_factor = max(0.1, 1.0 - (age / float(viz_params['BALL_FADE_DURATION_SKETCH'])))
                faded_color_np = (np.array(viz_params['ball_viz_color'], dtype=np.float32) * fade_factor).astype(np.uint8)
                faded_color_tuple = tuple(faded_color_np.tolist())
                # Yeni noktalar daha büyük çizilir.
                radius = 5 if age < 2 else (4 if age < viz_params['SKETCH_BALL_HISTORY_LEN'] // 2 else 3)
                cv2.circle(sketch_display, t_point_ball, radius, faded_color_tuple, -1)
    
    # Oyuncuları krokiye çiz.
    player_points_to_transform, player_colors, player_radii = [], [], []
    active_trackers = [t for t in player_trackers.values() if t is not None]
    for tracker in active_trackers:
        x, y, w, h = tracker.box
        player_points_to_transform.append((x + w/2, y + h)) # Ayak noktasını kullan
        
        # Oyuncu geçici olarak kaybolduysa farklı bir renkle göster.
        is_lost = tracker.consecutive_invisible_count > 0
        pid = 0 if tracker.court_side == 'top' else 1
        player_colors.append(viz_params['player_viz_colors_lost'][pid] if is_lost else viz_params['player_viz_colors'][pid])
        player_radii.append(6 if is_lost else 7)

    # Oyuncu noktalarını tek seferde dönüştür (daha verimli).
    if player_points_to_transform:
        transformed_player_points = transform_points_for_sketch(player_points_to_transform, homography_matrix)
        for i, t_point_player in enumerate(transformed_player_points):
            t_point_player_int = tuple(map(int, t_point_player))
            cv2.circle(sketch_display, t_point_player_int, player_radii[i], player_colors[i], -1)
            
    return sketch_display

def create_combined_view(main_frame, motion_mask, sketch_view, debug_mode, is_active):
    """
    Ana video, hareket maskesi ve krokiyi birleştirerek tek bir çıktı penceresi oluşturur.
    """
    h, w, _ = main_frame.shape
    # Debug modundaysa veya analiz aktifse hareket maskesini göster.
    if debug_mode and is_active and motion_mask is not None:
        bottom_panel = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    else: # Aksi takdirde, durum bilgisi gösteren siyah bir panel oluştur.
        bottom_panel = np.zeros_like(main_frame)
        status = "Analiz Aktif" if is_active else "Analiz Duraklatildi"
        cv2.putText(bottom_panel, status, (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bottom_panel, "Hareket Maskesi", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Ana video ve alt paneli dikey olarak birleştir.
    left_column = cv2.vconcat([main_frame, bottom_panel])

    # Kroki varsa, sol sütunun yüksekliğine uyacak şekilde yeniden boyutlandır.
    if sketch_view is not None:
        target_h = left_column.shape[0]
        target_w = int(sketch_view.shape[1] * (target_h / sketch_view.shape[0]))
        right_column = cv2.resize(sketch_view, (target_w, target_h))
    else: # Kroki yoksa, boş bir panel oluştur.
        target_h = left_column.shape[0]
        target_w = int(target_h / 2.2)
        right_column = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cv2.putText(right_column, "Kroki Yok", (target_w//2 - 50, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    # Sol ve sağ sütunları yatay olarak birleştir.
    return cv2.hconcat([left_column, right_column])

# --- Ana İşlem Fonksiyonu ---
def main(debug=False):
    video_path = 'tennis.mp4'
    if not os.path.exists(video_path):
        print(f"Hata: '{video_path}' bulunamadı.")
        return
        
    cap = cv2.VideoCapture(video_path)
    # Performans için videoyu sabit bir genişliğe yeniden boyutlandır.
    target_width = 800
    ret, test_frame = cap.read()
    if not ret:
        print("Hata: Video dosyası okunamadı.")
        return
    scale = target_width / test_frame.shape[1]
    target_height = int(test_frame.shape[0] * scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- Sezgisel Parametreler ve Eşik Değerleri ---
    # Bu değerler, belirli bir video ve kamera açısı için deneysel olarak ayarlanmıştır.
    
    # Kort geometrisi doğrulama parametreleri
    TOP_BOTTOM_RATIO_RANGE = (0.3, 0.98)
    ASPECT_RATIO_RANGE = (1.0, 2.5)
    VERTICAL_LINE_ANGLE_RANGE = (60, 120)
    SIDE_HEIGHT_RATIO_RANGE = (0.75, 1.25)
    INVALID_VIEW_THRESHOLD = 5  # Kort görünümünün kaybolduğunu kabul etmeden önce beklenecek kare sayısı
    COURT_TOP_PADDING_PIXELS = 30 # Kort alanını görselleştirme ve maskeleme için genişletme miktarı
    COURT_HORIZONTAL_PADDING_PIXELS = 20

    # Oyuncu takibi parametreleri
    MAX_PLAYER_JUMP_DISTANCE = 150 # Bir takipçinin yeni bir tespitle eşleşmesi için maksimum piksel mesafesi
    MAX_PLAYER_INVISIBLE_FRAMES = 25 # Bir oyuncu takipçisinin silinmeden önce görünmez kalabileceği max kare sayısı
    
    # Perspektife bağlı oyuncu boyutu parametreleri
    MAX_PLAYER_HEIGHT_TOP = 120    # Kortun üst kısmındaki bir oyuncu için beklenen maksimum yükseklik
    MAX_PLAYER_HEIGHT_BOTTOM = 280 # Kortun alt kısmındaki bir oyuncu için beklenen maksimum yükseklik

    # Top tespiti parametreleri
    ball_params = {
        'MIN_BALL_CONTOUR_AREA': 8, 'MAX_BALL_CONTOUR_AREA': 100,
        'BALL_MIN_WIDTH_HEIGHT': 3, 'BALL_MAX_WIDTH_HEIGHT': 25,
        'BALL_MIN_ASPECT_RATIO': 0.7, 'BALL_MAX_ASPECT_RATIO': 1.4,
        'BALL_MIN_SOLIDITY': 0.75, 'MAX_BALL_LOST_FRAMES': 10,
        'MAX_BALL_JUMP_DISTANCE': 80, 'SEARCH_REGION_PADDING': 50
    }
    # Görselleştirme parametreleri
    viz_params = {
        'player_viz_colors': {0:(255, 0, 255), 1:(128, 0, 0)}, # Üst/Alt oyuncu renkleri
        'player_viz_colors_lost': {0:(160, 0, 160), 1:(80, 0, 0)}, # Kayıp oyuncu renkleri
        'ball_viz_color':(0, 255, 0),
        'SKETCH_BALL_HISTORY_LEN': 7,  # Krokide top izinin uzunluğu
        'BALL_FADE_DURATION_SKETCH': 12 # Top izinin solma süresi (kare cinsinden)
    }    
    
    # 2D Kroki için kurulum
    KROKI_IMAGE_PATH = "kort.png"
    TARGET_SKETCH_WIDTH, TARGET_SKETCH_HEIGHT = 250, 430
    base_sketch_resized = None
    if os.path.exists(KROKI_IMAGE_PATH):
        base_sketch_resized = cv2.resize(cv2.imread(KROKI_IMAGE_PATH), (TARGET_SKETCH_WIDTH, TARGET_SKETCH_HEIGHT))
    else:
        print(f"UYARI: Kroki resmi '{KROKI_IMAGE_PATH}' bulunamadı.")
    
    # Homografi için hedef noktalar (Kroki görüntüsündeki kort köşeleri)
    margin_x_sketch, margin_y_sketch = 25, 35
    DST_POINTS_FOR_SKETCH = np.array([
        [margin_x_sketch, margin_y_sketch],
        [TARGET_SKETCH_WIDTH - margin_x_sketch, margin_y_sketch],
        [TARGET_SKETCH_WIDTH - margin_x_sketch, TARGET_SKETCH_HEIGHT - margin_y_sketch],
        [margin_x_sketch, TARGET_SKETCH_HEIGHT - margin_y_sketch]], dtype=np.float32)

    # Arka plan çıkarıcı (Background Subtractor)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    
    # --- Durum Değişkenleri ---
    last_known_corners = None       # En son geçerli kort köşeleri
    is_court_view_active = True     # Kortun geçerli bir açıda olup olmadığı
    invalid_view_counter = 0        # Geçersiz görünüm sayacı
    frame_counter = 0               # Toplam kare sayısı
    player_trackers = {'top': None, 'bottom': None} # Oyuncu takipçileri
    homography_matrix, homography_matrix_for_sketch = None, None
    last_known_ball_center = None   # Topun son bilinen merkezi
    ball_lost_counter = 0           # Topun kaç karedir kayıp olduğu
    ball_trail_video = deque(maxlen=viz_params['SKETCH_BALL_HISTORY_LEN']) # Topun video üzerindeki izi

    # Mantıksal işlemler (örn. oyuncu tarafı tespiti) için sanal bir kuşbakışı kort oluştur
    top_down_width_logic, top_down_height_logic = 400, 800
    dst_points_for_logic = np.array([
        [0, 0], [top_down_width_logic-1, 0],
        [top_down_width_logic-1, top_down_height_logic-1],
        [0, top_down_height_logic-1]], dtype=np.float32)
    court_center_y_top_down = top_down_height_logic / 2
    
    status_text, status_color = "", (0, 0, 0)
    
    print("İşlem başlıyor... Çıkmak için 'q' tuşuna basın.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (target_width, target_height))
        frame_counter += 1

        # --- Adım 1: Kort Tespiti ve Durum Yönetimi ---
        # Kortu sadece ekranın alt %80'inde ara (gökyüzünü ve üst tribünleri dışarıda bırak).
        h, w, _ = frame.shape
        roi_for_check = frame[int(h*0.2):, :]
        current_corners_relative = detect_court_in_roi(roi_for_check)
        is_geometry_ok = False
        if current_corners_relative is not None:
            # Göreceli koordinatları mutlak kare koordinatlarına dönüştür.
            current_corners_absolute = current_corners_relative + [0, int(h*0.2)]
            if is_court_geometry_valid(current_corners_absolute, TOP_BOTTOM_RATIO_RANGE, ASPECT_RATIO_RANGE, VERTICAL_LINE_ANGLE_RANGE, SIDE_HEIGHT_RATIO_RANGE):
                is_geometry_ok = True
        
        # Kort tespiti başarılıysa, durumu "aktif" yap ve son bilinen köşeleri güncelle.
        if is_geometry_ok:
            invalid_view_counter = 0
            is_court_view_active = True
            last_known_corners = current_corners_absolute
            
            # Homografi matrislerini hesapla (hem mantık hem de kroki için).
            sorted_corners = sorted(last_known_corners, key=lambda p: p[1])
            top_corners = sorted(sorted_corners[:2], key=lambda p: p[0]); bottom_corners = sorted(sorted_corners[2:], key=lambda p: p[0])
            src_points = np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
            homography_matrix, _ = cv2.findHomography(src_points, dst_points_for_logic)
            homography_matrix_for_sketch, _ = cv2.findHomography(src_points, DST_POINTS_FOR_SKETCH)
        else:
            invalid_view_counter += 1
        
        # Belirli bir süre kort tespit edilemezse (örneğin, kamera kayması, tekrar gösterimi), analizi duraklat.
        if invalid_view_counter > INVALID_VIEW_THRESHOLD:
            is_court_view_active = False

        if is_court_view_active: status_text, status_color = "Kort Tam Istenen Acida", (0, 255, 0)
        else: status_text, status_color = "Kort Aci Kontrolu Basarisiz (Replay?)", (0, 0, 255)

        # --- Adım 2: Oyuncu ve Top Tespiti (Sadece kort aktifse) ---
        detected_ball_info, fg_mask_court_only = None, None
        if is_court_view_active and last_known_corners is not None and homography_matrix is not None:
            # Sadece kort alanı içinde hareket tespiti yapmak için bir maske oluştur.
            padded_corners = np.copy(last_known_corners); y_indices = np.argsort(padded_corners[:, 1]); x_indices = np.argsort(padded_corners[:, 0])
            padded_corners[y_indices[:2], 1] -= COURT_TOP_PADDING_PIXELS; padded_corners[x_indices[:2], 0] -= COURT_HORIZONTAL_PADDING_PIXELS; padded_corners[x_indices[2:], 0] += COURT_HORIZONTAL_PADDING_PIXELS
            court_polygon = cv2.convexHull(padded_corners)
            play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(play_area_mask, [court_polygon], -1, 255, -1)
            
            # Arka plan çıkarma işlemini uygula.
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            fg_mask_raw = bg_subtractor.apply(blurred_frame, learningRate=0.005)
            fg_mask_court_only = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=play_area_mask)
            
            # --- Oyuncu Takibi ---
            # 1. Tahmin: Mevcut tüm takipçiler için yeni konumu tahmin et.
            for side in player_trackers:
                if player_trackers[side]:
                    player_trackers[side].predict()

            # 2. Tespit: Yeni karede oyuncuları tespit et.
            court_y_bounds = (np.min(last_known_corners[:, 1]), np.max(last_known_corners[:, 1]))
            player_size_params = {'max_height_top': MAX_PLAYER_HEIGHT_TOP, 'max_height_bottom': MAX_PLAYER_HEIGHT_BOTTOM}
            player_detections = detect_players_motion_only(fg_mask_court_only, court_polygon, court_y_bounds, player_size_params)
            
            # Tespitleri kortun üst ve alt tarafına göre ayır.
            top_detections, bottom_detections = [], []
            for i, det in enumerate(player_detections):
                side = get_court_side(det, homography_matrix, court_center_y_top_down)
                if side == 'top': top_detections.append({'box': det, 'id': i})
                elif side == 'bottom': bottom_detections.append({'box': det, 'id': i})

            # 3. Veri İlişkilendirme (Data Association): Tespitleri mevcut takipçilerle eşleştir.
            used_detection_ids = set()
            for side, tracker in player_trackers.items():
                if tracker:
                    detections_on_side = top_detections if side == 'top' else bottom_detections
                    best_match = None
                    min_dist = MAX_PLAYER_JUMP_DISTANCE
                    
                    # O taraftaki en yakın tespiti bul (en yakın komşu).
                    for det_info in detections_on_side:
                        dist = np.linalg.norm(np.array(tracker.box[:2]) - np.array(det_info['box'][:2]))
                        if dist < min_dist:
                            min_dist = dist
                            best_match = det_info
                    
                    # Eşleşme bulunursa, takipçiyi güncelle.
                    if best_match:
                        tracker.update(best_match['box'])
                        used_detection_ids.add(best_match['id'])

            # 4. Yeni Takipçi Oluşturma: Eşleşmemiş tespitler için yeni takipçiler başlat.
            for side in ['top', 'bottom']:
                if player_trackers[side] is None: # Sadece o tarafta bir takipçi yoksa yenisini oluştur.
                    detections_on_side = top_detections if side == 'top' else bottom_detections
                    available_detections = [d for d in detections_on_side if d['id'] not in used_detection_ids]
                    if available_detections:
                        # En büyük alana sahip eşleşmemiş tespiti yeni takipçi olarak seç.
                        best_new_detection = max(available_detections, key=lambda d: d['box'][2] * d['box'][3])
                        new_tracker = KalmanTracker(best_new_detection['box'])
                        new_tracker.court_side = side
                        player_trackers[side] = new_tracker
                        used_detection_ids.add(best_new_detection['id'])

            # 5. Takipçi Silme: Uzun süredir görünmeyen takipçileri sil.
            for side, tracker in player_trackers.items():
                if tracker and tracker.consecutive_invisible_count > MAX_PLAYER_INVISIBLE_FRAMES:
                    player_trackers[side] = None

            # --- Top Tespiti ---
            active_player_bboxes = [t for t in player_trackers.values() if t is not None]
            detected_ball_info = detect_ball(fg_mask_court_only, last_known_ball_center, ball_lost_counter, cv2.convexHull(last_known_corners), active_player_bboxes, ball_params)
            
            if detected_ball_info:
                last_known_ball_center = detected_ball_info[1]
                ball_lost_counter = 0
                ball_trail_video.append((last_known_ball_center, frame_counter))
            else:
                ball_lost_counter += 1
                if ball_lost_counter > ball_params['MAX_BALL_LOST_FRAMES']:
                    last_known_ball_center = None
                    ball_trail_video.clear()
        else:
            # Analiz aktif değilse, topu da kayıp say.
            ball_lost_counter += 1
            if ball_lost_counter > ball_params['MAX_BALL_LOST_FRAMES']:
                last_known_ball_center = None
                ball_trail_video.clear()

        # --- Adım 3: Görselleştirme ---
        display_frame = frame.copy()
        
        # Krokiyi çiz.
        sketch_display = None
        if base_sketch_resized is not None:
             sketch_display = draw_court_sketch(base_sketch_resized, homography_matrix_for_sketch, player_trackers if is_court_view_active else {}, ball_trail_video, frame_counter, viz_params)
        
        # Video üzerine çizimler yap.
        if is_court_view_active and last_known_corners is not None:
            # Kort alanını yarı saydam bir renkle vurgula.
            padded_corners_viz = np.copy(last_known_corners); y_indices_viz = np.argsort(padded_corners_viz[:, 1]); x_indices_viz = np.argsort(padded_corners_viz[:, 0])
            padded_corners_viz[y_indices_viz[:2], 1] -= COURT_TOP_PADDING_PIXELS; padded_corners_viz[x_indices_viz[:2], 0] -= COURT_HORIZONTAL_PADDING_PIXELS; padded_corners_viz[x_indices_viz[2:], 0] += COURT_HORIZONTAL_PADDING_PIXELS
            viz_mask = np.zeros(frame.shape[:2], dtype=np.uint8); cv2.drawContours(viz_mask, [cv2.convexHull(padded_corners_viz)], -1, 255, -1)
            viz_mask_bgr = cv2.cvtColor(viz_mask, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(viz_mask_bgr, 0.3, display_frame, 0.7, 0, display_frame)
            
        # Durum metnini ekle.
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3) # Gölge için siyah
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
        if is_court_view_active:
            # Oyuncu kutularını çiz.
            active_trackers = sorted([t for t in player_trackers.values() if t], key=lambda t: 0 if t.court_side == 'top' else 1)
            for tracker in active_trackers:
                x, y, w_box, h_box = map(int, tracker.box)
                pid = 0 if tracker.court_side == 'top' else 1
                color = viz_params['player_viz_colors'][pid]
                label = "P_Top" if tracker.court_side == 'top' else "P_Bottom"
                cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), color, 2)
                cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Topu çiz.
            if detected_ball_info:
                _, ball_center = detected_ball_info
                cv2.circle(display_frame, ball_center, 8, viz_params['ball_viz_color'], -1)

        # Tüm görünümleri birleştir ve ekranda göster.
        combined_view = create_combined_view(display_frame, fg_mask_court_only, sketch_display, debug, is_court_view_active)
        cv2.imshow("Tenis Analizi - Birlesik Gorunum", combined_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")

if __name__ == "__main__":
    main(debug=True)