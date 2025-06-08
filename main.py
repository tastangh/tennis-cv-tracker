import cv2
import numpy as np
import sys
from collections import OrderedDict
from scipy.spatial import distance as dist

# --- 1. AYARLAR VE SABİTLER ---
VIDEO_PATH = 'tennis.mp4'
COURT_TEMPLATE_PATH = 'court_template.jpg'
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Renk aralıkları (Bunları videonuza göre ayarlayabilirsiniz)
COURT_LOWER_COLOR = np.array([100, 120, 100])
COURT_UPPER_COLOR = np.array([130, 255, 255])

# Nesne tespiti için alan (piksel cinsinden) eşikleri
PLAYER_MIN_AREA = 800
PLAYER_MAX_AREA = 5000
BALL_MIN_AREA = 40
BALL_MAX_AREA = 200

# --- 2. YARDIMCI SINIF: OYUNCU TAKİBİ ---

class PlayerTracker:
    """ Basit bir Centroid tabanlı oyuncu takipçisi """
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict() # ID -> Centroid
        self.rects = OrderedDict()   # ID -> Bounding Box
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.rects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.rects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects): inputCentroids[i] = (x + w // 2, y + h // 2)

        if len(self.objects) == 0:
            for i in range(len(rects)): self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.rects[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            else:
                for col in unusedCols: self.register(inputCentroids[col], rects[col])
        return self.rects

# --- 3. YARDIMCI FONKSİYONLAR ---

def get_homography_matrix(court_img_shape):
    """ Homografi matrisini hesaplar. Bu noktalar videoya özeldir. """
    src_pts = np.array([[594, 212], [962, 212], [1280, 650], [290, 650]], dtype=np.float32)
    h, w = court_img_shape
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def create_court_mask(frame):
    """ Kort alanını belirleyen bir maske oluşturur. """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COURT_LOWER_COLOR, COURT_UPPER_COLOR)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

def detect_and_filter_objects(fg_mask, court_mask):
    """ Hareketli nesneleri bulur ve kort alanı ile filtreler. """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    player_rects, ball_rect = [], None
    valid_detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        centroid_y, centroid_x = y + h // 2, x + w // 2

        # 1. Filtre: Nesne kortun içinde mi?
        if court_mask[centroid_y, centroid_x] > 0:
            area = cv2.contourArea(cnt)
            valid_detections.append({'area': area, 'rect': (x,y,w,h)})

    # 2. Filtre: Boyuta göre sınıflandır
    if not valid_detections: return [], None

    # En büyük iki nesneyi oyuncu adayı olarak al
    valid_detections.sort(key=lambda d: d['area'], reverse=True)
    
    for det in valid_detections:
        area, rect = det['area'], det['rect']
        x,y,w,h = rect
        # Oyuncu Sınıflandırması
        if PLAYER_MIN_AREA < area < PLAYER_MAX_AREA and h > w and len(player_rects) < 2:
            player_rects.append(rect)
        # Top Sınıflandırması
        elif BALL_MIN_AREA < area < BALL_MAX_AREA and ball_rect is None:
            ball_rect = rect

    return player_rects, ball_rect

def transform_points(rects_dict, H):
    """ Oyuncuların ayak pozisyonlarını homografi ile dönüştürür. """
    if not rects_dict: return []
    # Oyuncunun yere bastığı nokta (kutunun alt-merkezi) en doğru sonucu verir
    points = [ (rect[0] + rect[2]//2, rect[1] + rect[3]) for rect in rects_dict.values() ]
    points_to_transform = np.float32(points).reshape(-1, 1, 2)
    if points_to_transform.size == 0: return []
    return cv2.perspectiveTransform(points_to_transform, H).reshape(-1, 2)

def draw_elements(frame, court_display, tracked_rects, ball_rect, transformed_players, H):
    """ Tüm görsel öğeleri çizer. """
    # Video üzerine çizim
    for (objectID, rect) in tracked_rects.items():
        x, y, w, h = rect
        color = (0, 0, 255) if objectID % 2 == 0 else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Oyuncu {objectID + 1}", (x, y - 10), FONT, 0.7, color, 2)
    if ball_rect:
        x,y,w,h = ball_rect
        cv2.circle(frame, (x+w//2, y+h//2), 7, (0, 255, 255), -1)

    # Kroki üzerine çizim
    for i, pt in enumerate(transformed_players):
        color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
        cv2.circle(court_display, (int(pt[0]), int(pt[1])), 15, color, -1)
    if ball_rect:
        ball_center = (ball_rect[0] + ball_rect[2]//2, ball_rect[1] + ball_rect[3]//2)
        transformed_ball = cv2.perspectiveTransform(np.float32([[ball_center]]).reshape(-1,1,2), H).reshape(-1,2)
        if transformed_ball is not None:
            pt = transformed_ball[0]
            cv2.circle(court_display, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)

# --- 4. ANA İŞLEV ---
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"Hata: '{VIDEO_PATH}' açılamadı."); sys.exit()
    court_template = cv2.imread(COURT_TEMPLATE_PATH)
    if court_template is None: print(f"Hata: '{COURT_TEMPLATE_PATH}' bulunamadı."); sys.exit()

    H = get_homography_matrix((court_template.shape[0], court_template.shape[1]))
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=35, detectShadows=False)
    player_tracker = PlayerTracker()
    court_mask = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        if court_mask is None: court_mask = create_court_mask(frame)

        fg_mask = bg_subtractor.apply(frame)
        player_rects, ball_rect = detect_and_filter_objects(fg_mask, court_mask)
        tracked_rects = player_tracker.update(player_rects)
        
        transformed_players = transform_points(tracked_rects, H)
        court_display = court_template.copy()
        
        draw_elements(frame, court_display, tracked_rects, ball_rect, transformed_players, H)
        
        cv2.imshow('Video', frame)
        cv2.imshow('Kort Krokisi', court_display)

        if cv2.waitKey(25) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()