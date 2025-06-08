import cv2
import numpy as np

VIDEO_PATH = "tennis.mp4"
COURT_TEMPLATE_PATH = "court_template.jpg"
SCALE_FACTOR = 2.0  # Kort büyütme faktörü
CROP_TEMPLATE = None  # Kort kontrol için şablon

# Kort görünürlüğünü template matching ile kontrol et
def is_full_court_visible(frame):
    global CROP_TEMPLATE
    if CROP_TEMPLATE is None:
        CROP_TEMPLATE = frame[50:150, 300:900]

    roi = frame[50:150, 300:900]
    if roi.shape[0:2] != CROP_TEMPLATE.shape[0:2]:
        return False

    res = cv2.matchTemplate(roi, CROP_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, matchVal, _, _ = cv2.minMaxLoc(res)
    return matchVal > 0.8  # Eşik değer

# Kareler arası fark ile hareketli nesne çıkarımı
def detect_with_frame_diff(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 5)
    return thresh

# Oyuncu ve top tespiti (debug çizim dahil)
def detect_objects(thresh, debug_frame=None):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    players = []
    ball = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        if debug_frame is not None:
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(debug_frame, f"{int(area)} {aspect_ratio:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)

        # Oyuncu: büyük ve dikdörtgen (dikey olabilir)
        if area > 400 and 0.2 < aspect_ratio < 1.2 and h > 30:
            players.append((x, y, w, h))

        # Top: küçük ve yuvarlağa yakın
        elif 10 < area < 300 and 0.8 < aspect_ratio < 1.3 and w < 35 and h < 35:
            if ball is None or area > ball[2] * ball[3]:
                ball = (x, y, w, h)

    return players, ball

# Homography dönüşümü
def apply_homography(points, H):
    if not points:
        return []
    pts = np.float32(points).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)

# Frame üzerine çizim
def draw_on_frame(frame, players, ball):
    for i, (x, y, w, h) in enumerate(players):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"P{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if ball:
        x, y, w, h = ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Kroki üzerine nokta olarak çizim
def draw_on_court(court_img, player_pts, ball_pt, scale):
    overlay = court_img.copy()
    for pt in player_pts:
        x, y = int(pt[0] * scale), int(pt[1] * scale)
        cv2.circle(overlay, (x, y), 10, (255, 0, 0), -1)
    if ball_pt is not None:
        x, y = int(ball_pt[0] * scale), int(ball_pt[1] * scale)
        cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1)
    return overlay

# Ana fonksiyon
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, prev_frame = cap.read()
    if not ret:
        print("Video yüklenemedi.")
        return
    prev_frame = cv2.resize(prev_frame, (1280, 720))
    court_img = cv2.imread(COURT_TEMPLATE_PATH)

    if court_img is None:
        print("Kort görseli yüklenemedi.")
        return

    resized_court = cv2.resize(court_img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    # Homography için manuel alınan 4 nokta (video ve krokiye göre güncellenmeli)
    src_pts = np.float32([[270, 80], [1010, 80], [1180, 680], [100, 680]])
    dst_pts = np.float32([[100, 100], [700, 100], [700, 400], [100, 400]])
    H, _ = cv2.findHomography(src_pts, dst_pts)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.resize(curr_frame, (1280, 720))

        # Kort görünmüyorsa bu kareyi atla
        if not is_full_court_visible(curr_frame):
            prev_frame = curr_frame.copy()
            continue

        thresh = detect_with_frame_diff(prev_frame, curr_frame)
        players, ball = detect_objects(thresh, debug_frame=curr_frame)

        draw_on_frame(curr_frame, players, ball)

        player_centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in players]
        ball_center = (ball[0] + ball[2] // 2, ball[1] + ball[3] // 2) if ball else None

        transformed_players = apply_homography(player_centers, H)
        transformed_ball = apply_homography([ball_center], H)[0] if ball_center else None

        court_display = draw_on_court(resized_court.copy(), transformed_players, transformed_ball, SCALE_FACTOR)

        cv2.imshow("Video Frame", curr_frame)
        cv2.imshow("Court View (Large)", court_display)
        # cv2.imshow("Thresh", thresh)  # İstersen aç

        prev_frame = curr_frame.copy()
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
