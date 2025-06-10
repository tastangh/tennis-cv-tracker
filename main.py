import cv2
import numpy as np

# --- Sabitler ve Yapılandırma ---
VIDEO_PATH = "tennis.mp4"
KROKI_IMAGE_PATH = "kort.png"

SRC_POINTS_ORIGINAL = np.array([[418,222],[862,223],[1037,576],[240,573]], dtype=np.float32)
ROI_MARGIN_TOP=140; ROI_MARGIN_SIDES=30; ROI_MARGIN_BOTTOM=60
SRC_POINTS_EXPANDED_FOR_ROI = np.array([
    [SRC_POINTS_ORIGINAL[0][0]-ROI_MARGIN_SIDES, SRC_POINTS_ORIGINAL[0][1]-ROI_MARGIN_TOP],
    [SRC_POINTS_ORIGINAL[1][0]+ROI_MARGIN_SIDES, SRC_POINTS_ORIGINAL[1][1]-ROI_MARGIN_TOP],
    [SRC_POINTS_ORIGINAL[2][0]+ROI_MARGIN_SIDES, SRC_POINTS_ORIGINAL[2][1]+ROI_MARGIN_BOTTOM],
    [SRC_POINTS_ORIGINAL[3][0]-ROI_MARGIN_SIDES, SRC_POINTS_ORIGINAL[3][1]+ROI_MARGIN_BOTTOM]
], dtype=np.float32)
TARGET_SKETCH_WIDTH=250; TARGET_SKETCH_HEIGHT=430
margin_x_sketch=25; margin_y_sketch=35
DST_POINTS_FOR_SKETCH = np.array([
    [margin_x_sketch,margin_y_sketch],
    [TARGET_SKETCH_WIDTH-margin_x_sketch,margin_y_sketch],
    [TARGET_SKETCH_WIDTH-margin_x_sketch,TARGET_SKETCH_HEIGHT-margin_y_sketch],
    [margin_x_sketch,TARGET_SKETCH_HEIGHT-margin_y_sketch]
], dtype=np.float32)

MIN_PLAYER_CONTOUR_AREA=800; MAX_PLAYER_CONTOUR_AREA=5000
MIN_BALL_CONTOUR_AREA=8; MAX_BALL_CONTOUR_AREA=120
BALL_MIN_WIDTH_HEIGHT=3; BALL_MAX_WIDTH_HEIGHT=28
BALL_MIN_ASPECT_RATIO=0.65; BALL_MAX_ASPECT_RATIO=1.45
BALL_MIN_SOLIDITY=0.75
MIN_WHITE_LINES_FOR_FULL_COURT=8
MAX_LOST_FRAMES_PLAYER=30; MAX_LOST_FRAMES_ACTIVE_DETECTION=4
MAX_MATCH_DISTANCE=150
MAX_BALL_JUMP_DISTANCE=80 # Biraz artırıldı
SEARCH_REGION_PADDING=50 # Biraz artırıldı
MAX_BALL_LOST_FRAMES=10
SKETCH_BALL_HISTORY_LEN = 7      # Kroki üzerinde kaç karelik top izi
BALL_FADE_DURATION_SKETCH = 12 # Topun kroki üzerinde kaç karede soluklaşacağı

# --- Yardımcı Fonksiyonlar ---
def get_homography_matrix(src_pts,dst_pts):H,_=cv2.findHomography(src_pts,dst_pts);return H
def is_full_court_visible(frame,min_lines_threshold):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);_,thresh=cv2.threshold(gray,195,255,cv2.THRESH_BINARY)
    kernel=np.ones((3,3),np.uint8);thresh_opened=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
    lines=cv2.HoughLinesP(thresh_opened,1,np.pi/180,threshold=30,minLineLength=50,maxLineGap=10)
    return lines is not None and len(lines)>min_lines_threshold
def get_valid_contours(fg_mask,min_area,max_area):
    contours,_=cv2.findContours(fg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    valid_contours=[cnt for cnt in contours if min_area<cv2.contourArea(cnt)<max_area]
    return sorted(valid_contours,key=cv2.contourArea,reverse=True)
def transform_points_for_sketch(points_list,H_matrix):
    if not points_list:return[]
    np_points=np.array([[p[0],p[1]]for p in points_list],dtype=np.float32)
    if np_points.shape[0]==0:return[]
    np_points_homogeneous=np.hstack((np_points,np.ones((np_points.shape[0],1))))
    transformed_homogeneous=H_matrix.dot(np_points_homogeneous.T).T;transformed_points=[]
    for p_h in transformed_homogeneous:
        w_coord=p_h[2]
        if w_coord!=0:transformed_points.append((int(p_h[0]/w_coord),int(p_h[1]/w_coord)))
        else:transformed_points.append((int(p_h[0]),int(p_h[1])))
    return transformed_points
def create_court_mask_on_video(frame_shape,video_court_corners):
    mask=np.zeros((frame_shape[0],frame_shape[1]),dtype=np.uint8);cv2.fillPoly(mask,[np.int32(video_court_corners)],255);return mask
def is_point_inside_bbox(point,bbox):x_p,y_p=point;x_b,y_b,w_b,h_b=bbox;return x_b<x_p<x_b+w_b and y_b<y_p<y_b+h_b

# --- Ana İşlem Bloğu ---
def main():
    cap=cv2.VideoCapture(VIDEO_PATH);kroki_base_original=cv2.imread(KROKI_IMAGE_PATH)
    if not cap.isOpened()or kroki_base_original is None:print("Hata: Video veya kroki resmi açılamadı!");return
    kroki_base_resized=cv2.resize(kroki_base_original,(TARGET_SKETCH_WIDTH,TARGET_SKETCH_HEIGHT))
    homography_matrix_for_sketch=get_homography_matrix(SRC_POINTS_ORIGINAL,DST_POINTS_FOR_SKETCH)
    video_court_boundary_points_expanded_roi=SRC_POINTS_EXPANDED_FOR_ROI.astype(np.int32)
    video_court_boundary_points_original_roi=SRC_POINTS_ORIGINAL.astype(np.int32)
    bg_subtractor=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=50,detectShadows=False)
    player_viz_colors={0:(255,100,0),1:(0,100,255)};player_viz_colors_lost={0:(150,60,0),1:(0,60,150)}
    ball_viz_color=(0,255,0);ball_viz_color_lost=(0,150,0)
    frame_h,frame_w=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    court_center_y_line=360
    tracked_players_dict={};court_roi_mask_expanded=create_court_mask_on_video((frame_h,frame_w),video_court_boundary_points_expanded_roi)
    last_known_ball_center=None;ball_lost_counter=0
    sketch_ball_trail = [] # [(center_video, frame_count_detected)]
    frame_count = 0

    while True:
        ret,frame=cap.read();frame_count+=1
        if not ret:break
        current_frame_display=frame.copy();sketch_display=kroki_base_resized.copy()
        detected_ball_info_this_frame=None;temp_player_bboxes=[]
        full_court_mode=is_full_court_visible(frame,MIN_WHITE_LINES_FOR_FULL_COURT)

        if not full_court_mode:
            cv2.putText(current_frame_display,"Kort Tam Gorunmuyor",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            for pid,data in list(tracked_players_dict.items()):
                tracked_players_dict[pid][2]+=1
                if tracked_players_dict[pid][2]>=MAX_LOST_FRAMES_PLAYER:del tracked_players_dict[pid]
            ball_lost_counter+=1
            if ball_lost_counter > MAX_BALL_LOST_FRAMES//2:last_known_ball_center=None
        else:
            fg_mask_raw=bg_subtractor.apply(frame)
            kernel_open=np.ones((3,3),np.uint8);kernel_close=np.ones((9,9),np.uint8)
            fg_mask_processed=cv2.morphologyEx(fg_mask_raw,cv2.MORPH_OPEN,kernel_open,iterations=1)
            fg_mask_processed=cv2.morphologyEx(fg_mask_processed,cv2.MORPH_CLOSE,kernel_close,iterations=2)
            ball_search_mask=None
            if last_known_ball_center and ball_lost_counter<MAX_BALL_LOST_FRAMES:
                ball_search_mask=np.zeros_like(fg_mask_processed)
                sx,sy=int(last_known_ball_center[0]-SEARCH_REGION_PADDING),int(last_known_ball_center[1]-SEARCH_REGION_PADDING)
                sw,sh=int(SEARCH_REGION_PADDING*2),int(SEARCH_REGION_PADDING*2)
                cv2.rectangle(ball_search_mask,(sx,sy),(sx+sw,sy+sh),255,-1)
                fg_mask_for_ball=cv2.bitwise_and(fg_mask_processed,fg_mask_processed,mask=ball_search_mask)
            else:fg_mask_for_ball=cv2.bitwise_and(fg_mask_processed,fg_mask_processed,mask=court_roi_mask_expanded)
            fg_mask_for_players=cv2.bitwise_and(fg_mask_processed,fg_mask_processed,mask=court_roi_mask_expanded)
            
            all_valid_contours_players=get_valid_contours(fg_mask_for_players,MIN_PLAYER_CONTOUR_AREA,MAX_PLAYER_CONTOUR_AREA)
            all_valid_contours_ball=get_valid_contours(fg_mask_for_ball,MIN_BALL_CONTOUR_AREA,MAX_BALL_CONTOUR_AREA)
            current_player_candidates={0:None,1:None}

            for contour in all_valid_contours_players:
                try:x_c,y_c,w_c,h_c=cv2.boundingRect(contour)
                except cv2.error:continue
                center_y_foot=y_c+h_c;candidate_data=(((x_c,y_c,w_c,h_c),(x_c+w_c//2,center_y_foot)))
                curr_bbox_area=w_c*h_c;pid_cand=0 if center_y_foot<court_center_y_line else 1
                if current_player_candidates[pid_cand]is None or curr_bbox_area>(current_player_candidates[pid_cand][0][2]*current_player_candidates[pid_cand][0][3]):
                    current_player_candidates[pid_cand]=candidate_data
                temp_player_bboxes.append((x_c,y_c,w_c,h_c))
            
            potential_balls_this_frame=[]
            for contour_ball in all_valid_contours_ball:
                area=cv2.contourArea(contour_ball);x_b,y_b,w_b,h_b=cv2.boundingRect(contour_ball)
                aspect_ratio_b=w_b/float(h_b)if h_b>0 else 0
                if BALL_MIN_WIDTH_HEIGHT<=w_b<=BALL_MAX_WIDTH_HEIGHT and BALL_MIN_WIDTH_HEIGHT<=h_b<=BALL_MAX_WIDTH_HEIGHT and BALL_MIN_ASPECT_RATIO<=aspect_ratio_b<=BALL_MAX_ASPECT_RATIO:
                    solidity=0
                    if len(contour_ball)>=5:hull=cv2.convexHull(contour_ball);hull_area=cv2.contourArea(hull);solidity=float(area)/hull_area if hull_area>0 else 0
                    if solidity>BALL_MIN_SOLIDITY or len(contour_ball)<5:
                        ball_center_candidate=(x_b+w_b//2,y_b+h_b//2)
                        if cv2.pointPolygonTest(video_court_boundary_points_original_roi,ball_center_candidate,False)<0:continue
                        if not any(is_point_inside_bbox(ball_center_candidate,p_bbox)for p_bbox in temp_player_bboxes):
                            potential_balls_this_frame.append({'bbox':(x_b,y_b,w_b,h_b),'center':ball_center_candidate,'area':area})
            
            best_ball_candidate_info=None
            if potential_balls_this_frame:
                if last_known_ball_center and ball_lost_counter<MAX_BALL_LOST_FRAMES:
                    min_dist_to_last=float('inf');temp_best_from_last=None
                    for ball_cand in potential_balls_this_frame:
                        dist=np.linalg.norm(np.array(last_known_ball_center)-np.array(ball_cand['center']))
                        if dist<MAX_BALL_JUMP_DISTANCE and dist<min_dist_to_last:min_dist_to_last=dist;temp_best_from_last=ball_cand
                    if temp_best_from_last:best_ball_candidate_info={'candidate':temp_best_from_last,'distance':min_dist_to_last,'from_last_known':True}
                if best_ball_candidate_info is None:
                    potential_balls_this_frame.sort(key=lambda b:b['area'],reverse=True)
                    if potential_balls_this_frame:
                        general_candidate=potential_balls_this_frame[0];dist_to_general=float('inf')
                        if last_known_ball_center:dist_to_general=np.linalg.norm(np.array(last_known_ball_center)-np.array(general_candidate['center']))
                        best_ball_candidate_info={'candidate':general_candidate,'distance':dist_to_general,'from_last_known':False}
            if best_ball_candidate_info:
                selected_ball=best_ball_candidate_info['candidate']
                detected_ball_info_this_frame=(selected_ball['bbox'],selected_ball['center'])
                last_known_ball_center=selected_ball['center'];ball_lost_counter=0
                sketch_ball_trail.append((selected_ball['center'], frame_count)) # İz için ekle
            else:
                ball_lost_counter+=1
                if ball_lost_counter>=MAX_BALL_LOST_FRAMES:last_known_ball_center=None
            
            new_tracked_players_this_frame={};active_player_ids_found_this_frame=[]
            for pid_tracked,old_data in tracked_players_dict.items():
                old_bbox,old_center,old_lost_count=old_data;candidate_data_for_this_id=current_player_candidates.get(pid_tracked);match_found=False
                if candidate_data_for_this_id:
                    cand_bbox,cand_center=candidate_data_for_this_id;distance=np.linalg.norm(np.array(old_center)-np.array(cand_center))
                    if distance<MAX_MATCH_DISTANCE or old_lost_count>MAX_LOST_FRAMES_ACTIVE_DETECTION//2:
                        new_tracked_players_this_frame[pid_tracked]=[cand_bbox,cand_center,0];active_player_ids_found_this_frame.append(pid_tracked);match_found=True
                if not match_found and old_lost_count+1<MAX_LOST_FRAMES_PLAYER:new_tracked_players_this_frame[pid_tracked]=[old_bbox,old_center,old_lost_count+1]
            for pid_candidate,candidate_data in current_player_candidates.items():
                if candidate_data and pid_candidate not in active_player_ids_found_this_frame and pid_candidate not in new_tracked_players_this_frame:
                    cand_bbox,cand_center=candidate_data;new_tracked_players_this_frame[pid_candidate]=[cand_bbox,cand_center,0]
            tracked_players_dict=new_tracked_players_this_frame

        cv2.line(current_frame_display,(0,int(court_center_y_line)),(frame_w,int(court_center_y_line)),(0,255,255),1)
        cv2.putText(current_frame_display,f"MidY:{int(court_center_y_line)}",(frame_w-150,frame_h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        video_points_for_sketch_mapping={0:None,1:None} # Topu buradan çıkardık, ayrı işlenecek
        
        # Oyuncuları video üzerine çiz
        for pid,data in tracked_players_dict.items():
            bbox,center_foot,lost_c=data;(x,y,w,h)=bbox;lbl=f"P{pid+1}"
            color_player = player_viz_colors_lost[pid] if lost_c > 0 and lost_c < MAX_LOST_FRAMES_PLAYER else player_viz_colors[pid]
            if lost_c == 0: # Sadece aktif oyuncuları çiz
                cv2.rectangle(current_frame_display,(x,y),(x+w,y+h),color_player,2)
                cv2.putText(current_frame_display,lbl,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color_player,1)
            if lost_c < MAX_LOST_FRAMES_PLAYER : video_points_for_sketch_mapping[pid]=center_foot # Kroki için her zaman ekle (kayıp olsa bile)
        
        # Topu video üzerine çiz (sadece bu karede bulunduysa)
        if detected_ball_info_this_frame:
            bbox_b,center_b=detected_ball_info_this_frame;(xb,yb,wb,hb)=bbox_b
            cv2.rectangle(current_frame_display,(xb,yb),(xb+wb,yb+hb),ball_viz_color,2)

        # Kroki için top izini güncelle ve çiz
        sketch_ball_trail = [(pos, fc) for pos, fc in sketch_ball_trail if (frame_count - fc) < BALL_FADE_DURATION_SKETCH]
        if len(sketch_ball_trail) > SKETCH_BALL_HISTORY_LEN: sketch_ball_trail = sketch_ball_trail[-SKETCH_BALL_HISTORY_LEN:]
        
        # Kroki için çizilecek noktaları ve renkleri hazırla
        pts_to_draw_on_sketch = []
        colors_for_sketch = []
        radii_for_sketch = []

        # Oyuncuları kroki için ekle
        for pid_k in [0,1]:
            if video_points_for_sketch_mapping.get(pid_k):
                pts_to_draw_on_sketch.append(video_points_for_sketch_mapping[pid_k])
                is_lost_k = (pid_k in tracked_players_dict and tracked_players_dict[pid_k][2]>0 and tracked_players_dict[pid_k][2]<MAX_LOST_FRAMES_PLAYER)
                colors_for_sketch.append(player_viz_colors_lost[pid_k] if is_lost_k else player_viz_colors[pid_k])
                radii_for_sketch.append(6 if is_lost_k else 7)
        
        # Top izini kroki için ekle
        transformed_ball_trail_sketch = []
        if sketch_ball_trail:
            ball_trail_points_video = [pos for pos, fc in sketch_ball_trail]
            transformed_ball_trail_sketch = transform_points_for_sketch(ball_trail_points_video, homography_matrix_for_sketch)
        
        for i, (original_ball_pos_video, fc_ball) in enumerate(sketch_ball_trail):
            if i < len(transformed_ball_trail_sketch): # Dönüşüm başarılıysa
                t_c_ball_sketch = transformed_ball_trail_sketch[i]
                age = frame_count - fc_ball
                fade_factor = max(0.1, 1.0 - (age / float(BALL_FADE_DURATION_SKETCH)))
                
                base_ball_color_np = np.array(ball_viz_color, dtype=np.float32) # Önce NumPy array'e çevir
                faded_color_np = (base_ball_color_np * fade_factor).astype(np.uint8) # uint8'e çevir (0-255 aralığı)
                faded_color_tuple = tuple(faded_color_np.tolist()) # Python tuple'ına çevir
                
                faded_color = tuple((np.array(ball_viz_color) * fade_factor).astype(int))
                radius = 5 if age < 2 else (4 if age < SKETCH_BALL_HISTORY_LEN // 2 else 3)
                cv2.circle(sketch_display, t_c_ball_sketch, radius, faded_color_tuple, -1) # Direkt çiz

        # Sadece oyuncu noktalarını (eğer varsa) dönüştür ve çiz (top zaten çizildi)
        if pts_to_draw_on_sketch: # Bu liste artık sadece oyuncuları içermeli
            transformed_player_points_sketch = transform_points_for_sketch(pts_to_draw_on_sketch, homography_matrix_for_sketch)
            for i, t_c_player_sketch in enumerate(transformed_player_points_sketch):
                 cv2.circle(sketch_display, t_c_player_sketch, radii_for_sketch[i], colors_for_sketch[i], -1)
        
        cv2.imshow("Video Tespiti", current_frame_display)
        cv2.imshow("Kort Krokisi", sketch_display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): cv2.waitKey(0)
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()