import cv2
import whisper
import subprocess
import os
import mediapipe as mp
import numpy as np
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 
MODEL_PATH = 'face_landmarker.task'        
INPUT_VIDEO = "podcast2.mp4"
OUTPUT_DIR = "output"
CLIP_DURATION = 60 

MOUTH_THRESH = 0.012    
FADE_DURATION = 0.2     

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# --- AI CORE ---

def get_mouth_dist(landmarks):
    return abs(landmarks[13].y - landmarks[14].y)

def find_home_seats(video_path, start_s):
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    l_seats, r_seats = [], []
    for i in range(25):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_s + i * 2) * 1000)
        success, frame = cap.read()
        if not success: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        if res.face_landmarks:
            coords = sorted([sum([l.x for l in lms]) / len(lms) for lms in res.face_landmarks])
            if len(coords) >= 2:
                l_seats.append(coords[0]); r_seats.append(coords[-1])
            elif len(coords) == 1:
                if coords[0] < 0.5: l_seats.append(coords[0])
                else: r_seats.append(coords[0])
    cap.release()
    return float(np.median(l_seats) if l_seats else 0.25), float(np.median(r_seats) if r_seats else 0.75)

def analyze_activity(video_path, start_s, h_lx, h_rx):
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    activity_map = []
    curr_mode, curr_coords = 'SOLO', (h_lx,)
    for i in range(0, CLIP_DURATION * 2):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_s + i * 0.5) * 1000)
        success, frame = cap.read()
        if not success: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        faces, speakers = [], []
        if res.face_landmarks:
            for lms in res.face_landmarks:
                cx = sum([l.x for l in lms]) / len(lms)
                faces.append(cx)
                if get_mouth_dist(lms) > MOUTH_THRESH:
                    speakers.append("LEFT" if abs(cx - h_lx) < abs(cx - h_rx) else "RIGHT")
        if len(faces) >= 2: mode, coords = 'STACK', (h_lx, h_rx)
        elif speakers: mode, coords = 'SOLO', ((h_lx,) if speakers[0] == "LEFT" else (h_rx,))
        elif len(faces) == 1: mode, coords = 'SOLO', ((h_lx,) if abs(faces[0] - h_lx) < abs(faces[0] - h_rx) else (h_rx,))
        else: mode, coords = curr_mode, curr_coords
        activity_map.append({'mode': mode, 'coords': coords})
        curr_mode, curr_coords = mode, coords
    cap.release()
    return activity_map

def render_segment(start_t, dur, mode, coords, clip_id, seg_id):
    # Temp files now saved directly in OUTPUT_DIR
    temp_path = os.path.join(OUTPUT_DIR, f"temp_seg_{clip_id}_{seg_id}.mp4")
    if mode == 'STACK':
        lx, rx = coords
        crop_w = "ih*(9/8)"
        base = (f"[0:v]crop={crop_w}:ih:iw*{lx}-({crop_w}/2):0,scale=1080:960,setsar=1[t]; "
                f"[0:v]crop={crop_w}:ih:iw*{rx}-({crop_w}/2):0,scale=1080:960,setsar=1[b]; [t][b]vstack=inputs=2")
    else:
        cx = coords[0]
        crop_w = "ih*(9/16)"
        base = f"crop={crop_w}:ih:iw*{cx}-({crop_w}/2):0,scale=1080:1920,setsar=1"
    v_filter = f"{base},fade=t=in:st=0:d={FADE_DURATION},fade=t=out:st={dur-FADE_DURATION}:d={FADE_DURATION}"
    cmd = [FFMPEG_PATH, '-y', '-ss', str(start_t), '-t', str(dur), '-i', INPUT_VIDEO,
           '-filter_complex' if mode == 'STACK' else '-vf', v_filter,
           '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-c:a', 'aac', temp_path]
    subprocess.run(cmd, capture_output=True)
    return temp_path

def run():
    print("--- 🚀 STARTING AI VIDEO GENERATION ---")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model = whisper.load_model("tiny")
    result = model.transcribe(INPUT_VIDEO, fp16=False)
    hooks = ["actually", "crazy", "imagine", "lesson", "secret", "wow"]
    starts = [s['start'] for s in result['segments'] if any(h in s['text'].lower() for h in hooks)][:5]

    for idx, start_s in enumerate(starts):
        print(f"🎬 Processing Short #{idx+1}...")
        lx, rx = find_home_seats(INPUT_VIDEO, start_s)
        activity = analyze_activity(INPUT_VIDEO, start_s, lx, rx)
        
        blocks = []
        curr = activity[0]
        dur = 0.5
        for entry in activity[1:]:
            if entry['mode'] == curr['mode'] and entry['coords'] == curr['coords']:
                dur += 0.5
            else:
                blocks.append((curr['mode'], curr['coords'], dur))
                curr, dur = entry, 0.5
        blocks.append((curr['mode'], curr['coords'], dur))

        seg_files = []
        curr_time = start_s
        for i, (mode, coords, d) in enumerate(blocks):
            seg_files.append(render_segment(curr_time, d, mode, coords, idx, i))
            curr_time += d

        list_txt = os.path.join(OUTPUT_DIR, f"list_{time_stamp}_{idx}.txt")
        with open(list_txt, 'w') as f:
            for sf in seg_files: f.write(f"file '{os.path.abspath(sf)}'\n")

        final_filename = f"Short_{time_stamp}_{idx+1}.mp4"
        final_out = os.path.join(OUTPUT_DIR, final_filename)
        
        subprocess.run([FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', list_txt, '-c', 'copy', final_out])
        
        # Cleanup temp files immediately
        for sf in seg_files: 
            if os.path.exists(sf): os.remove(sf)
        os.remove(list_txt)

    print("\n" + "="*50)
    print("✅ DONE! VIDEOS CREATED.")
    print(f"📂 Folder: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)

if __name__ == "__main__":
    run()