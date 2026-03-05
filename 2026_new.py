import cv2
import whisper
import subprocess
import os
import mediapipe as mp
import numpy as np
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 2026 ALGORITHM CONFIG ---
class Config:
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 
    MODEL_PATH = 'face_landmarker.task'        
    INPUT_VIDEO = "podcast3.mp4"
    OUTPUT_DIR = "output"
    
    # Timing
    TARGET_DURATION = 35 
    MIN_DURATION = 15
    MAX_DURATION = 59 
    
    # Detection
    MOUTH_THRESH_DEFAULT = 0.012    
    MOUTH_THRESH_FACTOR = 1.5 # Multiplier for dynamic threshold
    FADE_DURATION = 0.1 
    
    # System
    DEBUG = True

if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)

# --- ALGORITHM VALIDATOR ---

def check_shorts_eligibility(file_path):
    """Checks if the video meets 2026 Shorts technical standards."""
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    is_vertical = height > width
    is_valid_length = Config.MIN_DURATION <= duration <= Config.MAX_DURATION
    
    print(f"--- 🛠️ Technical Audit: {os.path.basename(file_path)} ---")
    print(f"✅ Vertical: {is_vertical} ({width}x{height})")
    print(f"✅ Duration: {duration:.2f}s (Target: {Config.TARGET_DURATION}s)")
    
    if not is_vertical: print("⚠️ WARNING: Video is not vertical. Algorithm may ignore it.")
    if duration > 60: print("⚠️ WARNING: Over 60s. Will be treated as Long-form.")
    
    return is_vertical and is_valid_length

# --- AI EDITING CORE ---

def get_mouth_dist(landmarks):
    return abs(landmarks[13].y - landmarks[14].y)

def calculate_dynamic_threshold(video_path, sample_start=10, sample_duration=2):
    """Samples frames to find a baseline mouth distance and set a dynamic threshold."""
    print(f"🔍 Calibrating mouth threshold at {sample_start}s...")
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=Config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    distances = []
    for i in range(int(sample_duration * 10)): # 10 frames per second sample
        cap.set(cv2.CAP_PROP_POS_MSEC, (sample_start + i * 0.1) * 1000)
        success, frame = cap.read()
        if not success: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        if res.face_landmarks:
            for lms in res.face_landmarks:
                distances.append(get_mouth_dist(lms))
    
    cap.release()
    if not distances:
        print("⚠️ Calibration failed (no faces). Using default threshold.")
        return Config.MOUTH_THRESH_DEFAULT
    
    baseline = np.median(distances)
    dynamic_thresh = baseline * Config.MOUTH_THRESH_FACTOR
    print(f"📊 Calibration Complete: Baseline={baseline:.4f}, Threshold={dynamic_thresh:.4f}")
    return dynamic_thresh

def find_home_seats(video_path, start_s):
    print(f"📍 Finding home seats for segment starting at {start_s}s...")
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=Config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    l_seats, r_seats = [], []
    for i in range(20):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_s + i) * 1000)
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
    lx, rx = float(np.median(l_seats) if l_seats else 0.25), float(np.median(r_seats) if r_seats else 0.75)
    print(f"✅ Seats Found: Left={lx:.2f}, Right={rx:.2f}")
    return lx, rx

def analyze_activity(video_path, start_s, h_lx, h_rx, mouth_thresh):
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=Config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    activity_map = []
    
    for i in range(0, Config.TARGET_DURATION * 2):
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
                if get_mouth_dist(lms) > mouth_thresh:
                    speakers.append("LEFT" if abs(cx - h_lx) < abs(cx - h_rx) else "RIGHT")
        
        if len(faces) >= 2: mode, coords = 'STACK', (h_lx, h_rx)
        elif speakers: mode, coords = 'SOLO', ((h_lx,) if speakers[0] == "LEFT" else (h_rx,))
        else: mode, coords = 'SOLO', (h_lx,) # Default
        activity_map.append({'mode': mode, 'coords': coords})
    cap.release()
    return activity_map

def render_segment(start_t, dur, mode, coords, clip_id, seg_id):
    temp_path = os.path.join(Config.OUTPUT_DIR, f"tmp_{clip_id}_{seg_id}.mp4")
    if mode == 'STACK':
        lx, rx = coords
        base = (f"[0:v]crop=ih*(9/8):ih:iw*{lx}-(ih*(9/8)/2):0,scale=1080:960,setsar=1[t]; "
                f"[0:v]crop=ih*(9/8):ih:iw*{rx}-(ih*(9/8)/2):0,scale=1080:960,setsar=1[b]; [t][b]vstack=inputs=2")
    else:
        cx = coords[0]
        base = f"crop=ih*(9/16):ih:iw*{cx}-(ih*(9/16)/2):0,scale=1080:1920,setsar=1"
    
    cmd = [Config.FFMPEG_PATH, '-y', '-ss', str(start_t), '-t', str(dur), '-i', Config.INPUT_VIDEO,
           '-filter_complex' if mode == 'STACK' else '-vf', base,
           '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22', '-c:a', 'aac', '-threads', '0', temp_path]
    
    # Run silently unless error
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 and Config.DEBUG:
        print(f"❌ FFmpeg Error: {res.stderr.decode()}")
    
    return temp_path

def run():
    print("\n" + "="*50)
    print("🚀 STARTING ALGO-OPTIMIZED GENERATION (v2026.2)")
    print("="*50)
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    
    # 1. Calibration
    mouth_thresh = calculate_dynamic_threshold(Config.INPUT_VIDEO)
    
    # 2. Transcription
    print("📡 Loading Whisper model...")
    model = whisper.load_model("tiny")
    print(f"📝 Transcribing {Config.INPUT_VIDEO}...")
    result = model.transcribe(Config.INPUT_VIDEO, fp16=False)
    
    # 2026 Hook Words
    hooks = ["khabib", "speed", "insane", "imagine", "secret", "actually"]
    starts = [s['start'] for s in result['segments'] if any(h in s['text'].lower() for h in hooks)][:3]

    if not starts:
        print("⚠️ No hooks found. Try adding more keywords to the 'hooks' list.")
        return

    for idx, start_s in enumerate(starts):
        print(f"\n🎬 Creating Algo-Short #{idx+1} of {len(starts)} (Start: {start_s}s)...")
        lx, rx = find_home_seats(Config.INPUT_VIDEO, start_s)
        activity = analyze_activity(Config.INPUT_VIDEO, start_s, lx, rx, mouth_thresh)
        
        # Group activity into blocks
        blocks = []
        if not activity: continue
        
        curr = activity[0]
        dur = 0.5
        for entry in activity[1:]:
            if entry['mode'] == curr['mode'] and entry['coords'] == curr['coords']: dur += 0.5
            else:
                blocks.append((curr['mode'], curr['coords'], dur))
                curr, dur = entry, 0.5
        blocks.append((curr['mode'], curr['coords'], dur))

        print(f"🎞️ Rendering {len(blocks)} segments...")
        seg_files = [render_segment(start_s + sum(b[2] for b in blocks[:i]), b[2], b[0], b[1], idx, i) for i, b in enumerate(blocks)]

        list_txt = os.path.join(Config.OUTPUT_DIR, f"list_{idx}.txt")
        with open(list_txt, 'w') as f:
            for sf in seg_files: f.write(f"file '{os.path.abspath(sf)}'\n")

        final_out = os.path.join(Config.OUTPUT_DIR, f"Short_{time_stamp}_{idx+1}.mp4")
        print(f"📦 Merging into final video: {os.path.basename(final_out)}")
        subprocess.run([Config.FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', list_txt, '-c', 'copy', final_out], capture_output=True)
        
        # CLEANUP & AUDIT
        for sf in seg_files: 
            if os.path.exists(sf): os.remove(sf)
        if os.path.exists(list_txt): os.remove(list_txt)
        
        # 2026 Final Eligibility Check
        check_shorts_eligibility(final_out)

    print("\n✅ GENERATION COMPLETE. READY FOR UPLOAD.")
    print(f"📂 Output Folder: {os.path.abspath(Config.OUTPUT_DIR)}")

if __name__ == "__main__":
    run()