import cv2
import whisper
import subprocess
import os
import mediapipe as mp
import numpy as np
import uuid
import random
import requests
import base64
import json
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MASTER CONFIG ---
class Config:
    # Paths
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 
    MODEL_PATH = 'face_landmarker.task'        
    INPUT_VIDEO = "podcast3.mp4"
    OUTPUT_DIR = "output"
    FINAL_DIR = "final_video"
    OLLAMA_MODEL = "llama3.2:1b"
    OLLAMA_VISION_MODEL = "moondream"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    
    # Timing & Generation
    TARGET_DURATION = 35 
    MIN_DURATION = 15                              
    MAX_DURATION = 59 
    NUM_SHORTS = 3
    
    # Viral Detection
    VIRAL_KEYWORDS = [
        "actually", "secret", "shocking", "money", "insane", "google", "failed", 
        "khabib", "speed", "imagine", "impossible", "wrong", "stop", "truth", 
        "never", "must", "scam", "billion", "mistake", "warning"
    ]
    
    # Face Tracking
    MOUTH_THRESH_DEFAULT = 0.012    
    MOUTH_THRESH_FACTOR = 1.5 
    
    # Subtitle Style (Viral Pop Style)
    SUBTITLE_STYLE = (
        "FontName=Arial Black,FontSize=28,PrimaryColour=&H0000FFFF," # Yellow
        "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=2,"
        "Alignment=2,MarginV=100" # Bottom-Center but slightly higher
    )
    
    DEBUG = True

# Ensure directories exist
for d in [Config.OUTPUT_DIR, Config.FINAL_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- UTILS ---

def format_timestamp(seconds):
    td = datetime.fromtimestamp(seconds) - datetime.fromtimestamp(0)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

def get_mouth_dist(landmarks):
    return abs(landmarks[13].y - landmarks[14].y)

def call_ollama(prompt, system="", images=None, model_override=None):
    """Hits the local Ollama API for text or vision tasks."""
    payload = {
        "model": model_override if model_override else Config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"num_predict": 1000, "temperature": 0.1} # Lower temp for stable JSON
    }
    if images:
        payload["images"] = images
    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=300) # Increased timeout for CPU
        return response.json().get("response", "")
    except Exception as e:
        print(f"⚠️ Ollama Error: {e}")
        return ""

# --- AI CORE ---

def scout_viral_clips(video_path):
    """Uses Whisper for text and Ollama for 'Editor' level logic."""
    print("📡 Scout: Analyzing narrative with Ollama AI...")
    model = whisper.load_model("tiny")
    result = model.transcribe(video_path, fp16=False)
    
    # 1. Prepare Transcript for Ollama
    transcript = ""
    for s in result['segments']:
        transcript += f"[{s['start']:.2f}s - {s['end']:.2f}s] {s['text']}\n"
    
    # Limit transcript size for Ollama to prevent timeouts
    if len(transcript) > 8000:
        print("📝 Transcript is long, sampling for the AI...")
        transcript = transcript[:8000] + "... [truncated]"
    
    system_prompt = (
        "You are a viral video editor. Your job is to find the MOST interesting parts of a podcast transcript. "
        "Respond ONLY with a valid JSON array of objects. No intro, no outro. "
        "Format: [{'start': seconds, 'end': seconds, 'reason': 'description'}] "
        "Keep segments between 15-45 seconds. Limit to 3 hook segments."
    )
    
    prompt = f"Analyze this transcript and find 3 viral hooks:\n\n{transcript}"
    
    print("🧠 Thinking (Ollama)...")
    raw_response = call_ollama(prompt, system=system_prompt)
    
    found = []
    try:
        # 1. Robust JSON extraction
        clean_str = raw_response.strip()
        # Remove markdown if present
        if "```json" in clean_str:
            clean_str = clean_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_str:
            clean_str = clean_str.split("```")[1].split("```")[0].strip()
        
        # Pull only content between [ and ]
        s_idx = clean_str.find('[')
        e_idx = clean_str.rfind(']')
        if s_idx != -1 and e_idx != -1:
            clean_str = clean_str[s_idx:e_idx+1]
            
        ollama_hooks = json.loads(clean_str)
        
        # 2. Re-align with Whisper word data for rendering
        print("✅ Ollama identified the hooks. Aligning timestamps...")
        for hook in ollama_hooks:
            best_segment = min(result['segments'], key=lambda x: abs(x['start'] - hook['start']))
            found.append({
                'start': best_segment['start'], 
                'text': best_segment['text'], 
                'words': best_segment.get('words', []),
                'score': 100
            })
            print(f"   📍 Hook at {best_segment['start']:.2f}s: {hook.get('reason', 'AI selected')}")
            
    except Exception as e:
        print(f"⚠️ Narrative Scouting Failed: {e}. Falling back to Keyword Score.")
        candidates = []
        for s in result['segments']:
            text = s['text'].lower()
            sc = sum(10 for k in Config.VIRAL_KEYWORDS if k in text)
            dur = s['end'] - s['start']
            if dur > 0 and 3.0 <= (len(text.split()) / dur) <= 5.0: sc += 15
            if sc > 0:
                candidates.append({'start': s['start'], 'text': s['text'], 'words': s.get('words', []), 'score': sc})
        
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        top_c = candidates[:15]
        random.shuffle(top_c)
        for cand in top_c:
            if all(abs(cand['start'] - f['start']) > 120 for f in found):
                found.append(cand)
            if len(found) >= Config.NUM_SHORTS: break
        if len(found) < Config.NUM_SHORTS:
            for c in top_c:
                if c not in found: found.append(c)
                if len(found) >= Config.NUM_SHORTS: break

    return sorted(found, key=lambda x: x['start'])

def get_scene_hint_ollama(video_path, start_s):
    """Uses Ollama Vision to understand the scene layout."""
    print("👁️ Vision: Analyzing scene layout with Ollama...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    success, frame = cap.read()
    cap.release()
    if not success: return "SOLO"
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    prompt = (
        "Identify the scene framing: Is it a WIDE shot (two people), a SOLO shot (one person), or a SPLIT-SCREEN interview? "
        "Answer ONLY with one of these words: WIDE, SOLO, SPLIT."
    )
    
    res = call_ollama(prompt, images=[img_str], model_override=Config.OLLAMA_VISION_MODEL)
    hint = res.strip().upper()
    if "WIDE" in hint: return "STACK" # We use STACK for wide shots
    elif "SPLIT" in hint: return "STACK"
    return "SOLO"

def calibrate_mouth(video_path, sample_start=10):
    """Determines dynamic mouth threshold."""
    print(f"🔍 Calibrating mouth threshold...")
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=Config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    distances = []
    for i in range(20): # Sample 2 seconds
        cap.set(cv2.CAP_PROP_POS_MSEC, (sample_start + i * 0.1) * 1000)
        success, frame = cap.read()
        if not success: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        if res.face_landmarks:
            for lms in res.face_landmarks:
                distances.append(get_mouth_dist(lms))
    cap.release()
    
    if not distances: return Config.MOUTH_THRESH_DEFAULT
    dynamic_thresh = np.median(distances) * Config.MOUTH_THRESH_FACTOR
    print(f"📊 Calibration: Threshold set to {dynamic_thresh:.4f}")
    return dynamic_thresh

def find_home_seats(video_path, start_s):
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=Config.MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)
    l_seats, r_seats = [], []
    for i in range(15):
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
    return float(np.median(l_seats) if l_seats else 0.25), float(np.median(r_seats) if r_seats else 0.75)

def analyze_activity(video_path, start_s, h_lx, h_rx, mouth_thresh, hint="SOLO"):
    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path=Config.MODEL_PATH)
    # Increase num_faces to 4 to catch everyone in wide shots
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=4)
    detector = vision.FaceLandmarker.create_from_options(options)
    activity_map = []
    
    # Last known good positions for smoothing/fallbacks
    last_lx, last_rx = h_lx, h_rx
    last_solo_cx = h_lx
    
    # HYSTERESIS CONFIG
    locked_side = "BOTH" if hint == "STACK" else (None if hint == "SOLO" else None)
    lock_timer = 0 if hint == "SOLO" else 10 # Start with some momentum if hint is STACK
    LOCK_DURATION = 3  # Stay locked for 1.5 seconds after speech ends
    
    for i in range(0, int(Config.TARGET_DURATION * 2)):
        current_ms = (start_s + i * 0.5) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
        success, frame = cap.read()
        if not success: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        faces, speakers = [], []
        
        if res.face_landmarks:
            for lms in res.face_landmarks:
                cx = float(sum([l.x for l in lms]) / len(lms))
                faces.append(cx)
                if get_mouth_dist(lms) > mouth_thresh:
                    side = "LEFT" if abs(cx - h_lx) < abs(cx - h_rx) else "RIGHT"
                    speakers.append(side)
        
        # SPEAKER LOCKING LOGIC
        if len(speakers) == 1:
            locked_side = speakers[0]
            lock_timer = LOCK_DURATION
        elif len(speakers) > 1:
            locked_side = "BOTH" # Signal for STACK
            lock_timer = LOCK_DURATION
        else:
            if lock_timer > 0:
                lock_timer -= 1
            else:
                locked_side = None
                
        # MODE DECISION BASED ON LOCK
        if locked_side == "BOTH":
            sorted_f = sorted(faces) if faces else [h_lx, h_rx]
            last_lx, last_rx = (sorted_f[0], sorted_f[-1]) if len(sorted_f) >= 2 else (h_lx, h_rx)
            mode, coords = 'STACK', (last_lx, last_rx)
        elif locked_side in ["LEFT", "RIGHT"]:
            target_seat = h_lx if locked_side == "LEFT" else h_rx
            closest_face = min(faces, key=lambda x: abs(x - target_seat)) if faces else target_seat
            last_solo_cx = closest_face
            mode, coords = 'SOLO', (closest_face,)
        elif len(faces) >= 2:
            # Wide Shot naturally -> STACK
            sorted_f = sorted(faces)
            last_lx, last_rx = sorted_f[0], sorted_f[-1]
            mode, coords = 'STACK', (last_lx, last_rx)
        elif len(faces) == 1:
            last_solo_cx = faces[0]
            mode, coords = 'SOLO', (faces[0],)
        else:
            mode, coords = 'SOLO', (last_solo_cx,)
            
        activity_map.append({'mode': mode, 'coords': coords, 'type': 'SPEAKER' if speakers else 'DEFAULT'})
    cap.release()
    return activity_map

# --- RENDERING & POLISHING ---

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
    subprocess.run(cmd, capture_output=True)
    return temp_path

def add_captions(video_path, start_s, output_path):
    """Re-transcribes slightly more accurately or uses passed word data to burn captions."""
    print(f"✍️ Adding styled captions to {os.path.basename(video_path)}...")
    unique_id = uuid.uuid4().hex[:6]
    srt_path = f"captions_{unique_id}.srt"
    
    # We re-run Whisper tiny with word timestamps for the specific segment to ensure alignment
    model = whisper.load_model("tiny")
    result = model.transcribe(video_path, fp16=False, word_timestamps=True)
    
    with open(srt_path, "w", encoding="utf-8") as f:
        counter = 1
        for segment in result['segments']:
            words = segment.get('words', [])
            # Group words in pairs for fast-paced viral look
            for i in range(0, len(words), 2): 
                chunk = words[i:i+2]
                if not chunk: continue
                start_t = format_timestamp(chunk[0]['start'])
                end_t = format_timestamp(chunk[-1]['end'])
                text = " ".join([w['word'].strip().upper() for w in chunk])
                f.write(f"{counter}\n{start_t} --> {end_t}\n{text}\n\n")
                counter += 1

    srt_fixed = srt_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        Config.FFMPEG_PATH, "-y", "-i", video_path,
        "-vf", f"subtitles='{srt_fixed}':force_style='{Config.SUBTITLE_STYLE}'",
        "-c:v", "libx264", "-crf", "22", "-c:a", "copy", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(srt_path): os.remove(srt_path)
    return output_path

# --- RUNNER ---

def run():
    print("\n" + "🔥" * 20)
    print("🚀 MASTER AI SHORTS FACTORY v1.0")
    print("🔥" * 20)
    
    time_stamp = datetime.now().strftime("%m%d_%H%M")
    
    # 1. Scout
    hooks = scout_viral_clips(Config.INPUT_VIDEO)
    if not hooks: return
    
    # 2. Global Calibration
    mouth_thresh = calibrate_mouth(Config.INPUT_VIDEO)
    
    for idx, hook in enumerate(hooks):
        start_s = hook['start']
        print(f"\n🎬 Processing Viral Short #{idx+1} (Start: {start_s:.2f}s)")
        
        # 3. Analyze Activity
        lx, rx = find_home_seats(Config.INPUT_VIDEO, start_s)
        scene_hint = get_scene_hint_ollama(Config.INPUT_VIDEO, start_s)
        print(f"👁️ AI Logic Hint: {scene_hint} mode suggested.")
        activity = analyze_activity(Config.INPUT_VIDEO, start_s, lx, rx, mouth_thresh, hint=scene_hint)
        
        # 4. Group Activity with Coordinate Averaging
        blocks = []
        if not activity: continue
        curr = activity[0]
        curr_block_coords = [curr['coords']]
        dur = 0.5
        for entry in activity[1:]:
            # We group by mode. We allow slight coordinate drift within a block.
            if entry['mode'] == curr['mode']: 
                dur += 0.5
                curr_block_coords.append(entry['coords'])
            else:
                # Calculate average coords for the block to keep the crop stable
                if curr['mode'] == 'STACK':
                    # Filter to only STACK coords to be safe (though mode check should handle it)
                    stack_coords = [c for c in curr_block_coords if len(c) == 2]
                    if not stack_coords: stack_coords = [(lx, rx)]
                    avg_lx = np.mean([c[0] for c in stack_coords])
                    avg_rx = np.mean([c[1] for c in stack_coords])
                    final_coords = (float(avg_lx), float(avg_rx))
                else:
                    # SOLO coords are usually 1-element tuples
                    solo_coords = [c[0] for c in curr_block_coords]
                    final_coords = (float(np.mean(solo_coords)),)
                
                blocks.append((curr['mode'], final_coords, dur))
                curr, dur = entry, 0.5
                curr_block_coords = [curr['coords']]
        
        # Last block
        if curr['mode'] == 'STACK':
            stack_coords = [c for c in curr_block_coords if len(c) == 2]
            if not stack_coords: stack_coords = [(lx, rx)]
            avg_lx = np.mean([c[0] for c in stack_coords])
            avg_rx = np.mean([c[1] for c in stack_coords])
            final_coords = (float(avg_lx), float(avg_rx))
        else:
            solo_coords = [c[0] for c in curr_block_coords]
            final_coords = (float(np.mean(solo_coords)),)
        blocks.append((curr['mode'], final_coords, dur))

        # 5. Render Base Segments
        print(f"🎞️ Rendering crops...")
        seg_files = [render_segment(start_s + sum(b[2] for b in blocks[:i]), b[2], b[0], b[1], idx, i) for i, b in enumerate(blocks)]

        # 6. Concat
        list_txt = os.path.join(Config.OUTPUT_DIR, f"list_{idx}.txt")
        with open(list_txt, 'w') as f:
            for sf in seg_files: f.write(f"file '{os.path.abspath(sf)}'\n")

        raw_short = os.path.join(Config.OUTPUT_DIR, f"raw_{idx}.mp4")
        subprocess.run([Config.FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', list_txt, '-c', 'copy', raw_short], capture_output=True)
        
        # 7. Add Viral Captions
        final_out = os.path.join(Config.FINAL_DIR, f"Viral_Short_{time_stamp}_{idx+1}.mp4")
        add_captions(raw_short, start_s, final_out)
        
        # 8. Cleanup
        for sf in seg_files: os.remove(sf)
        os.remove(list_txt)
        os.remove(raw_short)
        
        print(f"🌟 DONE! Short Ready: {os.path.basename(final_out)}")

    print("\n" + "="*50)
    print(f"✅ FACTORY COMPLETE. Check: {os.path.abspath(Config.FINAL_DIR)}")
    print("="*50)

if __name__ == "__main__":
    run()
