# --- CRITICAL CONFIG ---
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

import cv2
import whisper
import subprocess
import os
import mediapipe as mp
import numpy as np
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STANDARD MOVIEPY 1.0.3 IMPORTS
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, concatenate_videoclips

# --- CONFIG ---
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 
MODEL_PATH = 'face_landmarker.task'         
INPUT_VIDEO = "podcast4.mp4"
OUTPUT_DIR = "output"
TARGET_DURATION = 35 

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_face_center(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success: 
        cap.release()
        return 0.5
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_image)
    face_x = 0.5
    if res.face_landmarks:
        face_x = sum([l.x for l in res.face_landmarks[0]]) / len(res.face_landmarks[0])
    cap.release()
    return face_x

def apply_viral_edits(input_path, output_path, hook_text):
    print(f"🎬 Rendering: {hook_text}")
    
    with VideoFileClip(input_path) as clip:
        w, h = clip.size
        
        # --- THE BULLETPROOF FIX: FORCE BOTH TO EVEN ---
        # 1. Start with an even height
        final_h = h if h % 2 == 0 else h - 1
        # 2. Calculate width for 9:16 and force to even
        final_w = int(final_h * (9/16))
        if final_w % 2 != 0: final_w -= 1
        
        face_x_percent = get_face_center(input_path)
        center_x = int(w * face_x_percent)
        
        # Calculate x1 and ensure it doesn't push the crop out of bounds
        x1 = max(0, min(center_x - final_w//2, w - final_w))
        
        # Use the standard crop with our safe dimensions
        clip_vertical = clip.crop(x1=x1, y1=0, width=final_w, height=final_h)
        
        duration = clip_vertical.duration
        segments = []
        for i in range(0, int(duration), 2):
            start = i
            end = min(i + 2, duration)
            seg = clip_vertical.subclip(start, end)
            
            # Retention zoom
            if (i // 2) % 2 == 0:
                seg = seg.resize(1.12)
                # After resizing, we must re-crop to the exact original safe dimensions
                # to prevent the "odd pixel" issue from coming back
                seg = seg.crop(x_center=seg.w/2, y_center=seg.h/2, width=final_w, height=final_h)
                
            segments.append(seg)
        
        final_v = concatenate_videoclips(segments)

        # Header Hook
        hook_box = (ColorClip(size=(final_w, 140), color=(0,0,0))
                    .set_opacity(0.7)
                    .set_duration(min(duration, 5)))
        
        txt_hook = (TextClip(hook_text, fontsize=50, color='yellow', font='Arial-Bold',
                             method='caption', size=(final_w * 0.8, None))
                    .set_duration(min(duration, 5)))
        
        header = CompositeVideoClip([hook_box, txt_hook.set_position('center')])
        header = header.set_position(('center', 100))

        result = CompositeVideoClip([final_v, header])
        
        # Final Output
        result.write_videofile(
            output_path, 
            fps=30, 
            codec="libx264", 
            audio_codec="aac",
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        result.close()

def run():
    print("--- 🚀 GENERATING 3 VIRAL SHORTS (V3 STABLE) ---")
    model = whisper.load_model("tiny")
    result = model.transcribe(INPUT_VIDEO)
    
    viral_keywords = ["actually", "secret", "shocking", "money", "insane", "google", "failed"]
    
    found_segments = []
    for s in result['segments']:
        if any(k in s['text'].lower() for k in viral_keywords):
            # 60s gap to ensure variety
            if not found_segments or (s['start'] - found_segments[-1]['start'] > 60):
                found_segments.append({'start': s['start'], 'text': s['text'][:30].upper() + "..."})
        if len(found_segments) >= 3: break

    for idx, meta in enumerate(found_segments):
        print(f"\n📦 Processing Short #{idx+1} of {len(found_segments)}...")
        start_t = meta['start']
        temp_segment = os.path.join(OUTPUT_DIR, f"temp_{idx}.mp4")
        final_out = os.path.join(OUTPUT_DIR, f"Viral_Short_{idx+1}.mp4")

        subprocess.run([FFMPEG_PATH, '-y', '-ss', str(start_t), '-t', str(TARGET_DURATION), 
                        '-i', INPUT_VIDEO, '-c', 'copy', temp_segment], capture_output=True)

        try:
            apply_viral_edits(temp_segment, final_out, meta['text'])
        except Exception as e:
            print(f"❌ Error on Short #{idx+1}: {e}")
        finally:
            if os.path.exists(temp_segment):
                try: os.remove(temp_segment)
                except: pass

    print("\n✅ PROCESS COMPLETE. CHECK THE OUTPUT FOLDER.")

if __name__ == "__main__":
    run()