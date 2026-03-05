import cv2
import whisper
import subprocess
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
MODEL_PATH = 'blaze_face_short_range.tflite'
INPUT_VIDEO = "podcast.mp4"
OUTPUT_DIR = "output"
CLIP_DURATION = 60 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_video_duration(file_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def get_separated_faces(video_path, start_s):
    """Finds two distinct speakers by enforcing a minimum distance between them."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    success, frame = cap.read()
    
    # Default: Host on left (30%), Guest on right (70%)
    left_x, right_x = 0.30, 0.70
    
    if success:
        h, w, _ = frame.shape
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options)
        with vision.FaceDetector.create_from_options(options) as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_image)
            
            if res.detections and len(res.detections) >= 2:
                # Get all face centers
                centers = sorted([(d.bounding_box.origin_x + d.bounding_box.width/2)/w for d in res.detections])
                
                # Take the two furthest apart to ensure we get different people
                left_x = centers[0]
                right_x = centers[-1]
                
                # Fallback: If they are too close (likely same person), force separation
                if abs(right_x - left_x) < 0.25:
                    left_x = max(0.15, left_x - 0.2)
                    right_x = min(0.85, right_x + 0.2)

    cap.release()
    return left_x, right_x

def create_shorts():
    print("Step 1: Transcribing and finding hooks...")
    model_whisper = whisper.load_model("tiny")
    result = model_whisper.transcribe(INPUT_VIDEO, fp16=False)
    total_len = get_video_duration(INPUT_VIDEO)
    
    hooks = ["actually", "crazy", "imagine", "lesson", "secret", "wow", "because", "mistake"]
    potential_starts = []
    min_gap = 120
    last_time = -min_gap

    for segment in result['segments']:
        if any(hook in segment['text'].lower() for hook in hooks):
            if segment['start'] > last_time + min_gap and segment['start'] < (total_len - CLIP_DURATION):
                potential_starts.append(segment['start'])
                last_time = segment['start']

    clips = potential_starts[:7]
    print(f"Step 2: Found {len(clips)} viral moments. Rendering...")

    for i, start in enumerate(clips):
        lx, rx = get_separated_faces(INPUT_VIDEO, start)
        output_file = os.path.join(OUTPUT_DIR, f"viral_stack_{i+1}.mp4")
        
        # MATH: To get a 1080x1920 final video from two stacked boxes:
        # Each box must be 1080x960.
        # Aspect Ratio of 1080/960 = 1.125 (or 9:8).
        # We crop the original video at a width of (OriginalHeight * 1.125).
        
        crop_w = "ih*(9/8)" 
        
        filter_complex = (
            f"[0:v]crop={crop_w}:ih:iw*{lx}-({crop_w}/2):0,scale=1080:960,setsar=1[top]; "
            f"[0:v]crop={crop_w}:ih:iw*{rx}-({crop_w}/2):0,scale=1080:960,setsar=1[bottom]; "
            f"[top][bottom]vstack=inputs=2"
        )

        cmd = [
            FFMPEG_PATH, '-y', '-ss', str(start), '-t', str(CLIP_DURATION),
            '-i', INPUT_VIDEO,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22', '-c:a', 'aac',
            output_file
        ]
        
        print(f"Rendering Clip {i+1}/7...")
        subprocess.run(cmd)

if __name__ == "__main__":
    if os.path.exists(INPUT_VIDEO):
        create_shorts()
        print("Done! Check your output folder.")
    else:
        print(f"File {INPUT_VIDEO} not found.")