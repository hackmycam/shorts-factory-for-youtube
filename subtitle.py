import whisper
import subprocess
import os
import uuid
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- CONFIG ---
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
OUTPUT_FOLDER = r"C:\Users\praveen R\Documents\ShortsFactory\final_video"
WHISPER_MODEL = "tiny" 
MAX_WORKERS = 4 # Number of videos to process at the same time

def format_timestamp(seconds):
    td = datetime.fromtimestamp(seconds) - datetime.fromtimestamp(0)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

def process_single_video(input_path):
    """Handles the subtitle and rendering for one chosen video"""
    if not os.path.exists(input_path):
        return None

    # Generate unique output names
    unique_id = uuid.uuid4().hex[:6]
    timestamp = datetime.now().strftime("%H%M%S")
    video_filename = os.path.basename(input_path)
    output_name = f"Viral_{timestamp}_{unique_id}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    srt_path = f"temp_{unique_id}.srt"

    print(f"🎬 Processing: {video_filename}")

    try:
        # 1. Transcribe with Whisper
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(input_path, fp16=False, word_timestamps=True)
        
        # 2. Create SRT File
        with open(srt_path, "w", encoding="utf-8") as f:
            counter = 1
            for segment in result['segments']:
                words = segment['words']
                for i in range(0, len(words), 2): 
                    chunk = words[i:i+2]
                    if not chunk: continue
                    start_t = format_timestamp(chunk[0]['start'])
                    end_t = format_timestamp(chunk[-1]['end'])
                    text = " ".join([w['word'].strip().upper() for w in chunk])
                    f.write(f"{counter}\n{start_t} --> {end_t}\n{text}\n\n")
                    counter += 1

        # 3. Build FFmpeg command with Bottom-Center alignment
        style = (
            "FontName=Arial Black,FontSize=24,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=2.5,Shadow=1.5,"
            "Alignment=2,MarginV=60"
        )
        srt_path_ffmpeg = srt_path.replace("\\", "/").replace(":", "\\:")
        
        cmd = [
            FFMPEG_PATH, "-y", "-i", input_path,
            "-vf", f"subtitles='{srt_path_ffmpeg}':force_style='{style}'",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            output_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup
        if os.path.exists(srt_path):
            os.remove(srt_path)
            
        print(f"✅ Created: {output_name}")
        return output_path

    except Exception as e:
        print(f"❌ Error with {video_filename}: {e}")
        return None

def select_files():
    """Opens a window to let you pick videos"""
    root = tk.Tk()
    root.withdraw() # Hide the main tiny tkinter window
    root.attributes("-topmost", True) # Bring the dialog to the front
    
    file_paths = filedialog.askopenfilenames(
        title="Select videos for Viral Subtitles",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
    )
    
    root.destroy()
    return list(file_paths)

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Step 1: User selects files
    selected_videos = select_files()

    if not selected_videos:
        print("📂 No videos selected. Exiting.")
    else:
        print(f"🚀 Selected {len(selected_videos)} videos. Processing...")
        
        # Step 2: Run in parallel (3-4 at once)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(process_single_video, selected_videos)

    print("\n✨ All selected videos have been processed!")