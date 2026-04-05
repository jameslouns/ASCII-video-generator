from flask import Flask, request, jsonify, send_from_directory, render_template, Response
import subprocess
import sys
import os
import threading
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress
import cv2
import datetime
import re

app = Flask(__name__)
OUTPUT_DIR = "output"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

global current_process
current_process = None

def cleanup_file(filename):
    time.sleep(3600)  # 1 hour
    try:
        os.remove(os.path.join(OUTPUT_DIR, filename))
    except Exception as e:
        print(f"Cleanup error for {filename}: {e}")

def extract_video_id(url):
    patterns = [
        r'youtu\.be/([a-zA-Z0-9_-]+)',
        r'youtube.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'youtube.com/embed/([a-zA-Z0-9_-]+)',
        r'youtube.com/v/([a-zA-Z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/download', methods=['POST'])
def download_video():
    global current_process
    url = request.form.get('url')
    if not url or "youtube.com" not in url:
        return render_template('error.html', error="Invalid YouTube URL"), 400

    try:
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            video_id = "video_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Download video
        video = YouTube(url)
        video.streams.order_by('resolution').last().download(output_path='Videos', filename=video_id + '.mp4')

        # Download audio
        audio_stream = video.streams.get_lowest_resolution()
        audio_stream.download(output_path='Videos', filename=video_id + '_audio.webm')

        # Get metadata
        video_path = os.path.join('Videos', video_id + '.mp4')
        vid = cv2.VideoCapture(video_path)
        FrameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        FrameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        FrameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = vid.get(cv2.CAP_PROP_FPS)
        vid.release()

        # Render configure template
        return render_template('configure.html',
                               video_id=video_id,
                               resolution=f"{FrameWidth}x{FrameHeight}",
                               frame_count=FrameCount,
                               fps=FPS)
    except Exception as e:
        return render_template('error.html', error=f"Failed to download video: {str(e)}"), 500


@app.route('/process', methods=['POST'])
def process_video():
    global current_process
    video_id = request.form.get('video_id')
    textscale = int(request.form.get('textscale', 80))
    framesper = int(request.form.get('framesper', 50))

    video_path = os.path.join('Videos', video_id + '.mp4')

    main_old_path = os.path.join(os.path.dirname(__file__), 'converter.py')
    try:
        current_process = subprocess.Popen(
            [sys.executable, main_old_path, '--video_path', video_path,
             '--textscale', str(textscale), '--framesper', str(framesper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
            universal_newlines=True
        )
        return render_template('progress.html')
    except Exception as e:
        return render_template('error.html', error=f"Failed to start processing: {str(e)}"), 500


@app.route('/progress')
def progress():
    def generate():
        try:
            global current_process
            if current_process is None:
                yield "data: ERROR: No active process\n\n"
                return

            while True:
                # Read stdout
                stdout_line = current_process.stdout.readline()
                if stdout_line:
                    yield f"data: {stdout_line.strip()}\n\n"

                # Read stderr
                stderr_line = current_process.stderr.readline()
                if stderr_line and stderr_line.strip() !='' and not stderr_line.startswith('chunk') and not stderr_line.startswith('frame'):
                    print(f"Raw stderr: {repr(stderr_line)}", flush=True)
                    yield f"data: ERROR: {stderr_line.strip()}\n\n"

                # Check if process finished
                if current_process.poll() is not None:
                    # Read remaining stdout
                    while True:
                        line = current_process.stdout.readline()
                        if not line:
                            break
                        yield f"data: {line.strip()}\n\n"
                    # Read remaining stderr
                    while True:
                        line = current_process.stderr.readline()
                        if not line:
                            break
                        print(f"Raw stderr test2: {repr(line)}", flush=True)
                        yield f"data: ERROR: {line.strip()}\n\n"
                    break

                # Sleep to avoid tight loop
                time.sleep(0.05)

            # Check return code after process finished
            if current_process.returncode != 0:
                yield f"data: ERROR: Process failed with code {current_process.returncode}\n\n"
            current_process = None
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
            print(f"Exception in progress route: {e}", file=sys.stderr)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/download/<filename>', methods=['GET'])
def serve_file(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return render_template('error.html', error="File not found"), 404


@app.route('/download_success')
def download_success():
    filename = request.args.get('filename')
    if not filename:
        return render_template('error.html', error="No filename provided"), 400
    return render_template('download_success.html', filename=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)