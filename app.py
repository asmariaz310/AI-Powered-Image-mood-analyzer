from flask import Flask, render_template, request, redirect, url_for
import os
from emotion_detection import process_video_input  # Your detection function
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Handle detection based on input source."""
    input_source = request.form.get('input_source')
    try:
        if input_source == 'webcam':
            process_video_input(0)  # Webcam
        else:
            video_file = request.form.get('video_path')
            if os.path.exists(video_file):
                process_video_input(video_file)  # Video file
            else:
                return render_template('index.html', error="Video file not found.")
        return render_template('index.html', message="Detection completed.")
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)