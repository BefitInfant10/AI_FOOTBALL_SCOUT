"""
Flask app for uploading and processing football videos.
Run with: python app.py
"""

import os
import json
import uuid
import threading
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from src.detect import detect_video

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

processing_status = {
    "state": "idle",
    "message": "",
    "stats": None,
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_processing(input_path, output_path):
    global processing_status
    try:
        _, stats = detect_video(input_path, output_path)

        processing_status["state"] = "done"
        processing_status["message"] = "Processing complete!"
        processing_status["stats"] = stats
    except Exception as e:
        processing_status["state"] = "error"
        processing_status["message"] = f"Processing failed: {e}"
        processing_status["stats"] = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    global processing_status

    if request.method == "GET":
        return render_template("index.html")

    if "video_file" not in request.files:
        return render_template("index.html", message="No file part in the request.")

    file = request.files["video_file"]

    if file.filename == "":
        return render_template("index.html", message="Please select a video file.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not filename or filename == ".":
            filename = f"upload_{uuid.uuid4().hex[:8]}.mp4"

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        output_path = os.path.join(app.config["OUTPUT_FOLDER"], "processed_video.mp4")

        if processing_status["state"] == "processing":
            return render_template("index.html", message="A video is already being processed. Please wait.")

        processing_status["state"] = "processing"
        processing_status["message"] = "Analyzing video... This may take a few minutes."
        processing_status["stats"] = None

        thread = threading.Thread(target=run_processing, args=(save_path, output_path), daemon=True)
        thread.start()

        return redirect(url_for("status"))

    return render_template("index.html", message="Invalid file type. Please upload a video file.")


@app.route("/status")
def status():
    global processing_status
    return render_template(
        "status.html",
        state=processing_status["state"],
        message=processing_status["message"],
        stats=processing_status["stats"],
    )


@app.route("/api/status")
def api_status():
    global processing_status
    return {
        "state": processing_status["state"],
        "message": processing_status["message"],
        "stats": processing_status["stats"],
    }


@app.route("/download")
def download():
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], "processed_video.mp4")
    if os.path.exists(output_path) and processing_status["state"] != "processing":
        return send_file(output_path, as_attachment=True, download_name="processed_video.mp4")
    return redirect(url_for("index"))


@app.route("/video")
def video():
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], "processed_video.mp4")
    if os.path.exists(output_path) and processing_status["state"] != "processing":
        return send_file(output_path, mimetype="video/mp4")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
