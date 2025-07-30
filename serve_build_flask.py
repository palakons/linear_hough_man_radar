from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)

PUBLIC_DIR = (
    "/ist-nas/users/palakonk/singularity/home/palakons/linear_hough_man_radar/public"
)
SAMPLES_DIR = "/ist-nas/users/palakonk/singularity_data/palakons/new_dataset/MAN/mini/man-truckscenes/samples"


@app.route("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/samples/<path:filename>")
def serve_samples(filename):
    file_path = os.path.join(SAMPLES_DIR, filename)
    if os.path.isfile(file_path):
        return send_from_directory(SAMPLES_DIR, filename)
    abort(404)


@app.route("/<path:filename>")
def serve_public(filename):
    file_path = os.path.join(PUBLIC_DIR, filename)
    if os.path.isfile(file_path):
        return send_from_directory(PUBLIC_DIR, filename)
    abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
