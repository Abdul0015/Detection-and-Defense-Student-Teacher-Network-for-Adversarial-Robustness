from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from model.predict import predict 

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result=None, image_path=None)

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", result=None, image_path=None)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)
            result = predict(filepath)
            print("RESULT FROM MODEL:", result)
            image_path = filepath.replace("\\", "/")

    return render_template("index.html", result=result, image_path=image_path)

@app.route('/proposed')
def proposed():
    return render_template('proposed.html')

if __name__ == '__main__':
    app.run(debug=True)