from distutils.log import debug
from flask import Flask, request, redirect, render_template, url_for
from knn.sequential import sequential, query_with_radius
from knn.rtree_index import rtree_index
from initialization import load_json, calculate_radius
import face_recognition

app = Flask(__name__)

def name_path_dict(paths):
    name_paths = dict()
    i = 1
    for path in paths:
        temp = {}
        sep = path.split("/")
        name = sep[1]
        temp["name"] = name
        temp["path"] = path
        name_paths[i] = temp
        i += 1
    return name_paths



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            k = int(request.form.get("TopK"))
            dataset = load_json()
            if str(request.form.get("metodo")) == "KNN-Secuencial-prioridad":
                return sec_prioridad(file, k, dataset)
            elif str(request.form.get("metodo")) == "KNN-Secuencial-rango":
                return redirect(url_for("sec_rango"))
            elif str(request.form.get("metodo")) == "KNN-RTree":
                return redirect(url_for("rtree"))
            elif str(request.form.get("metodo")) == "KNN-HighD":
                return redirect(url_for("highD"))

    return render_template("index.html")

def sec_prioridad(file, k, dataset):
    img = face_recognition.load_image_file(file)
    encoded_faces = face_recognition.face_encodings(img)
    paths = sequential(encoded_faces, k, dataset)
    name_paths = name_path_dict(paths)
    return render_template("results.html", path = file, paths=name_paths)


if __name__ == "__main__":
    app.run(port=5000, debug=True)