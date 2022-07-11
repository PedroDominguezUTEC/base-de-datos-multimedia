from distutils.log import debug
from flask import Flask, request, redirect, render_template, url_for
from knn.sequential import knn_sequential, radius_sequential
from knn.rtree_index import knn_rtree
from knn.kdtree_index import knn_kdtree
from knn.faiss_index import knn_faiss
from initialization import load_json, calculate_radius
import face_recognition
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/"

def name_path_dict(paths):
    name_paths = dict()
    i = 1
    for path in paths:
        temp = {}
        sep = path.split("/")
        name = sep[1]
        temp["name"] = name
        temp["path"] = os.path.join(app.config["UPLOAD_FOLDER"], path)
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
            dataset = load_json()
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            if str(request.form.get("metodo")) == "KNN-Secuencial-prioridad":
                k = int(request.form.get("TopK"))
                return sec_prioridad(file, k, dataset, path)
            elif str(request.form.get("metodo")) == "KNN-Secuencial-rango":
                r = float(request.form.get("radio"))
                return sec_rango(file, r, dataset, path)
            elif str(request.form.get("metodo")) == "KNN-RTree":
                k = int(request.form.get("TopK"))
                return rtree(file, k, dataset, path)
            elif str(request.form.get("metodo")) == "KNN-KDTree":
                k = int(request.form.get("TopK"))
                return kdtree(file, k, dataset, path)
            elif str(request.form.get("metodo")) == "KNN-Faiss":
                k = int(request.form.get("TopK"))
                return faiss(file, k, dataset, path)

    return render_template("index.html")

def sec_prioridad(file, k, dataset, path):
    img = face_recognition.load_image_file(file)
    encoded_faces = face_recognition.face_encodings(img)
    paths = knn_sequential(encoded_faces, k, dataset)
    name_paths = name_path_dict(paths)
    return render_template("results.html", path = path, paths=name_paths)

def sec_rango(file, r, dataset, path):
    img = face_recognition.load_image_file(file)
    encoded_faces = face_recognition.face_encodings(img)
    paths = radius_sequential(encoded_faces, r, dataset)
    name_paths = name_path_dict(paths)
    return render_template("results.html", path = path, paths=name_paths)

def rtree(file, k, dataset, path):
    img = face_recognition.load_image_file(file)
    encoded_faces = face_recognition.face_encodings(img)
    paths = knn_rtree(encoded_faces, k, dataset)
    name_paths = name_path_dict(paths)
    return render_template("results.html", path = path, paths=name_paths)

def kdtree(file, k, dataset, path):
    img = face_recognition.load_image_file(file)
    encoded_faces = face_recognition.face_encodings(img)
    paths = knn_kdtree(encoded_faces, k, dataset)
    name_paths = name_path_dict(paths)
    return render_template("results.html", path = path, paths=name_paths)

def faiss(file, k, dataset, path):
    img = face_recognition.load_image_file(file)
    encoded_faces = face_recognition.face_encodings(img)
    paths = knn_faiss(encoded_faces, k, dataset)
    name_paths = name_path_dict(paths)
    return render_template("results.html", path = path, paths=name_paths)

if __name__ == "__main__":
    app.run(port=5000, debug=True)