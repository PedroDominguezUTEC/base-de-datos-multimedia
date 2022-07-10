import json
import numpy
import face_recognition
import random
import os

#Open and load the json file
file = open("encoded_faces.json", "r")
decodedJson = json.load(file)

#Open query image and extract characteristic vector
image_path = "yo_con_8_cursos.jpg"
query_image = face_recognition.load_image_file(image_path) 
faces_encoding = face_recognition.face_encodings(query_image)

#Dataset of each image and its vectors
dataset = []
for path, matrix_vector_faces in decodedJson.items():
    dataset.append((path, numpy.asarray(decodedJson[path])))

answer = []
for path, matrix_vector_faces in dataset:
    for dis in face_recognition.face_distance(matrix_vector_faces, faces_encoding[0]):
        answer.append((path, dis))

query_answer = sorted(answer, key = lambda x: x[1], reverse=True)
print(query_answer[:5])
dists = []
for path, dis in query_answer:
    dists.append(dis)
print(numpy.mean(dists))


data_row = dict()
for file in os.listdir("./lfw/"):
    subdir = os.path.join("./lfw/", file)
    for pic in os.listdir(subdir):
        path = os.path.join(subdir, pic)
        data_row[pic] = face_recognition.load_image_file(path)
        
N = 5000
vector_dist = []
for i in range(N):
    obj_1 = random.choice(list(data_row.values()))
    obj_2 = random.choice(list(data_row.values()))
    dist = face_recognition.face_distance(obj_1, obj_2[0])
    for dis in dist:
        vector_dist.append(dis)

print(numpy.mean(vector_dist))