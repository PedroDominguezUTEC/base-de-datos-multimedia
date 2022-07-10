import face_recognition
import random
import os

from knn.sequential import sequential
from knn.rtree_index import rtree_index
from initialization import load_json

#Initialize query
k = 5
image_path = "yo_con_8_cursos.jpg"

#Open query image and extract characteristic vector
query_image = face_recognition.load_image_file(image_path) 
faces_encoding = face_recognition.face_encodings(query_image)

dataset = load_json()
sequential(faces_encoding, k, dataset)
rtree_index(faces_encoding, k, dataset)


'''
    N = 5000
    vector_dist = []
    for i in range(N):
        obj_1 = random(data_row)
        obj_2 = random(data_row)
        dist = face_recognition.face_distance([obj_1], obj_2[0])

        vector_dist.append(dist)
'''

'''
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
'''
