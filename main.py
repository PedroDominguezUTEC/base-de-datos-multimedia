import face_recognition

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