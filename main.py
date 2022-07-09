import json
import numpy
import face_recognition

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

query_answer = sorted(answer, key = lambda x: x[1])
print(query_answer[:5])


'''
    N = 5000
    vector_dist = []
    for i in range(N):
        obj_1 = random(data_row)
        obj_2 = random(data_row)
        dist = face_recognition.face_distance([obj_1], obj_2[0])

        vector_dist.append(dist)
'''