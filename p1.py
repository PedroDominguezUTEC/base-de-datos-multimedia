import face_recognition

query_image = face_recognition.load_image_file("lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
encoded_query = face_recognition.face_encodings(query_image)[0]


dataset_image = face_recognition.load_image_file("lfw/Zhang_Wenkang/Zhang_Wenkang_0002.jpg")
encoded_dataset_image = face_recognition.face_encodings(dataset_image)[0]

distance = face_recognition.face_distance([encoded_query], encoded_dataset_image)

print(distance)