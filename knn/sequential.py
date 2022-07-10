import face_recognition

def sequential(faces_encoding, k, dataset):

    answer = []
    for path, matrix_vector_faces in dataset:
        for dis in face_recognition.face_distance(matrix_vector_faces, faces_encoding[0]):
            answer.append((path, dis))

    query_answer = sorted(answer, key = lambda x: x[1])

    return [path for path, distance in query_answer[:k]]


def query_with_radius(faces_encoding, r, dataset):
    answer = []
    
    for path, matrix_vector_faces in dataset:
        for dis in face_recognition.face_distance(matrix_vector_faces, faces_encoding[0]):
            if dis < r:
                answer.append((path, dis))

    query_answer = sorted(answer, key = lambda x: x[1])
    return [path for path, distance in query_answer]