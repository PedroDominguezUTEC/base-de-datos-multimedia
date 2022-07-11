import faiss
import numpy as np


def knn_faiss(faces_encoding, k , dataset):

    index = faiss.IndexHNSWFlat(128, 32)

    data = []

    dataset_index = 0
    faces_index = 0
    face_index_to_dataset_index = dict()
    
    for path, matrix_vector_faces in dataset:

        for face in matrix_vector_faces:
            face_index_to_dataset_index[faces_index] = dataset_index
            data.append(face)
            faces_index+=1

        dataset_index += 1

    data = np.float32(np.array(data))

    index.add(data)

    xq = np.float32(np.array([xq]))
    
    D, I = index.search(xq, k)
    
    return [dataset[face_index_to_dataset_index[idx]][0] for idx in I[0]]