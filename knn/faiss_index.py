import faiss
import numpy as np
def knn_faiss(faces_encoding, k , dataset):
    d = 128
    M = 32
    index = faiss.IndexHNSWFlat(d, M)

    data = []
    for path, matrix_vector_faces in dataset:
        data.append(matrix_vector_faces[0])
    data = np.float32(np.array(data))
    
    index.add(data)

    xq = np.float32(np.array([faces_encoding[0]]))
    
    D, I = index.search(xq, k)

    for idx in I[0]:
        print(dataset[idx][0])