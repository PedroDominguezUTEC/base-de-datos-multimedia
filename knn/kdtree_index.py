from scipy.spatial import KDTree

def knn_kdtree(faces_encoding, k , dataset):

    data = []
    for path, matrix_vector_faces in dataset:
        data.append(matrix_vector_faces[0])
    
    tree = KDTree(data)
    distances, indexes = tree.query(faces_encoding, k)
    
    return [dataset[index][0] for index in indexes[0]]