from scipy.spatial import KDTree

def knn_kdtree(faces_encoding, k , dataset):

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
        
    tree = KDTree(data)
    distances, indexes = tree.query(faces_encoding, k)
    
    return [dataset[face_index_to_dataset_index[index]][0] for index in indexes[0]]