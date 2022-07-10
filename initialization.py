import face_recognition
import os
import json
import numpy
import random
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def initialize():
    '''
    file = open("encoded_faces.json", "r")
    decodedArrays = json.load(file)

    finalNumpyArray = numpy.asarray(decodedArrays["lfw/Larry_Tanenbaum/Larry_Tanenbaum_0001.jpg"])
    print(finalNumpyArray)
    '''


    with open("encoded_faces.json", "w") as json_file:
        dictionary = {}
        for root, subdirectories, files in os.walk("lfw/"):
            for file in files:
                path = f'{root}/{file}'
                
                image = face_recognition.load_image_file(path)
                faces_on_image = face_recognition.face_encodings(image)

                if len(faces_on_image) > 0:
                    dictionary[path] = faces_on_image
                    
        json.dump(dictionary, json_file, cls=NumpyArrayEncoder)


def load_json():
    #Open and load the json file
    file = open("encoded_faces.json", "r")
    decodedJson = json.load(file)

    #Dataset of each image and its vectors
    dataset = []
    for path, matrix_vector_faces in decodedJson.items():
        dataset.append((path, numpy.asarray(decodedJson[path])))
    
    return dataset


def calculate_radius(experiments, dataset):
    vector_dist = []
    for i in range(experiments):
        obj_1 = random.choice(dataset)[1]
        obj_2 = random.choice(dataset)[1]
        dist = face_recognition.face_distance(obj_1, obj_2[0])
        for dis in dist:
            vector_dist.append(dis)

    print(numpy.mean(vector_dist))