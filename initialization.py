import face_recognition
import os
import json
import numpy
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



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