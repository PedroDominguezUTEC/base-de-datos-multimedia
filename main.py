import json
import numpy

file = open("encoded_faces.json", "r")
decodedArrays = json.load(file)
image_path = "lfw/John_Edwards/John_Edwards_0003.jpg"

finalNumpyArray = numpy.asarray(decodedArrays[image_path])
print(finalNumpyArray)