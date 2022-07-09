from pandas import NamedAgg
from rtree import index
import os

name = 'highD_index'
data_file = name + '.data'
index_file = name + '.index'

if not (os.path.exists(data_file) or os.path.exists(index_file)):
    p = index.Property()
    p.dimension = 4 #D
    p.buffering_capacity = 4 #M
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    idx = index.Index(name, properties=p)

#insertar puntos
idx.insert(1, (44, 45, 35, 41))
idx.insert(0, (15, 13, 19, 18))
idx.insert(2, (3, 4, 3, 2))

#retornar elementos de la interseccion con el rectangulo 
q = (2, 3, 4, 7)
lres = list(idx.nearest(coordinates=q, num_results=2))
print("El vecino mas cercano de (2, 3, 4, 7): ", lres)



