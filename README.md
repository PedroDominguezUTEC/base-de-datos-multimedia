# Proyecto N°2 - Base de datos II

## Integrantes

- Arróspide, Eduardo
- Domínguez, Pedro

| Lista de actividades Realizadas                                      | Responsable                     | Participación |
| -------------------------------------------------------------------- | ------------------------------- | ------------- |
| Sequential, Query_Radius, R_tree, Kd_tree, Faiss HNSW                | Pedro Domínguez  | 100%          |
| Frontend, Experimentación                                            | Eduardo Arróspide| 100%          |


## Introducción

El objetivo del proyecto es el aprendizaje de estructuras multidimensionales que permitan búsquedas y recuperaciones eficientes de imágenes en un servicio web de reconocimiento facial.

Este proyecto utiliza algoritmos KNN de búsqueda **secuencial con cola de prioridad** y **rango con radio de búsqueda.** Asimismo, para búsquedas más eficientes aprovecha estructuras multidimensionales como el **Rtree**, **KDtree** y **Hierarchical Navigable Small World(HNSW)**

## Datos

Las imagenes utilizadas provienen de la [página](http://vis-www.cs.umass.edu/lfw/). Las  consultas son  cualquier  imagen  que  incluya  un  rostro y el resultado son el top-k de los personajes más parecidos.

## Construcción del índice

Para evitar codificar la colección de imágenes en cada consulta se contruyó un índice *.json* en donde se guarda la información de los rostros de cada imagen como un vector característico de 128 atributos. 


## Búsqueda secuencial

### Búsqueda KNN

```python
def knn_sequential(faces_encoding, k, dataset):

    answer = []
    for path, matrix_vector_faces in dataset:
        for dis in face_recognition.face_distance(matrix_vector_faces, faces_encoding[0]):
            answer.append((path, dis))

    query_answer = sorted(answer, key = lambda x: x[1])

    return [path for path, distance in query_answer[:k]]
```

### Búsqueda por rango

```python
def radius_sequential(faces_encoding, r, dataset):
    answer = []
    
    for path, matrix_vector_faces in dataset:
        for dis in face_recognition.face_distance(matrix_vector_faces, faces_encoding[0]):
            if dis < r:
                answer.append((path, dis))

    query_answer = sorted(answer, key = lambda x: x[1])
    return [path for path, distance in query_answer]
```



1. **Pre-Processing:**
  El contenido de cada noticia debe ser pre-procesado, ignorando "stop words", realizar el stemming y normalizando el texto en general. Específicamente, nuestra aplicación realiza un proceso como el descrito a continuación:
    - Lowercase: todas las palabras a letras minúsculas
    - Stop words: remover todos los stop words. Utilizamos los stop words de inglés de la librería [NLTK](https://www.nltk.org/)
    - Puntuación: Remover todo símbolo de puntuación, puesto que no son necesarios para la creación del índice invertido.
    - Apostrofes: Remove apostrofes (muy usados en inglés) ya que, al igual que la puntuación, no son relevantes.
    - Stemming: Convertir todas las palabras a su raíz. Para esta acción utilizamos el [Porter Stemmer](https://www.nltk.org/howto/stem.html) de NLTK.
    - Num2Words: Utilizamos la librería [num2words](https://pypi.org/project/num2words/) para convertir cada número a su contraparte como palabra en inglés. Ej: 15 a "fifteen".
2. **Cálculo del TF:** 
  Para el cálculo de la frecuencia de términos utilizamos la libreria NLTK, ya que nos proporciona la funcionabilidad de manera directa utilizando "word_tokenize". Para la construcción de este índice buscamos un archivo de texto dentro de la carpeta de indices "indexes", si no es encontrada se crea un nuevo documento y se agrega el título de la noticia (que se está tomando como "documento" de forma teorica) junto a su TF.

## Ejecución óptima de consultas

La ejecución de la consulta sigue la siguiente secuencia de pasos:

1. Se le solicita al usuario un *query* y un número *k* arbitrario de respuestas que desea que se le muestre.
2. Al *query* se le formatea, tokeniza y se le calcula las frecuencias por cada término (*tf*)
3. Para obtener los pesos (*w*) de la query y de cada documento se utiliza el siguiente algoritmo: ![](img/cosine.png)
    
    - Acceder al par **(documento, tf)** y calcular los peso de cada término es sencillo gracias al índice invertido previamente construido
    -  Solo es necesario calcular la norma una sola vez al realizar una iteración a los vectores de pesos de cada documento.
4. Siguiendo el algoritmo se realiza las multiplicación de los pesos y su normalización.
5. Finalmente, se ordena el vector de *scoring* de manera descendente y se retorna los primeros *k* resultados solicitados por el cliente.


## PostgreSQL / MongoDB
Para ambos se utiliza el archivo utils.py, que contiene dos funciones: remove_punctuation y query_tokens, que ayudarán a formatear la query, filtrando stopwords, eliminando caracteres no deseados y tokenizando las palabras. 

**IMPORTANTE**
Antes de iniciar la configuración de las bases de datos debe crear una carpeta llamada `datasets` y **DEBE DESCARGAR** los datos a utilizar en esta. [Link del dataset](https://drive.google.com/drive/folders/1Ou_c_Lz7tsS4ptNCefre7EpP72eQ43sn?usp=sharing).

### PostgreSQL
Según lo requerido en el enunciado del proyecto se implementó una base de datos en PostgresSQL para realizar consultas con el **Índice Invertido GIN**.
Entonces, para poder utilizar la base de datos implementada.
En primer lugar, setear las credenciales para que nuestra aplicación se conecte a su base de datos local. 

Debemos modificar los datos en `backend/credentials.json`. 
```json
{
    "username": "",
    "password": "",
}
```
Solo basta con poner el usuario y contraseña con el que queramos que se loguee. 

Luego, debemos llenar crear y llenar la base de datos. Con el script `init-postgresql.py`.
Ejecutar,
```python
python3 backend/init-postgresql.py
```

Este script va crear la base de datos `allthenews`, va crear una tabla llamada `news`, agregará un campo con un índice llamado `content_ts`. Y llenará los datos de todas las noticias a la tabla. 

Con esto ya tendriamos nuestra base de datos Postgres lista! :)

### MongoDB
Para poder seguir comparando resultados, también se implementó la solución utilizando una base de datos NoSQL con MongoDB. MongoDB es un sistema de base de datos orientado a documentos que utiliza la estructura de datos **BSON**, similar a JSON. Para este proyecto, se descargó [MongoDB Community](https://www.mongodb.com/try/download/community), y se instaló junto a MongoDB Compass, una aplicación para visualizar de forma amigable las bases de datos y colecciones que se tienen en el local. El código está escrito considerando que, a través de MongoDB Compass, se establece una conexión localhost en el puerto 27017 (el predeterminado), se crea una base de datos llamada db2project y se crea la colección articles1. 

![Conexión](img/MongoDBCompass_Connection.png)
![DB](img/MongoDBCompass_DB.png)

Se utilizó la librería pymongo para utilizar MongoDB desde Python, y se instaló con 
```
pip install pymongo
```
Para poblar la base de datos se utiliza csv.DictReader() y se va agregando la data, fila por fila, en diccionarios.

Si se quiere hacer full text search en MongoDB se tiene que crear, manualmente, un index en los campos que se quieran cubrir. Para este proyecto, se creó un índice llamado "content_ts" que cubre los campos "title" y "content". Para ello, se utilizó la siguiente línea de código: 
```
db.articles1.create_index([('title', 'text'), ('content', 'text')], name="content_ts")
```
![Index](img/MongoDBCompass_Index.png)

Conforme se vayan corriendo los archivos en Python, se pueden ver los resultados en MongoDB Compass (dándole refresh). Una vez creado el índice, se utiliza el método .find() para la búsqueda, cuyos parámetros serán dos diccionarios. El primero es el que se necesita para hacer el text search, pero si se hace solo así los resultados no estarán ordenados por relevancia. Para ello, se debe agregar el segundo diccionario, "score" y luego hacer un sort con el mismo "score" de parámetro. 

```{python}
search = db.articles1.find(
    { "$text": { "$search": format_text } },
    { "score": { "$meta": "textScore" } }).limit(k)

search.sort([( "score", { "$meta": "textScore" } )])
```

## Frontend
El frontend de este proyecto se realizó con React.js

**Moverse a la carpeta `/frontend`**
### Setup
```terminal
npm install
```

### Run
```terminal
npm start
```

## Backend
El backend de este proyecto se realizó con Flask. 

**Moverse a la carpeta `/backend`**
### Setup
```terminal
pip3 install requeriments.txt
```

### Run
```terminal
python3 backend/main.py
```

## Video
[Link del video](https://drive.google.com/drive/folders/1_y8E0AEloaDDQBZHqMdQkBr-fyT8duqr). 