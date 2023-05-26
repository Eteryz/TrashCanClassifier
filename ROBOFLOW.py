from roboflow import Roboflow

api_key = "U6OeEoeGPTUtbgEpxQlb"
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("trashcanclassification")
model_classifier = project.version(1).model

URL_IMAGE = ""
# Преобразование массива в массив байтов
response = model_classifier.predict(URL_IMAGE, hosted=True).json()

print(response)