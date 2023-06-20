import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.metrics import classification_report 


# Directorio de datos en Google Drive
data_dir = '/Users/shuaishen/Desktop/lfw_20_exactas_5'

# Preprocesamiento de datos
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2  # Porcentaje para validación (ajusta según tus necesidades)
)

batch_size = 5
img_height = 224
img_width = 224

# Generadores de datos
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print(class_names)

# Construcción del modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilación y entrenamiento del modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 6  # Ajusta el número de épocas según tus necesidades

model.fit(train_generator, epochs=num_epochs, validation_data=val_generator)

# Guardar el modelo entrenado en Google Drive
model.save('/Users/shuaishen/Desktop/Reconocimiento de patrones/trained_model_local.h5')

# Evaluación del modelo
test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print("Accuracy on test data:", accuracy)
predictions = model.predict(test_generator)

# Convertir las predicciones en etiquetas numéricas
predicted_labels = np.argmax(predictions, axis=1)

# Obtener las etiquetas reales
true_labels = test_generator.classes

# Obtener el informe de métricas de clasificación
report = classification_report(true_labels, predicted_labels, target_names=class_names)

# Imprimir el informe de métricas de clasificación
print(report)


