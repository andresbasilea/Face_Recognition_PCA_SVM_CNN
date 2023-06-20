import cv2
import face_recognition 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import time
from face_deepface import BestRecognition
from face_clasification_PCA import modelo_2, normalize_data
#pip install face_recognition

#Importar ek modelo de clasificación
# Ruta donde se encuentra el modelo entrenado

#df = BestRecognition("/Users/shuaishen/Desktop/Test_Dataset_PCA/Shuai_Shen/IMG_1315.jpg", "/Users/shuaishen/Desktop/lfw_20_exactas_5")


def prueba(pca, clf, image):
    image = np.array(image).flatten()
    image.resize((64,64))
    image = normalize_data(image) 
    image_transformed = pca.transform(image.reshape(1,-1))
    prediction = clf.predict(image_transformed)
    return prediction[0]

def clasificacion_1(frame):
    model_path = '/Users/shuaishen/Desktop/Reconocimiento de patrones/trained_model_local.h5'
    model = load_model(model_path)
    img_height = 224
    img_width = 224

    # Preprocesamiento del frame
    image_array = cv2.resize(frame, (img_width, img_height))
    image_array = image_array.reshape((1, img_height, img_width, 3))
    image_array = image_array.astype('float32')
    image_array /= 255.0

    # Realizar la predicción
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    class_names = ['Andres_Basile', 'Angelina_Jolie', 'Rodolfo_Keller', 'Shuai_Shen', 'Vicente_Fox']
    predicted_class = class_names[predicted_class_index]

    return predicted_class


def capture_frames_from_webcam():
    # Crear el objeto VideoCapture para acceder a la cámara
    cap = cv2.VideoCapture(0)

    # Configurar la resolución y velocidad de fotogramas deseadas
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    start_time = time.time()
    interval = 1  # Intervalo en segundos
    text_duration = 1  # Duración del texto en segundos

    #pca, clf = modelo_2()

    while True:
        # Leer el frame de la cámara
        ret, frame = cap.read()

        # Convertir el frame a RGB (face_recognition utiliza RGB)
        rgb_frame = frame[:, :, ::-1]

        # Verificar si ha pasado el intervalo de tiempo
        elapsed_time = time.time() - start_time
        if elapsed_time >= interval:
            # Detectar las caras en el frame
            face_locations = face_recognition.face_locations(rgb_frame)

            # Si se detecta al menos una cara, realizar la clasificación
            if len(face_locations) > 0:
                # Seleccionar el primer cuadro para clasificar
                top, right, bottom, left = face_locations[0]

                # Dibujar un cuadro alrededor de la cara detectada
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Realizar la clasificación
                text = BestRecognition(frame, '/Users/shuaishen/Desktop/lfw_10_train')
                #text = clasificacion_1(frame)

                #text = prueba(pca, clf, frame)
                # Mostrar el texto debajo del cuadro
                cv2.putText(frame, text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Mostrar el frame con el texto durante la duración deseada
                start_text_time = time.time()
                while (time.time() - start_text_time) < text_duration:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Reiniciar el temporizador
            start_time = time.time()

        # Mostrar el frame en una ventana
        cv2.imshow("Frame", frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()


capture_frames_from_webcam()
