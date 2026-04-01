import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import os
import time

# 1. Configuración de Voz
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Velocidad de la voz

def hablar(texto):
    engine.say(texto)
    engine.runAndWait()

# 2. Cargar el "cerebro" (Modelo entrenado)
# Asegúrate de que el archivo existe en esta ruta
model_path = './models/modelo_senas.p'
if os.path.exists(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
else:
    print(f"Error: No se encontró el archivo {model_path}. Entrena el modelo primero.")
    exit()

# 3. Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Variables de control
palabra_completa = ""
ultima_letra_detectada = ""
tiempo_inicio = 0
segundos_para_confirmar = 2 # Tiempo que debes mantener la seña

print("Traductor Kawsay iniciado.")
print("Comandos: [ESPACIO] Borrar palabra | [ESC] Salir")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    # Convertir a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos de la mano (opcional, ayuda visual)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extraer coordenadas
            coords = [item for lm in hand_landmarks.landmark for item in [lm.x, lm.y, lm.z]]
            prediccion = model.predict([np.asarray(coords)])
            letra_actual = prediccion[0].upper()

            # LÓGICA DE CONSTRUCCIÓN:
            if letra_actual == ultima_letra_detectada:
                tiempo_transcurrido = time.time() - tiempo_inicio
                
                # Barra de carga visual (color cian)
                ancho_barra = int(min(tiempo_transcurrido / segundos_para_confirmar, 1) * 200)
                cv2.rectangle(frame, (50, 100), (50 + ancho_barra, 120), (255, 255, 0), -1)
                cv2.rectangle(frame, (50, 100), (250, 120), (255, 255, 255), 2)
                
                if tiempo_transcurrido >= segundos_para_confirmar:
                    palabra_completa += letra_actual
                    hablar(letra_actual)
                    ultima_letra_detectada = "" # Reset
                    tiempo_inicio = time.time()
            else:
                ultima_letra_detectada = letra_actual
                tiempo_inicio = time.time()

            cv2.putText(frame, f'Letra detectada: {letra_actual}', (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # MOSTRAR LA PALABRA QUE SE VA FORMANDO
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f'Palabra: {palabra_completa}', (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # UNA SOLA VENTANA
    cv2.imshow('Kawsay - Traductor Inteligente', frame)
    
    # MANEJO DE TECLAS (Consolidado)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC para salir
        break
    elif key == ord(' '): # ESPACIO para borrar
        palabra_completa = ""

cap.release()
cv2.destroyAllWindows()