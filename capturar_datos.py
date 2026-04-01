import cv2
import mediapipe as mp
import csv
import os

# Acceso directo a las soluciones
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not os.path.exists('data'):
    os.makedirs('data')

print("Cámara iniciada. Presiona una letra para guardar o ESC para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Guardar si se presiona una tecla
            if ord('a') <= key <= ord('z'):
                label = chr(key)
                # Extraemos los 21 puntos (x, y, z)
                coords = [item for lm in hand_landmarks.landmark for item in [lm.x, lm.y, lm.z]]
                coords.append(label)
                
                with open('data/datos_senas.csv', 'a', newline='') as f:
                    csv.writer(f).writerow(coords)
                print(f"Guardado: {label}")

    cv2.imshow('Kawsay - Captura', frame)

cap.release()
