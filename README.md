# 🖐️ Kawsay: Traductor de Lenguaje de Señas con IA

**Kawsay** es una solución tecnológica desarrollada para facilitar la comunicación con personas que presentan discapacidades cognitivas o auditivas. Utiliza **Visión Artificial** y **Machine Learning** para interpretar gestos en tiempo real.

---

## 🚀 Funcionalidades
* **Detección de Landmarks:** Rastreo de 21 puntos de la mano mediante MediaPipe.
* **Modelo Inteligente:** Clasificador Random Forest con alta precisión.
* **Salida de Voz:** Conversión de señas a audio en tiempo real con `pyttsx3`.
* **Interfaz Interactiva:** Barra de carga para confirmación de letras y limpieza de texto con `ESPACIO`.

---

## 🛠️ Tecnologías
* **Lenguaje:** Python
* **IA:** Scikit-Learn, NumPy, Pandas
* **Visión:** OpenCV, MediaPipe

---

## 📂 Estructura
```text
Kawsay/
 ┣ 📂 data          # Dataset de señas (CSV)
 ┣ 📂 models        # Modelo entrenado (.p)
 ┣ 📜 entrenar.py   # Script de entrenamiento
 ┣ 📜 traductor_final.py # Aplicación principal
 ┗ 📜 requirements.txt # Librerías necesarias
