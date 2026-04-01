import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Cargar los datos del Excel (CSV)pip install pandas scikit-learn
archivo_datos = os.path.join('data', 'datos_senas.csv')
data = pd.read_csv(archivo_datos, header=None)

# 2. Dividir en características (puntos) y etiquetas (letras)
X = data.iloc[:, :-1] # Todos los números
y = data.iloc[:, -1]  # La letra del final

# 3. Separar datos para entrenar y otros para probar si aprendió bien
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# 4. Crear el modelo (el "cerebro")
modelo = RandomForestClassifier()

# 5. Entrenar
print("Entrenando al cerebro... por favor espera.")
modelo.fit(x_train, y_train)

# 6. Ver qué tan inteligente es
y_predict = modelo.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"¡Entrenamiento terminado! Precisión: {score * 100:.2f}%")

# 7. Guardar el cerebro en la carpeta 'models'
with open('models/modelo_senas.p', 'wb') as f:
    pickle.dump({'model': modelo}, f)

print("El modelo se guardó como 'models/modelo_senas.p'")