import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Función para realizar la predicción y mostrar el resultado en una ventana de mensaje
def perform_prediction():
    # Obtener los valores ingresados por el usuario
    user_input = []
    for entry in entries:
        value = entry.get()
        user_input.append(float(value))

    # Preprocesar los datos de entrada del usuario
    user_input = scaler.transform([user_input])

    # Realizar la predicción del tumor
    prediction = knn.predict(user_input)[0]

    # Mostrar el resultado de la predicción en una ventana de mensaje
    if prediction == 1:
        messagebox.showinfo("Resultado", "El tumor es maligno.")
    else:
        messagebox.showinfo("Resultado", "El tumor es benigno.")

# Crear la ventana principal
window = tk.Tk()
window.title("Sistema de Pre Diagnóstico del Cáncer de Mama")

# Estilos de la ventana
window.configure(bg="#F5F5F5")
window.geometry("400x300")

# Cargar el conjunto de datos desde el archivo CSV descargado
df = pd.read_csv('C:/Users/oem/Downloads/breast-cancer/data.csv')

# Eliminar la columna "id" ya que no es relevante para la predicción
df = df.drop('id', axis=1)

# Convertir la columna "diagnosis" a valores numéricos (Maligno: 1, Benigno: 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Eliminar la columna "Unnamed: 32" si está presente
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Dividir los datos en características (X) y etiquetas (y)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Realizar imputación de valores faltantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Escalar los datos para normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo KNN con los datos de entrenamiento
knn.fit(X_train, y_train)

# Obtener la lista de nombres de características
feature_names = df.columns[1:]

# Crear etiquetas y campos de entrada para las características
labels = []
entries = []
for i, feature in enumerate(feature_names):
    if i < len(feature_names) // 2:
        label = tk.Label(window, text=feature + ":", bg="#F5F5F5")
        entry = tk.Entry(window)
        label.grid(row=i, column=0, pady=5, padx=10, sticky="w")
        entry.grid(row=i, column=1, pady=5, padx=10, sticky="e")
    else:
        label = tk.Label(window, text=feature + ":", bg="#F5F5F5")
        entry = tk.Entry(window)
        label.grid(row=i - len(feature_names) // 2, column=2, pady=5, padx=10, sticky="w")
        entry.grid(row=i - len(feature_names) // 2, column=3, pady=5, padx=10, sticky="e")
    labels.append(label)
    entries.append(entry)

# Crear botón para realizar la predicción
predict_button = tk.Button(window, text="Realizar Predicción", command=perform_prediction, bg="#007BFF", fg="white")
predict_button.grid(row=len(feature_names)//2, column=0, columnspan=4, pady=10)

# Ejecutar el bucle principal de la ventana
window.mainloop()

