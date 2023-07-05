import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

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

# Crear un clasificador KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo KNN con todos los datos
knn.fit(X, y)

# Obtener la lista de nombres de características
feature_names = df.columns[1:]

# Solicitar al usuario que ingrese las características del tumor
user_input = []
for feature in feature_names:
    value = input(f"Ingrese el valor de {feature}: ")
    user_input.append(float(value))

# Preprocesar los datos de entrada del usuario
user_input = scaler.transform([user_input])

# Realizar la predicción del tumor
prediction = knn.predict(user_input)[0]

# Mostrar el resultado de la predicción
if prediction == 1:
    print("El tumor es maligno.")
else:
    print("El tumor es benigno.")
