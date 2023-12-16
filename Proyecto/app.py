import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from joblib import load
import plotly.express as px

df_melted = load("df_melted.df")
df_semanal = load("df_semanal.df")
predicciones = load("predicciones.df")
df_metricas = load("df_metricas.df")
prediccion_2024 = load("prediccion_2024.df")


st.title("Análisis de la variación de los precios")

df_media = df_melted.groupby(['Alimento', 'Año']).mean().reset_index()
fig = px.bar(df_media, x="Alimento", y="Precio", color="Año", facet_col="Año", barmode="group")


# Agregar título y etiquetas
fig.update_layout(title="Variación Precio Alimentos", xaxis_title="Alimento", yaxis_title="Precio")
fig.update_xaxes(tickmode='linear', tick0=0.5, dtick=1)

# Mostrar gráfico
st.plotly_chart(fig)













# Heatmap de correlación

st.title("Correlación de precios")

plt.figure(figsize=(15,10))
sns.heatmap(df_semanal.corr().round(2), annot=True)
heatmap_fig = plt.gcf()
st.pyplot(heatmap_fig)











# Predicciones de los modelos

st.title("Predicción de los modelos")

plt.figure(figsize=(8, 6))
sns.kdeplot(predicciones, shade=True, color="skyblue")

# Configurar el diseño del gráfico
plt.title("Distribución del precio de la lubina")
plt.xlabel("Valores")
plt.ylabel("Densidad")

# Guardar el gráfico en una variable
kde_fig = plt.gcf()

# Mostrar el gráfico en Streamlit
st.pyplot(kde_fig)










# Métricas de los modelos
st.title("Métrica modelos")

fig, ax = plt.subplots(1, 2, figsize= (10, 5))
ax= ax.flatten()
sns.barplot(data=df_metricas, x="modelo", y="mse", ax= ax[0])
sns.barplot(data=df_metricas, x="modelo", y="r2", ax= ax[1])
ax[0].set_title("MSE")
ax[1].set_title("R2")

# Configurar el diseño de la figura
plt.tight_layout()

# Mostrar la figura en Streamlit
st.pyplot(fig)




df_preprocesado = df_semanal.copy()

columns = ['Semana', 'Año', 'Lenguado', 'Merluza', 'Dorada', 'Pollo', 'Cerdo', 'Cebolla', 'Patata', 'Plátano', 'Manzana', 'Naranja',
           'Precio_combustibles', 'Electricidad MWh', 'Temperatura_Barcelona', 'Precipitaciones_Barcelona']

standard = StandardScaler()
df_preprocesado[columns] = standard.fit_transform(df_preprocesado[columns])

# Dividir el conjunto de datos en entrenamiento y prueba
X = df_preprocesado.drop("Lubina", axis=1)
y = df_preprocesado["Lubina"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=20)

# Inicializar y entrenar el modelo de Random Forest
modelo_rf = KNeighborsRegressor()
modelo_rf.fit(X_train, y_train)



st.title("Predicción precio de la lubina")


predicciones_2024 = load('prediccion_2024.df')
df_semanal = load('df_semanal.df')


#Creamos las variables que influiran en el precio
st.sidebar.header('Ingresa los valores para la semana 1 del año 2024:')
datos_2024 = {}
for col in columns:
    datos_2024[col] = st.sidebar.number_input(col, value=0.0)

# Crear el DataFrame para la semana 1 del año 2024
X_2024 = pd.DataFrame([datos_2024])

# Aplicar la transformación de StandardScaler a las características de 2024
X_2024[columns] = standard.transform(X_2024[columns].values)

# Realizar la predicción con el modelo de Random Forest
prediccion_2024 = modelo_rf.predict(X_2024)

# Mostrar la predicción
st.write(f'Predicción del precio de la lubina para la semana y año elegido: {prediccion_2024[0]}')