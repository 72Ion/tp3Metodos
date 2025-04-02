# TP03 - Métodos Numéricos y Optimización (MNyO) - Primer Semestre 2024

Este trabajo práctico se centra en el análisis y aplicación de la **descomposición en valores singulares (SVD)** y su uso en tareas de compresión de datos, reducción de la dimensionalidad y regresión lineal.

📄 [Descargar informe TP03 en PDF](MNyO_TP03.pdf.zip)

---

## 🖼️ PUNTO 1: Compresión de Imágenes

Se implementa una técnica de compresión basada en **SVD** sobre un conjunto de imágenes representadas como vectores de dimensión \(p \times p\).

### Objetivos:

- Aplicar **SVD** para aprender una representación de baja dimensión.
- Visualizar las imágenes reconstruidas con distintas cantidades de componentes principales.
- Analizar el error de compresión (norma de Frobenius) en función de la dimensión \(d\).
- Determinar el valor de \(d\) que garantiza un error menor al 10%.

---

## 📉 PUNTO 2: Reducción de Dimensionalidad y Cuadrados Mínimos

Se trabaja con un dataset de muestras de sensores (archivo `dataset.csv`) y una variable respuesta (`y.txt`) para realizar reducción de dimensionalidad y predicción lineal.

### Subpuntos:

#### 🔻 2.1 Reducción de dimensionalidad

- Se aplica **SVD** para reducir el dataset original \(X\) a un nuevo espacio \(Z = V_d^\top X\).
- Se analizan las similitudes par-a-par entre muestras en espacios de distintas dimensiones (d = 2, 6, 10, p) usando **PCA**.
- Se visualizan las matrices de similaridad y se discute la elección óptima de \(d\) según la estructura de los valores singulares.

#### 📐 2.2 Regresión por Cuadrados Mínimos

- Se entrena un modelo de regresión \( \hat{y} = X \hat{\beta} \) resolviendo el problema de mínimos cuadrados en el espacio original.
- Se analizan los pesos asignados a cada dimensión original en el vector \( \hat{\beta} \).

#### 🔁 2.3 Regresión sobre espacio reducido

- Se entrena un modelo en el espacio reducido \(Z\) con \(d=2\).
- Se compara el error de predicción con el modelo original y se identifican las muestras mejor ajustadas.

---

## ✅ Requisitos

- Python 3.x
- numpy
- matplotlib
- pandas
- scikit-learn
- Jupyter Notebook

---


