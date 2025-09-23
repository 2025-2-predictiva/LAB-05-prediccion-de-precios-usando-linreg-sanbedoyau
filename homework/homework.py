#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta 'files/input/'.
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como 'files/models/model.pkl.gz'.
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
import os
import json
import gzip
import pickle
from glob import glob

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def get_datasets():
    train_df = pd.read_csv(
        './files/input/train_data.csv.zip',
        compression='zip',
        index_col=False,
    )
    test_df = pd.read_csv(
        './files/input/test_data.csv.zip',
        compression='zip',
        index_col=False,
    )
    return train_df, test_df


def preprocess(df):
    df_processed = df.copy()
    base_year = 2021
    df_processed['Age'] = base_year - df_processed['Year']
    df_processed = df_processed.drop(columns=['Year', 'Car_Name'])
    return df_processed


def separate_features_target(df):
    X = df.drop(columns=['Present_Price'])
    y = df['Present_Price']
    return X, y


def build_pipeline(X):
    cat_vars = ['Fuel_Type', 'Selling_type', 'Transmission']
    num_vars = [col for col in X.columns if col not in cat_vars]

    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_vars),
            ('num', MinMaxScaler(), num_vars),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ('preprocessor', transformer),
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('classifier', LinearRegression()),
        ]
    )
    return model_pipeline


def get_grid_search(estimator):
    parameters = {
        'feature_selection__k': range(1, 12),
        'classifier__fit_intercept': [True, False],
        'classifier__positive': [True, False],
    }
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    return grid


def ensure_directory(path):
    if os.path.exists(path):
        for file in glob(os.path.join(path, '*')):
            os.remove(file)
        os.rmdir(path)
    os.makedirs(path)


def export_model(model, filepath):
    model_dir = os.path.dirname(filepath)
    ensure_directory(model_dir)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(model, f)


def compute_metrics(y_true, y_pred, dataset_label):
    metrics = {
        'type': 'metrics',
        'dataset': dataset_label,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred)),
    }
    return metrics


def main():
    train_df, test_df = get_datasets()
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    X_train, y_train = separate_features_target(train_df)
    X_test, y_test = separate_features_target(test_df)

    pipe = build_pipeline(X_train)
    grid_search = get_grid_search(pipe)

    grid_search.fit(X_train, y_train)

    export_model(grid_search, os.path.join('files', 'models', 'model.pkl.gz'))

    pred_train = grid_search.predict(X_train)
    pred_test = grid_search.predict(X_test)

    metrics_train = compute_metrics(y_train, pred_train, 'train')
    metrics_test = compute_metrics(y_test, pred_test, 'test')

    output_dir = os.path.join('files', 'output')
    os.makedirs(output_dir, exist_ok=True)
    metrics_filepath = os.path.join(output_dir, 'metrics.json')
    with open(metrics_filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')


if __name__ == '__main__':
    main()