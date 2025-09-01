import re
import pandas as pd
import numpy as np
import uuid
import logging
from django.shortcuts import render
import traceback

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.decomposition import PCA

from django.db import transaction
from django.core.exceptions import ValidationError
from sklearn.preprocessing import LabelEncoder, StandardScaler
from django.http import HttpRequest, HttpResponse

from .data_cleaning import preprocess_dataset_for_ml
from ..models import DataNettoyer, DataTransform
from .views_importation import load_latest_dataset, recreate_table_from_df_copy

logger = logging.getLogger(__name__)


# ====================================================
# Détection du type d’analyse ML
# ====================================================

def detect_ml_type(df: pd.DataFrame, target_column: str | None = None) -> str:
    '''
    Détecte automatiquement le type d'analyse ML :
    - classification
    - regression
    - clustering (si target absente)
    '''
    # Si aucune colonne cible → clustering non supervisé
    if not target_column or target_column not in df.columns:
        return "clustering"

    y = df[target_column].dropna()
    n_samples = len(y)
    if n_samples == 0:
        return "clustering"

    n_unique = y.nunique(dropna=False)

    # 1. Si type catégoriel ou string → classification
    if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
        return "classification"
    
    # 2. Si valeurs booléennes ou seulement 2 classes
    if pd.api.types.is_bool_dtype(y) or n_unique == 2:
        return "classification"

    # 3. Si données numériques
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = n_unique / n_samples

        # Heuristique : peu de classes distinctes = classification
        if n_unique < 15 and unique_ratio < 0.1:
            return "classification"
        else:
            return "regression"

    # 4. Par défaut
    return "classification"

# ====================================================
# Feature Engineering sur les colonnes date
# ====================================================
def feature_engineering_dates(df: pd.DataFrame,
                              min_nonnull_fraction: float = 0.6,
                              drop_original: bool = False,
                              parse_int_yyyymmdd: bool = True) -> pd.DataFrame:
    """
    Détecte automatiquement les colonnes de type date et génère uniquement :
    - <col>_year1
    - <col>_month1
    - <col>_day1
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        ser = df_copy[col]

        date_ser = None

        # Vérifier si colonne datetime ou convertible
        if pd.api.types.is_datetime64_any_dtype(ser):
            date_ser = ser
        elif pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            coerced = pd.to_datetime(ser, errors='coerce', infer_datetime_format=True)
            if coerced.notna().mean() < min_nonnull_fraction:
                continue
            date_ser = coerced
            df_copy[col] = date_ser
        elif parse_int_yyyymmdd and pd.api.types.is_integer_dtype(ser):
            coerced = pd.to_datetime(ser.astype(str), format='%Y%m%d', errors='coerce')
            if coerced.notna().mean() < min_nonnull_fraction:
                continue
            date_ser = coerced
            df_copy[col] = date_ser
        else:
            continue

        # Extraire year, month, day seulement
        df_copy[f"{col}_year1"] = date_ser.dt.year.astype("Int64")
        df_copy[f"{col}_month1"] = date_ser.dt.month.astype("Int64")
        df_copy[f"{col}_day1"] = date_ser.dt.day.astype("Int64")

        if drop_original:
            df_copy.drop(columns=[col], inplace=True)

        logger.info(f"Feature engineering appliqué sur colonne date : {col}")

    return df_copy

# ====================================================
# Prétraitement global et sauvegarde table nettoyée
# ====================================================

def preprocess_and_save(df, latest_entry):
    """
    Applique le prétraitement global et met à jour la table nettoyée (_nettoye)
    """
    # Prétraitement global
    df_cleaned, methode_outliers, stats_outliers = preprocess_dataset_for_ml(df, return_info=True)

    # Nom de la table nettoyée
    nettoyee_name = f"{latest_entry.table_name}_nettoye"

    # Sauvegarde table nettoyée
    recreate_table_from_df_copy(df_cleaned, nettoyee_name)

    # Mise à jour ou création des métadonnées
    existing = DataNettoyer.objects.filter(table_name_nettoyee=nettoyee_name).first()
    if not existing:
        DataNettoyer.objects.create(table_name_nettoyee=nettoyee_name, table_originale=latest_entry)
    else:
        existing.table_originale = latest_entry
        existing.save()

    return df_cleaned


# ====================================================
# Transformation ML du dataset
# ====================================================

def transform_dataset(df, target_column=None, request=None, latest_entry=None):
    """
    Applique :
    - Prétraitement global + sauvegarde table _nettoye
    - Label Encoding sur colonnes catégorielles
    - Standardisation sur colonnes numériques (hors target)
    """

    # Si latest_entry non fourni, le récupérer automatiquement
    if latest_entry is None:
        df_from_db, latest_entry = load_latest_dataset()

    if latest_entry is None:
        raise ValueError("Impossible de récupérer latest_entry pour sauvegarder la table nettoyée.")

    # Prétraitement global + mise à jour table nettoyée
    df_cleaned = preprocess_and_save(df, latest_entry)

    # Feature engineering dates
    df_cleaned = feature_engineering_dates(df_cleaned)

    df_transformed = df_cleaned.copy()
    mappings = {}
    scalers = {}

    # Transformation ML des colonnes
    for col in df_transformed.columns:
        if col == target_column:
            print(f"Ignoré (target) : {col}")
            continue

        if df_transformed[col].dtype == 'object' or df_transformed[col].dtype.name == 'category':
            try:
                le = LabelEncoder()
                df_transformed[col] = df_transformed[col].astype(str)
                df_transformed[col] = le.fit_transform(df_transformed[col])
                mappings[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
                print(f"Encodé : {col} → {mappings[col]}")
            except Exception as e:
                print(f"Erreur LabelEncoder sur '{col}': {e}")

        elif pd.api.types.is_numeric_dtype(df_transformed[col]):
            try:
                scaler = StandardScaler()
                df_transformed[col] = scaler.fit_transform(df_transformed[[col]])
                scalers[col] = scaler # Pour prediction
            except Exception as e:
                print(f"Erreur StandardScaler sur {col}: {e}")
    
    # Sauvegarde des mappings
    if request:
        request.session['label_mappings'] = mappings
        request.session['scalers'] = {col: scaler.mean_.tolist() + scaler.scale_.tolist()
                                      for col, scaler in scalers.items()}

    return df_transformed

# ====================================================
# Calcul PCA pour tableau
# ====================================================
def calcul_pca(df_transformed, n_components=None):
    """
    Calcule les composantes principales pour le dataset transformé.
    Retourne une liste de tuples :
    (PC_name, eigenvalue, variance_expliquee_pct, variance_cumulee_pct)
    """
    # Sélection des colonnes numériques uniquement
    numeric_cols = df_transformed.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return []

    df_numeric = df_transformed[numeric_cols].fillna(0)  # Remplacer NaN par 0 si nécessaire

    pca = PCA(n_components=n_components)
    pca.fit(df_numeric)

    eigenvalues = pca.explained_variance_
    variance_pct = pca.explained_variance_ratio_ * 100
    cum_variance_pct = np.cumsum(variance_pct)

    pca_table = []
    for i, (eig, var, cum_var) in enumerate(zip(eigenvalues, variance_pct, cum_variance_pct), start=1):
        pca_table.append((f"PC{i}", eig, var, cum_var))
    
    return pca_table



# ====================================================
# Sauvegarde dataset transformé
# ====================================================
@transaction.atomic
def save_transformed_dataset_to_db(df: pd.DataFrame, nettoyee_entry: DataNettoyer) -> DataTransform:
    """
    Sauvegarde un DataFrame transformé dans une nouvelle table PostgreSQL
    liée à une entrée DataNettoyer existante.
    """
    base_name = re.sub(r'^dataset', '', nettoyee_entry.table_name_nettoyee.lower())
    base_name = re.sub(r'\W+', '_', base_name)
    table_name = f"dataset_transforme_{uuid.uuid4().hex[:2]}{base_name}"

    recreate_table_from_df_copy(df, table_name)

    transform_entry = DataTransform.objects.create(
        table_name_transformee=table_name,
        table_nettoyee=nettoyee_entry
    )

    logger.info(f"Dataset transformé sauvegardé : {table_name}")
    return transform_entry

# ====================================================
# Chargement du dernier dataset transformé
# ====================================================
def load_latest_transformed_dataset() -> tuple[pd.DataFrame, DataTransform]:
    """
    Récupère le dernier dataset transformé depuis PostgreSQL.
    """
    latest_entry = DataTransform.objects.select_related('table_nettoyee').order_by('-uploaded_at').first()
    if not latest_entry:
        raise ValidationError("Aucun dataset transformé trouvé en base.")

    table_name = latest_entry.table_name_transformee
    query = f'SELECT * FROM "{table_name}"'

    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    return df, latest_entry

# ====================================================
# Vue : choix de la colonne cible (ML)
# ====================================================

def choix_cible_view(request):
    context = {}

    try:
        # Charger le dernier dataset original ou nettoyé
        df, latest_entry = load_latest_dataset()
        colonnes = df.columns.tolist()
        context["colonnes"] = colonnes

        # Récupérer la dernière transformation pour préremplir la cible
        latest_transform = DataTransform.objects.select_related("table_nettoyee").order_by('-uploaded_at').first()
        previous_target = latest_transform.target_column if latest_transform else None
        context["previous_target"] = previous_target

        if request.method == "POST":
            selected_target = request.POST.get("selected_target", "").strip()

            # Mode clustering si aucune cible sélectionnée
            if not selected_target or selected_target.lower() == "aucune":
                ml_type = detect_ml_type(df, None)
                df_transformed = transform_dataset(df, None, request=request, latest_entry=latest_entry)

                # Sauvegarder transformation avec target = None
                nettoyee_entry = DataNettoyer.objects.filter(table_originale=latest_entry).first()
                if nettoyee_entry:
                    transform_entry = save_transformed_dataset_to_db(df_transformed, nettoyee_entry)
                    transform_entry.target_column = None
                    transform_entry.save(update_fields=["target_column"])

                context.update({
                    "selected_target": None,
                    "features_preview": colonnes,
                    "target_preview": [],
                    "ml_type": ml_type,
                    "success": "Aucune colonne cible sélectionnée. Mode clustering activé."
                })

            else:
                if selected_target not in colonnes:
                    raise ValidationError(f"La colonne '{selected_target}' n'existe pas dans le dataset.")

                ml_type = detect_ml_type(df, selected_target)
                df_transformed = transform_dataset(df, selected_target, request=request, latest_entry=latest_entry)

                # Sauvegarder transformation avec target sélectionnée
                nettoyee_entry = DataNettoyer.objects.filter(table_originale=latest_entry).first()
                if nettoyee_entry:
                    transform_entry = save_transformed_dataset_to_db(df_transformed, nettoyee_entry)
                    transform_entry.target_column = selected_target
                    transform_entry.save(update_fields=["target_column"])

                context.update({
                    "selected_target": selected_target,
                    "features_preview": [col for col in colonnes if col != selected_target],
                    "target_preview": [selected_target],
                    "ml_type": ml_type,
                    "success": f"Colonne cible '{selected_target}' sélectionnée. Mode {ml_type} détecté."
                })

        return render(request, "pages/machine_learning/choix_cible.html", context)

    except ValidationError as ve:
        context["error"] = str(ve)
    except Exception as e:
        import traceback
        context["error"] = f"Erreur interne : {e}"
        context["traceback"] = traceback.format_exc()

    return render(request, "pages/machine_learning/choix_cible.html", context)



# ====================================================
# FONCTION DE MATRICE DE CORRÉLATION
# ====================================================
def generer_correlation_image(df: pd.DataFrame, target_column: str | None = None) -> str:
    """
    Calcule la matrice de corrélation avec la target encodée si nécessaire.
    Retourne une image encodée en base64.
    Ignore les colonnes constantes (toutes les valeurs = 0).
    """
    try:
        df_corr = df.copy()

        # Encodage temporaire de la target si elle est catégorielle
        if target_column:
            target_series = df_corr[target_column]
            if target_series.dtype == "object" or pd.api.types.is_categorical_dtype(target_series):
                try:
                    le = LabelEncoder()
                    df_corr[target_column] = le.fit_transform(target_series.astype(str))
                except Exception as e:
                    print(f"Erreur encodage temporaire de la target pour la corrélation : {e}")
                    df_corr.drop(columns=[target_column], inplace=True)

        # Suppression des colonnes contenant uniquement des 0
        df_corr = df_corr.loc[:, ~(df_corr.eq(0).all())]

        # Calcul de la corrélation
        correlation_matrix = df_corr.corr(numeric_only=True)

        # Génération de la figure
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title("Matrice de Corrélation")

        # Sauvegarde en buffer et encodage base64
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        print(f"Erreur génération image de corrélation : {e}")
        return None


# ====================================================
# Vue : résumé des colonnes transformées
# ====================================================
def detect_column_transformation(df: pd.DataFrame, target_column: str | None) -> dict:
    transformations = {}
    for col in df.columns:
        if col == target_column:
            transformations[col] = "Cible (non transformée)"
        elif pd.api.types.is_numeric_dtype(df[col]):
            transformations[col] = "Standardisation"

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Colonnes générées exactement comme dans feature_engineering_dates
            created_cols = [f"{col}_year1", f"{col}_month1", f"{col}_day1"]
            transformations[col] = created_cols

        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            transformations[col] = "Label Encoding"
        else:
            transformations[col] = "Aucune / Non transformée"
    return transformations


def transformer_colonnes_view(request: HttpRequest) -> HttpResponse:
    context = {}

    try:
        # Récupérer le dernier dataset transformé depuis PostgreSQL
        df_transformed, latest_transform = load_latest_transformed_dataset()
        target_column = latest_transform.target_column  # colonne cible enregistrée dans la base

        context["nb_lignes"] = df_transformed.shape[0]
        context["nb_colonnes"] = df_transformed.shape[1]
        context["colonnes"] = df_transformed.columns.tolist()
        context["target_column"] = target_column

        # Détection des transformations appliquées par colonne
        transformations = detect_column_transformation(df_transformed, target_column)
        context["transformations"] = transformations

        # Génération de la matrice de corrélation
        correlation_image = generer_correlation_image(df_transformed, target_column)
        context["correlation_image"] = correlation_image

        # --- Feature Engineering : informations détaillées ---
        date_features = [col for col in df_transformed.columns if any(col.endswith(suffix) for suffix in ["_year1", "_month1", "_day1"])]
        date_feature_info = []
        for col in date_features:
            info = {
                "colonne": col,
                "nb_null": int(df_transformed[col].isna().sum()),
                "min": float(df_transformed[col].min()) if pd.api.types.is_numeric_dtype(df_transformed[col]) else None,
                "max": float(df_transformed[col].max()) if pd.api.types.is_numeric_dtype(df_transformed[col]) else None,
                "mean": float(df_transformed[col].mean()) if pd.api.types.is_numeric_dtype(df_transformed[col]) else None,
            }
            date_feature_info.append(info)

        context["date_feature_info"] = date_feature_info

        # --- PCA ---
        pca_table = df_transformed.drop(columns=[target_column]) if target_column else df_transformed
        context["pca_table"] = calcul_pca(pca_table)

        return render(request, "pages/machine_learning/colonnes_transformees.html", context)

    except ValidationError as ve:
        context["error"] = str(ve)
    except Exception as e:
        context["error"] = f"Erreur lors du chargement des données transformées : {e}"
        context["traceback"] = traceback.format_exc()

    return render(request, "pages/machine_learning/colonnes_transformees.html", context)

# ====================================================
# Télécharger le dernier dataset transformé en Excel
# ====================================================
def telecharger_dataset_transforme(request):
    """
    Permet de télécharger le dernier dataset transformé au format Excel (.xlsx).
    """
    try:
        # Récupération du dataset transformé et de ses métadonnées
        df, latest_entry = load_latest_transformed_dataset()
    except ValidationError as e:
        return HttpResponse(f"Erreur : {str(e)}", status=404)

    # Nom du fichier Excel basé sur le nom de la table
    filename = f"{latest_entry.table_name_transformee}.xlsx"

    # Création d'une réponse HTTP avec Excel
    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    # Sauvegarde du DataFrame directement dans la réponse HTTP
    with pd.ExcelWriter(response, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Dataset_Transforme", index=False)

    return response


# ----------------------------------------------------
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64


def plot_clusters(X, labels, title):
    """Génère une visualisation PCA 2D des clusters avec couleurs fixes et retourne une image base64."""
    try:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Palette de couleurs (tu peux l'étendre si besoin)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(6, 5))

        for label in unique_labels:
            cluster_points = X_reduced[labels == label]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[color_map[label]],
                label=f"Cluster {label}",
                s=40,
                alpha=0.7
            )

        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return img_base64
    except Exception:
        return None



def entrainer_modeles():
    context = {}

    try:
        # Charger le dataset et son entrée en base
        df_transformed, latest_transform = load_latest_transformed_dataset()
        target_column = latest_transform.target_column

        # Détection du type ML
        ml_type = detect_ml_type(df_transformed, target_column)

        if df_transformed is None or ml_type is None:
            raise ValueError("Données transformées ou type ML non disponibles en base.")

        if target_column:
            X = df_transformed.drop(columns=[target_column])
            y = df_transformed[target_column]
        else:
            X = df_transformed
            y = None

        # Supprimer les colonnes non numériques
        X = X.select_dtypes(include=[np.number])

        # --- Si régression, encoder / standardiser y
        if ml_type == "regression" and y is not None:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Standardisation de y
            y_array = y.values.reshape(-1, 1)
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_array).flatten()
            y = pd.Series(y_scaled, index=df_transformed.index)

        # --- Split Train/Test uniquement pour supervised
        if ml_type in ["classification", "regression"]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train = X_test = X
            y_train = y_test = None

        results_supervised = []

        # ==============================
        # Classification
        # ==============================
        if ml_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForestClassifier": RandomForestClassifier(random_state=42),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "SVC": SVC(),
                "LGBMClassifier": LGBMClassifier()
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    results_supervised.append({
                        "model": name,
                        "score_name": "Accuracy / Precision / Recall / F1",
                        "score_value": f"Acc: {acc:.3f}, Préc: {prec:.3f}, Rappel: {rec:.3f}, F1: {f1:.3f}"
                    })
                except Exception as e:
                    results_supervised.append({
                        "model": name,
                        "score_name": "Erreur",
                        "score_value": str(e)
                    })

        # ==============================
        # Régression
        # ==============================
        elif ml_type == "regression":
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
                "SVR": SVR(),
                "LGBMRegressor": LGBMRegressor()
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    results_supervised.append({
                        "model": name,
                        "score_name": "MAE / MSE / RMSE / R²",
                        "score_value": f"MAE: {mae:.1f}, MSE: {mse:.1f}, RMSE: {rmse:.1f}, R²: {r2:.3f}"
                    })
                except Exception as e:
                    results_supervised.append({
                        "model": name,
                        "score_name": "Erreur",
                        "score_value": str(e)
                    })

        else:
            context["supervised_info"] = "Pas de modèle supervisé entraîné (mode clustering)"

        context["ml_type"] = ml_type
        context["results_supervised"] = results_supervised

        # ==============================
        # Clustering (Unsupervised)
        # ==============================
        results_unsupervised = []
        cluster_images = {}

        clustering_models = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }

        for name, model in clustering_models.items():
            try:
                labels = model.fit_predict(X)

                # Nombre de clusters valides (ignorer -1 pour DBSCAN)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                # Vérifier qu’il y a au moins 2 clusters valides
                if n_clusters >= 2:
                    silhouette = silhouette_score(X, labels)
                    dbi = davies_bouldin_score(X, labels)
                    ch = calinski_harabasz_score(X, labels)

                    # Générer visualisation
                    img_base64 = plot_clusters(X, labels, f"{name} Clusters (n={n_clusters})")
                    if img_base64:
                        cluster_images[name] = img_base64

                    results_unsupervised.append({
                        "model": name,
                        "n_clusters": n_clusters,
                        "score_name": "Silhouette / DBI / CH",
                        "score_value": f"Sil: {silhouette:.3f}, DBI: {dbi:.3f}, CH: {ch:.1f}"
                    })
                else:
                    results_unsupervised.append({
                        "model": name,
                        "n_clusters": n_clusters,
                        "score_name": "Clustering non valide",
                        "score_value": "Moins de 2 clusters détectés"
                    })
            except Exception as e:
                results_unsupervised.append({
                    "model": name,
                    "n_clusters": 0,
                    "score_name": "Erreur",
                    "score_value": str(e)
                })

        # Dictionnaire pour template : nombre de clusters par modèle
        cluster_counts = {result["model"]: result.get("n_clusters", "?") for result in results_unsupervised}

        context["results_unsupervised"] = results_unsupervised
        context["cluster_images"] = cluster_images
        context["cluster_counts"] = cluster_counts

        return context

    except Exception as e:
        context["error"] = f"Erreur entraînement modèle : {e}"
        return context

def afficher_resultats_entrainement_view(request: HttpRequest) -> HttpResponse:
    context = entrainer_modeles()
    return render(request, "pages/machine_learning/resultats_entrainement.html", context)

#-----------------------------------------------

# --- Fonction de prédiction générique ---
def faire_prediction(nouvelles_donnees: pd.DataFrame,
                     modele_nom: str,
                     ml_type: str,
                     X_train: pd.DataFrame = None,
                     y_train: pd.Series = None):
    """
    Réentraîne (si nécessaire) le modèle sur X_train / y_train puis prédit sur nouvelles_donnees.
    Retourne un dict: {"predictions": ..., "probabilities": ... (optionnel)}
    """
    # Définitions des modèles disponibles
    MODELS_CLASSIFICATION = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "SVC": SVC(probability=True),
        "LGBMClassifier": LGBMClassifier()
    }

    MODELS_REGRESSION = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
        "LGBMRegressor": LGBMRegressor()
    }

    MODELS_CLUSTERING = {
        "KMeans": KMeans(n_clusters=3, random_state=42)
    }

    model = None
    if ml_type == "classification":
        model = MODELS_CLASSIFICATION.get(modele_nom)
    elif ml_type == "regression":
        model = MODELS_REGRESSION.get(modele_nom)
    elif ml_type == "clustering":
        # pour le clustering on force KMeans (par défaut)
        model = MODELS_CLUSTERING.get(modele_nom, MODELS_CLUSTERING["KMeans"])
    else:
        raise ValueError(f"Type ML inconnu : {ml_type}")

    if model is None:
        raise ValueError(f"Modèle demandé '{modele_nom}' non supporté pour le type '{ml_type}'.")

    # Vérifications
    if ml_type in ["classification", "regression"]:
        if X_train is None or y_train is None:
            raise ValueError("X_train et y_train sont requis pour entraîner un modèle supervisé.")
        # Entraînement
        model.fit(X_train, y_train)
        preds = model.predict(nouvelles_donnees)
        result = {"predictions": preds}
        # si probas disponibles
        if hasattr(model, "predict_proba"):
            try:
                result["probabilities"] = model.predict_proba(nouvelles_donnees)
            except Exception:
                result["probabilities"] = None
        return result

    # clustering
    else:
        if X_train is None:
            raise ValueError("X_train est requis pour entraîner le modèle de clustering.")
        model.fit(X_train)
        # KMeans a predict
        if hasattr(model, "predict"):
            preds = model.predict(nouvelles_donnees)
            return {"predictions": preds}
        else:
            raise ValueError(f"Le modèle de clustering choisi ({modele_nom}) ne supporte pas predict().")


# --- Vue Django pour la page de prédiction ---
def prediction_view(request: HttpRequest) -> HttpResponse:
    context = {}
    try:
        # Charger dataset original
        df_original, _ = load_latest_dataset()

        # Récupérer le dernier dataset transformé
        df_transformed, latest_transform = load_latest_transformed_dataset()
        target_column = latest_transform.target_column
        ml_type = detect_ml_type(df_transformed, target_column)

        # Construire X (features numériques uniquement) et y si disponible
        if target_column:
            X = df_transformed.drop(columns=[target_column])
            y = df_transformed[target_column]
        else:
            X = df_transformed
            y = None

        # garder uniquement les colonnes numériques (tes modèles attendent des valeurs numériques)
        X = X.select_dtypes(include=[np.number])

        # Models disponibles selon le type ML
        if ml_type == "classification":
            available_models = ["LogisticRegression", "RandomForestClassifier",
                                "DecisionTreeClassifier", "SVC", "LGBMClassifier"]
        elif ml_type == "regression":
            available_models = ["LinearRegression", "RandomForestRegressor",
                                "DecisionTreeRegressor", "SVR", "LGBMRegressor"]
        else:
            available_models = ["KMeans"]  # par défaut KMeans pour clustering

        # Préparer valeurs uniques par colonne
        unique_values = {}
        for col in df_original.drop(columns=[target_column], errors="ignore").columns:
            uniques = df_original[col].dropna().unique().tolist()[:50]
            unique_values[col] = uniques

        context.update({
            "available_models": available_models,
            "colonnes_features": X.columns.tolist(),
            "ml_type": ml_type,
            "unique_values": unique_values
        })

        if request.method == "POST":
            modele_nom = request.POST.get("modele_nom")
            if not modele_nom:
                raise ValueError("Aucun modèle sélectionné.")

            # Récupération des valeurs saisies par l'utilisateur selon l'ordre des colonnes X
            new_row = {}
            mappings = request.session.get("label_mappings", {})
            scalers = request.session.get("scalers", {})

            for col in X.columns:
                val_select = request.POST.get(f"{col}_select", "").strip()
                val_input = request.POST.get(f"{col}_input", "").strip()

                # Priorité à l'input si rempli, sinon select
                raw = val_input if val_input else val_select
                if raw == "":
                    raise ValueError(f"Valeur manquante pour '{col}'.")

                # Ensuite transformation 
                # Cas catégoriel
                if col in mappings:
                    if raw not in mappings[col]:
                        raise ValueError(f"Valeur '{raw}' non reconnue pour {col}.")
                    val = mappings[col][raw]

                # Cas numérique
                elif col in scalers:
                    try:
                        val = float(raw.replace(",", "."))
                        mean, scale = scalers[col][:len(scalers[col])//2], scalers[col][len(scalers[col])//2:]
                        # si tu stockes juste mean_ et scale_ en float, alors :
                        mean_val, scale_val = scalers[col][0], scalers[col][1]
                        val = (val - mean_val) / scale_val
                    except Exception as e:
                        raise ValueError(f"Impossible de convertir '{raw}' en nombre pour {col}: {e}")

                else:
                    # fallback direct float
                    val = float(raw.replace(",", "."))

                new_row[col] = val

            nouvelles_donnees = pd.DataFrame([new_row], columns=X.columns)

            # Appel à la fonction de prédiction
            res = faire_prediction(
                nouvelles_donnees=nouvelles_donnees,
                modele_nom=modele_nom,
                ml_type=ml_type,
                X_train=X,
                y_train=y
            )

            # Préparer le contexte pour le template
            preds = res.get("predictions")
            if hasattr(preds, "tolist"):
                preds = preds.tolist()
            context["predictions"] = preds
            if "probabilities" in res and res["probabilities"] is not None:
                probs = res["probabilities"]
                if hasattr(probs, "tolist"):
                    probs = probs.tolist()
                context["probabilities"] = probs
            context["selected_model"] = modele_nom

        return render(request, "pages/machine_learning/prediction.html", context)

    except Exception as e:
        context["error"] = str(e)
        context["traceback"] = traceback.format_exc()
        return render(request, "pages/machine_learning/prediction.html", context)
