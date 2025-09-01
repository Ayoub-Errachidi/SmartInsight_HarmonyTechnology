# views/data_statistiques.py
import traceback
import pandas as pd
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
import logging

logger = logging.getLogger(__name__)

# Matplotlib backend "headless" pour serveur
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .views_importation import load_latest_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -------------------- Calcul des statistiques --------------------
def calculer_statistiques(data: pd.DataFrame) -> dict:
    """Calcule toutes les statistiques d'un dataset et retourne un dictionnaire des résultats."""
    result = {
        "stats_base": None,
        "stats_base_dict": {},
        "stats_adv": None,
        "stats_adv_dict": {},
        "stats_text": None,
        "stats_text_dict": {},
        "top_values_dict": {},
        "missing_values": None,
        "outliers": None,
        "pca_html": None,
        "date_cols": []
    }

    if data.empty:
        return result

    # -------------------- Détection colonnes numériques / textuelles / dates --------------------
    numeric_cols = []
    text_cols = []
    date_cols = []

    for col in data.columns:
        series = data[col]

        # Si déjà datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue

        # Si numérique
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        # Tester si c'est une date (>= 80% valeurs convertibles en datetime)
        try:
            converted = pd.to_datetime(series, errors="coerce", utc=True)
            valid_dates = converted.notna().sum()
            if valid_dates / len(series) >= 0.8:
                date_cols.append(col)
            else:
                text_cols.append(col)
        except Exception:
            text_cols.append(col)

    result["date_cols"] = date_cols

    # -------------------- Données numériques --------------------
    if numeric_cols:
        numeric_data = data[numeric_cols]

        stats_base_df = numeric_data.describe()
        result["stats_base"] = stats_base_df.to_html(classes="table table-striped")
        result["stats_base_dict"] = stats_base_df.T.to_dict(orient="index")

        stats_adv_df = pd.DataFrame({
            "Médiane": numeric_data.median(),
            "Variance": numeric_data.var(),
            "Écart-type": numeric_data.std(),
            "Coef de variation": numeric_data.std() / numeric_data.mean(),
            "Asymétrie": numeric_data.skew(),
            "Aplatissement": numeric_data.kurt()
        })
        result["stats_adv"] = stats_adv_df.to_html(classes="table table-striped")
        result["stats_adv_dict"] = stats_adv_df.to_dict(orient="index")

        # ==== Valeurs aberrantes ====
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers_detected = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
        result["outliers"] = outliers_detected.to_frame(
            name="Nombre de valeurs aberrantes"
        ).to_html(classes="table table-striped")

        # Vérifier s’il existe au moins un outlier > 0
        result["has_outliers"] = (outliers_detected > 0).any()

        # PCA
        if numeric_data.shape[1] > 1:
            try:
                numeric_for_pca = numeric_data.dropna()
                if numeric_for_pca.shape[0] >= 2:
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(numeric_for_pca)
                    pca = PCA()
                    pca_result = pca.fit(data_scaled)

                    pca_df = pd.DataFrame({
                        "Composante": [f"PC{i+1}" for i in range(len(pca.explained_variance_))],
                        "Valeur propre": pca.explained_variance_,
                        "Variance expliquée (%)": pca.explained_variance_ratio_ * 100,
                        "Variance cumulée (%)": pca.explained_variance_ratio_.cumsum() * 100
                    })
                    result["pca_html"] = pca_df.to_html(classes="table table-striped", index=False)
                else:
                    result["pca_html"] = "<p class='text-warning'>PCA non calculable (trop peu de lignes après suppression des NaN).</p>"
            except Exception as e:
                logger.warning(f"Erreur PCA : {e}")
                result["pca_html"] = "<p class='text-warning'>Erreur lors du calcul de la PCA.</p>"

    # -------------------- Données textuelles --------------------
    if text_cols:
        text_data = data[text_cols]
        unique_values_dict = {}
        for col in text_data.columns:
            uniques = sorted(set(str(val) for val in text_data[col].dropna().unique()))
            unique_values_dict[col] = {"count": len(uniques), "values": uniques}

        unique_values_df = pd.DataFrame({
            col: {
                "Nb valeurs uniques": unique_values_dict[col]["count"],
                "Valeurs uniques": ", ".join(unique_values_dict[col]["values"][:20])
            } for col in unique_values_dict
        }).T

        result["stats_text"] = {
            "unique_values": unique_values_df.to_html(classes="table table-striped", escape=False),
            "top_values": text_data.describe().T.to_html(classes="table table-striped"),
            "avg_word_count": text_data.apply(
                lambda x: x.dropna().astype(str).apply(lambda y: len(y.split())).mean()
            ).to_frame(name="Nombre moyen de mots").to_html(classes="table table-striped")
        }

        result["stats_text_dict"] = {
            col: {"nb_uniques": unique_values_dict[col]["count"], "values": unique_values_dict[col]["values"][:5]}
            for col in unique_values_dict
        }
        result["top_values_dict"] = text_data.describe().T.to_dict(orient="index")

    # -------------------- Valeurs manquantes --------------------
    null_counts = data.isnull().sum()
    result["missing_values"] = null_counts.to_frame(
        name="Valeurs manquantes"
    ).to_html(classes="table table-striped")

    return result


# -------------------- Vue Django --------------------
def statistique_view(request: HttpRequest) -> HttpResponse:
    try:
        data, _latest_entry = load_latest_dataset()
        if data.empty:
            return render(request, "pages/statistiques.html", {"error": "Le dataset importé est vide."})

        stats = calculer_statistiques(data)

        return render(request, "pages/statistiques.html", stats)

    except Exception as e:
        logger.error("Erreur lors du calcul des statistiques : %s", traceback.format_exc())
        return render(request, "pages/statistiques.html", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })