import pandas as pd
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.core.exceptions import ValidationError
from .views_importation import load_latest_dataset, logger  # PostgreSQL version


# -------------------- Vue Informations Dataset --------------------
def information_view(request: HttpRequest) -> HttpResponse:
    try:
        # Récupération du dernier dataset depuis PostgreSQL
        data, latest_entry = load_latest_dataset()

        numeric_cols = []
        text_cols = []
        date_cols = []

        # --- Détection des colonnes date ---
        for col in data.columns:
            series = data[col]

            # Si déjà datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols.append(col)
                continue

            # Si numérique pur
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                continue

            # Tester si c'est une date
            try:
                converted = pd.to_datetime(series, errors='coerce', utc=True)
                valid_dates = converted.notna().sum()
                if valid_dates / len(series) >= 0.8:
                    date_cols.append(col)
                else:
                    text_cols.append(col)
            except Exception:
                text_cols.append(col)

        # --- Types de Données ---
        dtypes_data = [(col, str(dtype)) for col, dtype in data.dtypes.items()]
        
        # --- Valeurs nulles ---
        null_counts = data.isnull().sum()
        null_percentages = (null_counts / len(data)) * 100
        null_stats_df = pd.DataFrame({
            "Valeurs nulles": null_counts,
            "Pourcentage (%)": null_percentages.round(2)
        })

        colonnes_avec_null = (null_counts > 0).sum()
        colonnes_sans_null = (null_counts == 0).sum()

        # -------------------- Doublons --------------------
        # --- Toutes les lignes dupliquées ---
        all_duplicates = data[data.duplicated(keep=False)]
        duplicated_rows_all = all_duplicates.sort_values(by=list(data.columns))

        group_ids = (data != data.shift()).any(axis=1).cumsum()
        duplicated_consec = all_duplicates.groupby(group_ids).filter(lambda x: len(x) > 1)
        duplicated_rows_consecutive_df = duplicated_consec.groupby(group_ids).first().copy()
        duplicated_rows_consecutive_df['count_in_group'] = duplicated_consec.groupby(group_ids).size().values
        duplicated_rows_consecutive_df.reset_index(drop=True, inplace=True)

        unique_dups = all_duplicates.drop(index=duplicated_consec.index)
        non_consec = unique_dups.drop_duplicates()
        grouped = data.groupby(list(data.columns)).size()
        non_consec['count_duplicates'] = [grouped.get(tuple(row), 1) for _, row in non_consec.iterrows()]

        return render(request, "pages/informations_dataset.html", {
            "head": data.head().to_html(classes="table table-striped"),
            "tail": data.tail().to_html(classes="table table-striped"),
            "shape": f"{data.shape[0]} lignes, {data.shape[1]} colonnes",
            "lignes": data.shape[0],
            "colonnes": data.shape[1],
            "columns": list(data.columns),
            "numeric_colonne": numeric_cols,
            "text_colonne": text_cols,
            "date_colonne": date_cols,
            "num_columns": len(numeric_cols),
            "text_columns": len(text_cols),
            "date_columns": len(date_cols),
            "dtypes_data": dtypes_data,
            "null_values": null_stats_df.to_html(classes="table table-striped"),
            "null_values_list": [(col, int(null_counts[col]), round(null_percentages[col], 2)) for col in data.columns],
            "colonnes_avec_valeurs_nulles": colonnes_avec_null,
            "colonnes_sans_valeurs_nulles": colonnes_sans_null,
            "duplicated_rows_consecutive": duplicated_rows_consecutive_df.to_dict(orient="records"),
            "duplicated_non_consecutive": non_consec.to_dict(orient="records"),
            "duplicated_rows_all": duplicated_rows_all.to_dict(orient="records"),
            "duplicated_rows_consecutive_count": duplicated_rows_consecutive_df.shape[0],
            "total_lignes_doublons_consecutifs": duplicated_consec.shape[0],
            "duplicated_non_consecutive_count": non_consec.shape[0],
            "duplicates": all_duplicates.shape[0],
            "duplicated_rows_all_count": duplicated_rows_all.shape[0],
            "dataset_name": latest_entry.table_name,
            "original_filename": latest_entry.original_filename
        })

    except ValidationError as ve:
        return render(request, "pages/informations_dataset.html", {"error": str(ve)})
    except Exception as e:
        return render(request, "pages/informations_dataset.html", {"error": f"Erreur : {str(e)}"})
