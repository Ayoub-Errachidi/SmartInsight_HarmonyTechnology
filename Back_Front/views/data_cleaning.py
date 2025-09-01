import pandas as pd
import traceback
from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from ..models import DataNettoyer
from .views_importation import load_latest_dataset, recreate_table_from_df_copy, build_dataset_context
import logging

logger = logging.getLogger(__name__)


# -----  Supprimer des colonnes -----
def supprimer_colonnes_page(request: HttpRequest) -> HttpResponse:
    """
    Affiche la page avec la liste des colonnes du dernier dataset importé.
    """
    try:
        df, latest_entry = load_latest_dataset()
        colonnes_list = df.columns.tolist()  # conversion en liste
        return render(request, "pages/nettoyage/supprimer_colonnes.html", {
            "colonnes": colonnes_list
        })
    except Exception as e:
        logger.error("Erreur chargement colonnes : %s", traceback.format_exc())
        return render(request, "pages/nettoyage/supprimer_colonnes.html", {
            "error": f"Impossible de charger les colonnes : {e}"
        })

def supprimer_colonnes_action(request: HttpRequest) -> HttpResponse:
    """
    Supprime les colonnes sélectionnées et met à jour la table nettoyée.
    """
    if request.method == "POST":
        try:
            # Charger dataset original
            df, latest_entry = load_latest_dataset()

            # Colonnes sélectionnées depuis le formulaire
            colonnes_a_supprimer = request.POST.getlist("colonnes")

            if colonnes_a_supprimer:
                # Vérifier si le dataset n'a qu'une seule colonne
                if df.shape[1] <= 1:
                    return render(request, "pages/nettoyage/supprimer_colonnes.html", {
                        "error": "Impossible de supprimer la dernière colonne : le dataset doit contenir au moins une colonne."
                    })

                # Vérifier que la suppression ne supprime pas toutes les colonnes
                if len(colonnes_a_supprimer) >= df.shape[1]:
                    return render(request, "pages/nettoyage/supprimer_colonnes.html", {
                        "error": "Vous ne pouvez pas supprimer toutes les colonnes."
                    })
                
                # Supprimer seulement si au moins 1 colonne restera
                df = df.drop(columns=[c for c in colonnes_a_supprimer if c in df.columns], errors="ignore")

                # Nom unique pour la table nettoyée (toujours même nom !)
                base_name = latest_entry.table_name
                nettoyee_name = f"{base_name}_nettoye"

                # Remplacer complètement la table nettoyée
                recreate_table_from_df_copy(df, nettoyee_name)

                # Sauvegarde métadonnées dans DataNettoyer (si première fois)
                existing = DataNettoyer.objects.filter(table_name_nettoyee=nettoyee_name).first()
                if not existing:
                    DataNettoyer.objects.create(
                        table_name_nettoyee=nettoyee_name,
                        table_originale=latest_entry
                    )
                else:
                    # Mise à jour de la référence vers le dataset original
                    existing.table_originale = latest_entry
                    existing.save()

                return render(request, "pages/Dataset.html",
                              build_dataset_context(df, nettoyee_name, latest_entry.file_type, latest_entry.encoding))

            return render(request, "pages/nettoyage/supprimer_colonnes.html", {"error": "Aucune colonne sélectionnée."})

        except Exception as e:
            logger.error("Erreur suppression colonnes : %s", traceback.format_exc())
            return render(request, "pages/nettoyage/supprimer_colonnes.html", {"error": f"Erreur : {e}"})
    
    return redirect("supprimer_colonnes_page")



# -------------------- Supprimer les doublons --------------------
def remove_duplicates(data: pd.DataFrame):
    ''' Supprime les doublons et retourne informations détaillées '''
    duplicated_mask = data.duplicated(keep='first')
    deleted_rows = data[duplicated_mask].copy()
    deleted_indices = deleted_rows.index.tolist()

    # Groupes de doublons (clé = tuple de ligne)
    grouped = data[data.duplicated(keep=False)].copy()
    grouped['__row_key__'] = grouped.apply(lambda row: tuple(row), axis=1)

    groups = []
    for key, group_df in grouped.groupby('__row_key__'):
        indices = group_df.index.tolist()
        if len(indices) > 1:
            kept_idx = indices[0]
            removed_idxs = indices[1:]

            kept_row = data.loc[kept_idx].to_dict()
            removed_rows = [data.loc[idx].to_dict() for idx in removed_idxs]

            groups.append({
                "ligne_conservee_index": kept_idx,
                "ligne_conservee_data": kept_row,
                "lignes_supprimees_index": removed_idxs,
                "lignes_supprimees_data": removed_rows
            })

    cleaned_data = data.drop_duplicates(keep='first')
    duplicates_removed = len(deleted_rows)
    unique_deleted_indices = deleted_rows.drop_duplicates().index.tolist()
    return cleaned_data, duplicates_removed, deleted_rows, deleted_indices, unique_deleted_indices, groups

# -------------------- Supprimer les doublons dans table nettoyée --------------------
def supprimer_doublons_view(request: HttpRequest) -> HttpResponse:
    """
    Supprime les doublons du dernier dataset et met à jour la table nettoyée (_nettoye).
    """
    try:
        # Charger le dernier dataset depuis PostgreSQL
        df, latest_entry = load_latest_dataset()
        rows_before = df.shape[0]

        # Supprimer doublons
        cleaned_data, duplicates_removed, deleted_rows, deleted_indices, unique_deleted_indices, groups = remove_duplicates(df)
        rows_after = cleaned_data.shape[0]

        # Nom de la table nettoyée
        nettoyee_name = f"{latest_entry.table_name}_nettoye"

        # Remplacer la table nettoyée
        recreate_table_from_df_copy(cleaned_data, nettoyee_name)

        # Sauvegarder ou mettre à jour les métadonnées
        existing = DataNettoyer.objects.filter(table_name_nettoyee=nettoyee_name).first()
        if not existing:
            DataNettoyer.objects.create(
                table_name_nettoyee=nettoyee_name,
                table_originale=latest_entry
            )
        else:
            existing.table_originale = latest_entry
            existing.save()

        # Rendu HTML pour affichage
        table_deleted_html = deleted_rows.to_html(classes="table table-bordered table-danger", index=True)
        
        # Index des lignes conservées + Nombre de lignes conservées
        lignes_conservees_avec_doublons = [g["ligne_conservee_index"] for g in groups]
        nombre_lignes_conservees_avec_doublons = len(lignes_conservees_avec_doublons)

        return render(request, "pages/nettoyage/resultat_doublons.html", {
            "table_doublons_supprimes": table_deleted_html,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": duplicates_removed,
            "deleted_indices": deleted_indices,
            "unique_deleted_indices": unique_deleted_indices,
            "nombre_doublons_uniques": len(unique_deleted_indices),
            "groupes_doublons": groups,
            "lignes_conservees_avec_doublons": lignes_conservees_avec_doublons,
            "nombre_lignes_conservees_avec_doublons": nombre_lignes_conservees_avec_doublons
        })

    except Exception as e:
        logger.error("Erreur suppression doublons : %s\n%s", e, traceback.format_exc())
        return render(request, "pages/nettoyage/resultat_doublons.html", {"error": str(e)})


# -------------------- Nettoyage des valeurs manquantes --------------------
def clean_missing_values(data: pd.DataFrame, threshold_drop: float = 0.6) -> pd.DataFrame:
    cleaned_data = data.copy()
    n_rows = len(cleaned_data)
    col_thresh = n_rows * threshold_drop
    cleaned_data = cleaned_data.dropna(axis=1, thresh=col_thresh)

    for col in cleaned_data.columns:
        missing_ratio = cleaned_data[col].isnull().sum() / n_rows
        if missing_ratio == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned_data[col]):
            if missing_ratio < 0.1:
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear', limit_direction='forward', axis=0)
            elif missing_ratio < 0.4:
                cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
            else:
                cleaned_data = cleaned_data.dropna(subset=[col])

        elif pd.api.types.is_datetime64_any_dtype(cleaned_data[col]):
            try:
                cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0], inplace=True)
            except IndexError:
                cleaned_data[col].fillna(method='ffill', inplace=True)
        else:
            if missing_ratio < 0.5:
                mode_val = cleaned_data[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Inconnu"
                cleaned_data[col].fillna(fill_val, inplace=True)
            else:
                cleaned_data = cleaned_data.dropna(subset=[col])
    return cleaned_data


def nettoyer_dataset_view(request: HttpRequest) -> HttpResponse:
    """ Nettoie les valeurs manquantes et met à jour la table nettoyée """
    try:
        df, latest_entry = load_latest_dataset()
        rows_before, cols_before = df.shape
        columns_before = list(df.columns)

        cleaned_data = clean_missing_values(df)
        rows_after, cols_after = cleaned_data.shape
        columns_after = list(cleaned_data.columns)

        removed_columns = list(set(columns_before) - set(columns_after))
        cols_removed = len(removed_columns)

        cols_touched = [col for col in columns_after
                        if df[col].isnull().sum() > 0 and cleaned_data[col].isnull().sum() == 0]
        cols_filled = len(cols_touched)

        # Remplacer la table nettoyée
        nettoyee_name = f"{latest_entry.table_name}_nettoye"
        recreate_table_from_df_copy(cleaned_data, nettoyee_name)

        existing = DataNettoyer.objects.filter(table_name_nettoyee=nettoyee_name).first()
        if not existing:
            DataNettoyer.objects.create(table_name_nettoyee=nettoyee_name, table_originale=latest_entry)
        else:
            existing.table_originale = latest_entry
            existing.save()

        null_counts = cleaned_data.isnull().sum()
        null_percentages = (null_counts / len(cleaned_data)) * 100

        return render(request, "pages/nettoyage/resultat_nettoyage.html", {
            "table_nettoyer": cleaned_data.head(3000).to_html(classes="table table-striped table-hover"),
            "colonnes_avec_valeurs_nulles": (null_counts > 0).sum(),
            "colonnes_sans_valeurs_nulles": (null_counts == 0).sum(),
            "null_values_list": [(col, int(null_counts[col]), round(null_percentages[col], 2))
                                 for col in cleaned_data.columns],
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "cols_removed": cols_removed,
            "columns_after": columns_after,
            "removed_columns": removed_columns,
            "cols_filled": cols_filled,
            "cols_touched": cols_touched
        })
    except Exception as e:
        logger.error("Erreur nettoyage : %s\n%s", e, traceback.format_exc())
        return render(request, "pages/nettoyage/resultat_nettoyage.html", {"error": "Erreur lors du nettoyage des données."})
    

# ----------- Nettoyage des valeurs aberrantes -----------
def remove_outliers_iqr(df: pd.DataFrame, seuil: float = 0.1) -> tuple[pd.DataFrame, str, list[int]]:
    """
    Applique suppression ou winsorisation des outliers selon leur proportion.
    Retourne :
    - DataFrame nettoyé
    - Méthode appliquée ("Suppression" ou "Winsorisation")
    - Liste des index supprimés (vide si Winsorisation)
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include='number').columns

    total_rows = df_out.shape[0]
    indices_to_remove = set()

    for col in numeric_cols:
        q1 = df_out[col].quantile(0.25)
        q3 = df_out[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Détection des indices à supprimer
        outliers = df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)]
        indices_to_remove.update(outliers.index)

    outlier_ratio = len(indices_to_remove) / total_rows if total_rows > 0 else 0

    if outlier_ratio > seuil:
        # Trop d’outliers → Winsorisation
        for col in numeric_cols:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
        return df_out, "Winsorisation", []
    else:
        # Suppression
        df_out = df_out.drop(index=indices_to_remove)
        return df_out, "Suppression", sorted(list(indices_to_remove))


def remove_outliers_view(request: HttpRequest) -> HttpResponse:
    '''
    Supprime ou winsorise les valeurs aberrantes selon la proportion,
    puis met à jour la table nettoyée (_nettoye).
    '''
    try:
        # Charger dataset
        df, latest_entry = load_latest_dataset()
        nb_lignes_initial = df.shape[0]
        colonnes_numeriques = list(df.select_dtypes(include='number').columns)
        nb_colonnes_numeriques = len(colonnes_numeriques)

        # Nettoyer
        df_cleaned, methode, indices_supprimes = remove_outliers_iqr(df)
        nb_lignes_final = df_cleaned.shape[0]
        nb_supprimees = len(indices_supprimes)
        pourcentage_supprimees = round((nb_supprimees / nb_lignes_initial) * 100, 2) if nb_supprimees else 0.0

        # Nom de la table nettoyée
        nettoyee_name = f"{latest_entry.table_name}_nettoye"

        # Réécriture table nettoyée
        recreate_table_from_df_copy(df_cleaned, nettoyee_name)

        # Sauvegarde métadonnées
        existing = DataNettoyer.objects.filter(table_name_nettoyee=nettoyee_name).first()
        if not existing:
            DataNettoyer.objects.create(
                table_name_nettoyee=nettoyee_name,
                table_originale=latest_entry
            )
        else:
            existing.table_originale = latest_entry
            existing.save()

        # Contexte enrichi
        context = build_dataset_context(df_cleaned, latest_entry.original_filename, latest_entry.file_type, latest_entry.encoding)
        context.update({
            "methode_outliers": methode,
            "nb_lignes_initiales": nb_lignes_initial,
            "nb_lignes_finales": nb_lignes_final,
            "nb_colonnes_numeriques": nb_colonnes_numeriques,
            "colonnes_numeriques": colonnes_numeriques,
            "nb_lignes_supprimees": nb_supprimees,
            "pourcentage_supprimees": pourcentage_supprimees,
            "indices_supprimes": indices_supprimes if methode == "Suppression" else None,
            "seuil_outliers": 0.1
        })

        return render(request, "pages/nettoyage/outliers.html", context)

    except Exception as e:
        logger.error("Erreur nettoyage outliers : %s\n%s", e, traceback.format_exc())
        return render(request, "pages/nettoyage/outliers.html", {"error": f"Erreur lors du nettoyage des valeurs aberrantes : {e}"})


# ====================================================
# Prétraitement global : Doublons + Valeurs manquantes + Outliers
# ====================================================
def preprocess_dataset_for_ml(df: pd.DataFrame, return_info: bool = False):
    """
    Applique enchaînement de nettoyage :
    1. Suppression doublons
    2. Nettoyage des valeurs manquantes
    3. Suppression / Winsorisation des outliers

    Retourne un DataFrame propre prêt pour les transformations ML.
    """
    try:
        # 1. Supprimer doublons
        df_clean, duplicates_removed, deleted_rows, deleted_indices, unique_deleted_indices, groupes_doublons = remove_duplicates(df)

        # 2. Nettoyer valeurs manquantes
        df_clean = clean_missing_values(df_clean)

        # 3. Supprimer ou corriger outliers
        df_clean, methode_outliers, indices_supprimes = remove_outliers_iqr(df_clean)

        if return_info:
            stats_outliers = {
                "nb_lignes_supprimees": len(indices_supprimes),
                "indices_supprimes": indices_supprimes if methode_outliers == "Suppression" else None
            }
            return df_clean, methode_outliers, stats_outliers
        else:
            return df_clean

    except Exception as e:
        logger.error(f"Erreur lors du prétraitement global du dataset : {e}\n{traceback.format_exc()}")
        raise

# ====================================================
# Vue : Prétraitement complet avant ML
# ====================================================
def pretraitement_ml_view(request: HttpRequest) -> HttpResponse:
    """
    Applique le prétraitement global (doublons, valeurs manquantes, outliers)
    puis met à jour la table nettoyée (_nettoye).
    """
    try:
        # Charger dernier dataset
        df, latest_entry = load_latest_dataset()
        rows_before, cols_before = df.shape

        # Appliquer pipeline complet
        df_cleaned, methode_outliers, stats_outliers = preprocess_dataset_for_ml(df, return_info=True)
        rows_after, cols_after = df_cleaned.shape

        # Nom de la table nettoyée
        nettoyee_name = f"{latest_entry.table_name}_nettoye"

        # Réécriture de la table nettoyée
        recreate_table_from_df_copy(df_cleaned, nettoyee_name)

        # Sauvegarde ou mise à jour des métadonnées
        existing = DataNettoyer.objects.filter(table_name_nettoyee=nettoyee_name).first()
        if not existing:
            DataNettoyer.objects.create(table_name_nettoyee=nettoyee_name, table_originale=latest_entry)
        else:
            existing.table_originale = latest_entry
            existing.save()

        # Calcul des infos post-nettoyage
        null_counts = df_cleaned.isnull().sum()
        null_percentages = (null_counts / len(df_cleaned)) * 100

        context = build_dataset_context(
            df_cleaned,
            latest_entry.original_filename,
            latest_entry.file_type,
            latest_entry.encoding
        )

        context.update({
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "cols_removed": cols_before - cols_after,
            "colonnes_apres": list(df_cleaned.columns),
            "colonnes_nulles": [(col, int(null_counts[col]), round(null_percentages[col], 2))
                                for col in df_cleaned.columns],
            "methode_outliers": methode_outliers,
            "stats_outliers": stats_outliers
        })

        return render(request, "pages/nettoyage/pretraitement_global.html", context)

    except Exception as e:
        logger.error("Erreur prétraitement ML : %s\n%s", e, traceback.format_exc())
        return render(request, "pages/nettoyage/pretraitement_global.html", {
            "error": f"Erreur lors du prétraitement ML : {e}"
        })

# ====================================================
# Téléchargement du dataset nettoyé
# ====================================================

import io
from django.utils.encoding import smart_str

def download_dataset_view(request: HttpRequest) -> HttpResponse:
    '''
    Permet de télécharger le dernier dataset nettoyé (ou original si pas encore nettoyé) en format CSV
    '''
    try:
        # Récupérer le dernier dataset depuis PostgreSQL
        df, latest_entry = load_latest_dataset()

        # Nom du fichier à télécharger
        filename_base = latest_entry.original_filename.rsplit(".", 1)[0]
        filename = f"{filename_base}_nettoye.csv"

        # Conversion DataFrame → CSV en mémoire
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, encoding="utf-8-sig")  # UTF-8 BOM pour Excel
        buffer.seek(0)

        # Réponse HTTP avec fichier attaché
        response = HttpResponse(buffer.getvalue(), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{smart_str(filename)}"'
        return response

    except Exception as e:
        logger.error(f"Téléchargement échoué : {traceback.format_exc()}")
        return HttpResponse("Erreur lors du téléchargement du dataset.", status=500)
    