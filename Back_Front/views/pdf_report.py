# views/pdf_report.py
import io
from django.http import FileResponse, HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .views_importation import load_latest_dataset
from .data_statistiques import calculer_statistiques

def generate_pdf_report(request):
    try:
        # Charger le dataset
        df, latest_entry = load_latest_dataset()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', alignment=1, fontSize=16, spaceAfter=12))
        styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=10, textColor=colors.HexColor("#1a237e")))
        styles.add(ParagraphStyle(name='Small', fontSize=10, spaceAfter=6))

        # -------------------- TITRE --------------------
        elements.append(Paragraph("Rapport détaillé du Dataset", styles['CenterTitle']))
        elements.append(Paragraph(f"Nom du dataset : {latest_entry.table_name}", styles['Normal']))
        elements.append(Paragraph(f"Fichier original : {latest_entry.original_filename}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # -------------------- INFORMATIONS GENERALES --------------------
        elements.append(Paragraph("1. Informations générales", styles['Heading']))
        elements.append(Paragraph(f"Forme des données : {df.shape}", styles['Small']))
        elements.append(Paragraph(f"Nombre de lignes : {df.shape[0]}", styles['Small']))
        elements.append(Paragraph(f"Nombre de colonnes : {df.shape[1]}", styles['Small']))
        elements.append(Spacer(1, 12))

        # -------------------- LISTE DES COLONNES + TYPES DE DONNEES --------------------
        elements.append(Paragraph("2. Liste des colonnes avec types de données", styles['Heading']))

        # Préparer les données : numéro, nom de colonne, type
        col_data = [[i+1, col, str(dtype)] for i, (col, dtype) in enumerate(df.dtypes.items())]

        # Créer la table
        t_cols_types = Table([["#", "Nom de la colonne", "Type"]] + col_data, hAlign='LEFT')

        # Style de la table
        t_cols_types.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),12),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))

        elements.append(t_cols_types)
        elements.append(Spacer(1, 12))

        # -------------------- CATEGORIES DE COLONNES --------------------
        elements.append(Paragraph("3. Catégories de colonnes", styles['Heading']))

        numeric_cols = []
        text_cols = []
        date_cols = []

        for col in df.columns:
            series = df[col]

            # Si déjà datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols.append(col)
                continue

            # Si numérique pur
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                continue

            # Tester si c'est une date (au moins 80% des valeurs convertibles)
            try:
                converted = pd.to_datetime(series, errors='coerce', utc=True)
                valid_dates = converted.notna().sum()
                if valid_dates / len(series) >= 0.8:
                    date_cols.append(col)
                else:
                    text_cols.append(col)
            except Exception:
                text_cols.append(col)

        # Ajouter au PDF
        elements.append(Paragraph(
            f"Colonnes numériques ({len(numeric_cols)}) : {', '.join(numeric_cols) if numeric_cols else 'Aucune'}", 
            styles['Small']
        ))
        elements.append(Paragraph(
            f"Colonnes textuelles ({len(text_cols)}) : {', '.join(text_cols) if text_cols else 'Aucune'}", 
            styles['Small']
        ))
        elements.append(Paragraph(
            f"Colonnes de dates ({len(date_cols)}) : {', '.join(date_cols) if date_cols else 'Aucune'}", 
            styles['Small']
        ))
        elements.append(Spacer(1, 12))
        

        # -------------------- VALEURS MANQUANTES --------------------
        elements.append(Paragraph("4. Valeurs manquantes", styles['Heading']))
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df) * 100).round(2)
        null_table_data = [["Colonne", "Nb valeurs manquantes", "Pourcentage (%)"]]
        for col in df.columns:
            null_table_data.append([col, int(null_counts[col]), float(null_percentages[col])])

        t_null = Table(null_table_data, hAlign='LEFT')
        t_null.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        elements.append(t_null)
        elements.append(Spacer(1, 12))

        # -------------------- Résumé des colonnes avec valeurs manquantes --------------------
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            summary_text = f"Nombre de colonnes avec valeurs manquantes : {len(cols_with_nulls)}.<br/>"
            summary_text += "Détail : " + ", ".join([f"{col} ({null_percentages[col]}%)" for col in cols_with_nulls.index])
            elements.append(Paragraph(summary_text, styles['Normal']))
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph(" - Aucune colonne ne contient de valeurs manquantes.", styles['Normal']))
            elements.append(Spacer(1, 12))

        # -------------------- DOUBLONS --------------------
        elements.append(Paragraph("5. Lignes dupliquées", styles['Heading']))

        if df.empty:
            elements.append(Paragraph("Dataset vide – aucune analyse de doublons possible.", styles['Small']))
        else:
            # --- Toutes les lignes dupliquées ---
            all_duplicates = df[df.duplicated(keep=False)]
            duplicated_rows_all = all_duplicates.sort_values(by=list(df.columns))

            # --- Groupes consécutifs ---
            group_ids = (df != df.shift()).any(axis=1).cumsum()
            duplicated_consec = all_duplicates.groupby(group_ids).filter(lambda x: len(x) > 1)
            duplicated_rows_consecutive_df = duplicated_consec.groupby(group_ids).first().copy()
            duplicated_rows_consecutive_df['count_in_group'] = duplicated_consec.groupby(group_ids).size().values
            duplicated_rows_consecutive_df.reset_index(drop=True, inplace=True)

            # --- Doublons non consécutifs ---
            unique_dups = all_duplicates.drop(index=duplicated_consec.index)
            non_consec = unique_dups.drop_duplicates()
            grouped = df.groupby(list(df.columns)).size()
            non_consec['count_duplicates'] = [grouped.get(tuple(row), 1) for _, row in non_consec.iterrows()]

            # ---- Résumé numérique ----
            elements.append(Paragraph(f"Doublons non consécutifs : {non_consec.shape[0]} lignes.", styles['Small']))
            elements.append(Paragraph(f"Groupes de doublons consécutifs : {duplicated_rows_consecutive_df.shape[0]} groupes.", styles['Small']))
            elements.append(Paragraph(f"Total de lignes dans les doublons consécutifs : {duplicated_consec.shape[0]} lignes.", styles['Small']))
            elements.append(Paragraph(f"Nombre total de lignes dupliquées : {duplicated_rows_all.shape[0]} lignes.", styles['Small']))

            elements.append(Spacer(1, 8))

        # ==================== Partie Statistiques ====================
        stats = calculer_statistiques(df)

        # --- Statistiques de base (variables numériques) ---
        if stats["stats_base_dict"]:
            elements.append(Paragraph("6. Statistiques descriptives (variables numériques)", styles['Heading']))

            # Convertir le dictionnaire en DataFrame
            stats_base_df = pd.DataFrame(stats["stats_base_dict"])  # colonnes = colonnes du dataset, index = statistiques
            stats_base_df = stats_base_df.round(2)  # Limiter à 2 décimales
            stats_table = stats_base_df.T  # transpose pour que colonnes = statistiques

            # Préparer les données pour ReportLab
            data_table = [ ["Colonne"] + stats_table.columns.tolist() ]  # header
            for col_name, row in stats_table.iterrows():
                data_table.append([col_name] + row.tolist())

            # Créer la table
            table = Table(data_table, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#b3e5fc")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 12))

        # --- Statistiques avancées ---
        if stats["stats_adv_dict"]:
            elements.append(Paragraph("7. Statistiques avancées", styles['Heading']))

            # Convertir le dictionnaire avancé en DataFrame et transposer
            stats_adv_df = pd.DataFrame(stats["stats_adv_dict"]).T.round(2) 
            data_table = [ ["Colonne"] + stats_adv_df.columns.tolist()] + stats_adv_df.reset_index().values.tolist()

            # Créer la table
            table = Table(data_table, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#b3e5fc")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))


        # --- Statistiques textuelles ---
        if stats["stats_text_dict"]: 
            elements.append(Paragraph("8. Analyse des variables textuelles", styles['Heading']))

            for col, info in stats["stats_text_dict"].items():
                # Nom de la variable en gras et bleu foncé
                elements.append(Paragraph(f"<b><font color='#1F4E79'>{col}</font></b>", styles['Small']))

                # Nombre de valeurs uniques en retrait
                elements.append(Paragraph(
                    f"&nbsp;&nbsp;• <b>{info['nb_uniques']}</b> valeurs uniques", styles['Small']))

                # Exemples formatés
                exemples = ", ".join(info["values"])
                if len(exemples) > 120:
                    exemples = exemples[:120] + "..."

                elements.append(Paragraph(
                    f"&nbsp;&nbsp;• Exemples : <i>{exemples}</i>", styles['Small']))

                # Espacement après chaque variable
                elements.append(Spacer(1, 8))

            elements.append(Spacer(1, 12))







        # -------------------- CONSTRUCTION DU PDF --------------------
        doc.build(elements)
        buffer.seek(0)
        return FileResponse(buffer, as_attachment=True, filename=f"rapport_dataset_{latest_entry.table_name}.pdf")

    except Exception as e:
        return HttpResponse(f"Erreur lors de la génération du PDF : {str(e)}")
