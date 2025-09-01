import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from .views_importation import load_latest_dataset

# -------------------- Helpers --------------------

def encode_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ===== =====  Affichage de Plot Histograms ===== ===== 
def generate_histograms(numeric_data):
    plots = []
    for col in numeric_data.columns:
        data = numeric_data[col].dropna()
        fig, ax = plt.subplots(figsize=(12, 5))

        mean, median, std = data.mean(), data.median(), data.std()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        mode_val = data.mode().iloc[0] if not data.mode().empty else None

        sns.histplot(data, bins=30, kde=True, color='cornflowerblue',
                     edgecolor='black', alpha=0.8, stat='density', ax=ax)

        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f"Moyenne: {mean:.2f}")
        ax.axvline(median, color='green', linestyle='-', linewidth=2, label=f"Médiane: {median:.2f}")
        ax.axvline(q1, color='orange', linestyle=':', linewidth=1.5, label=f"Q1: {q1:.2f}")
        ax.axvline(q3, color='purple', linestyle=':', linewidth=1.5, label=f"Q3: {q3:.2f}")
        if mode_val is not None:
            ax.axvline(mode_val, color='blue', linestyle='-.', linewidth=1.5, label=f"Mode: {mode_val:.2f}")

        ax.annotate(f"Écart-type: {std:.2f}", xy=(0.98, 0.90), xycoords='axes fraction',
                    ha='right', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

        ax.set_title(f"Histogramme : {col}", fontsize=15)
        ax.set_xlabel(col)
        ax.set_ylabel("Densité")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plots.append({
            'col_name': col,
            'image': encode_plot_to_base64(fig),
            'mean': round(mean, 2),
            'median': round(median, 2),
            'std': round(std, 2),
            'q1': round(q1, 2),
            'q3': round(q3, 2),
            'mode': round(mode_val, 2) if mode_val is not None else None
        })

    return plots


# ===== =====  Affichage de Plot Box ===== ===== 
def generate_boxplots(numeric_data):
    plots = []

    for col in numeric_data.columns:
        data = numeric_data[col].dropna()
        fig, ax = plt.subplots(figsize=(8, 6))

        # --- Calcul des statistiques ---
        Q1 = np.percentile(data, 25)
        Median = np.median(data)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_whisker = max(data.min(), Q1 - 1.5 * IQR)
        upper_whisker = min(data.max(), Q3 + 1.5 * IQR)
        outliers = data[(data < lower_whisker) | (data > upper_whisker)]

        # --- Boxplot avec seaborn ---
        sns.boxplot(y=data, ax=ax, color='lightblue', width=0.4, fliersize=6)

        # --- Annotations Q1, Médiane, Q3 ---
        ax.annotate(f"Q1: {Q1:.2f}", xy=(0, Q1), xytext=(-50, 0),
                    textcoords="offset points", ha='right', va='center', fontsize=9, color="brown", fontweight='bold')
        ax.annotate(f"Médiane: {Median:.2f}", xy=(0, Median), xytext=(50, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=9, color="green", fontweight='bold')
        ax.annotate(f"Q3: {Q3:.2f}", xy=(0, Q3), xytext=(50, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=9, color="purple", fontweight='bold')

        # --- Annotation IQR avec léger décalage vertical pour éviter chevauchement ---
        ax.annotate(f"IQR: {IQR:.2f}", xy=(0, (Q1 + Q3)/2), xytext=(-50, 0),  # décalage vertical
                    textcoords="offset points", ha='right', va='center', fontsize=9, color="brown", fontweight='bold')

        # --- Min/Max ---
        ax.annotate(f"Min: {lower_whisker:.2f}", xy=(0, lower_whisker), xytext=(5, -5),
                    textcoords="offset points", ha='left', va='top', fontsize=9, color="blue")
        ax.annotate(f"Max: {upper_whisker:.2f}", xy=(0, upper_whisker), xytext=(5, 5),
                    textcoords="offset points", ha='left', va='bottom', fontsize=9, color="blue")

        # --- Outliers ---
        if len(outliers) > 0:
            ax.scatter([0]*len(outliers), outliers, color='red', s=30, zorder=10, label='Outliers')
        # --- Légende seulement si des labels valides ---
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=9)

        # --- Style final ---
        ax.set_title(f"Box Plot : {col}", fontsize=14, fontweight='bold')
        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks([])

        plots.append({
            'col_name': col,
            'image': encode_plot_to_base64(fig),
            'Q1': round(Q1, 2),
            'Median': round(Median, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2),
            'Min': round(lower_whisker, 2),
            'Max': round(upper_whisker, 2),
            'Outliers': outliers.tolist()
        })

    return plots

# ===== ===== Affichage de Plot Scatter ===== ===== 
def generate_scatterplot(numeric_data, col_x, col_y):
    clean_data = numeric_data[[col_x, col_y]].dropna()

    if clean_data.empty:
        raise ValueError("Pas assez de données valides pour générer un scatter plot avec les colonnes sélectionnées.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=col_x, y=col_y, data=clean_data, ax=ax,
                    color="cornflowerblue", edgecolor="k", alpha=0.7, s=60)

    # --- Régression linéaire optionnelle (tendance) ---
    sns.regplot(x=col_x, y=col_y, data=clean_data,
                scatter=False, ax=ax, color="red", line_kws={"linewidth": 2, "alpha": 0.7})

    # --- Corrélation ---
    corr = clean_data[col_x].corr(clean_data[col_y])
    ax.annotate(f"Corrélation: {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction",
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

    ax.set_title(f"Scatter Plot : {col_x} vs {col_y}", fontsize=14, fontweight='bold')
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.grid(True, linestyle='--', alpha=0.5)

    encoded_image = encode_plot_to_base64(fig)
    return {
        'image': encoded_image,
        'title': f"Scatter Plot : {col_x} vs {col_y}",
        'corr': round(corr, 2)
    }


# ===== ===== Affichage de Line Plots par colonne ===== ===== 
def generate_lineplots(numeric_data):
    plots = []

    for col in numeric_data.columns:
        data = numeric_data[col].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))

        # --- Tracer ligne principale ---
        ax.plot(data.index, data.values, linestyle='-', color='cornflowerblue', alpha=0.7, linewidth=2, zorder=1)

        # --- Ajouter points plus visibles ---
        ax.scatter(data.index, data.values, color='darkblue', s=40, edgecolor='white', zorder=2, alpha=0.9)

        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        # Lignes moyennes et médiane
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Moyenne: {mean_val:.2f}", zorder=3)
        ax.axhline(median_val, color='green', linestyle='-', linewidth=1.5, label=f"Médiane: {median_val:.2f}", zorder=3)

        # Annotation écart-type
        ax.annotate(f"Écart-type: {std_val:.2f}", xy=(0.98, 0.90), xycoords='axes fraction',
                    ha='right', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

        ax.set_title(f"Line Plot : {col}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)

        plots.append({
            'col_name': col,
            'image': encode_plot_to_base64(fig),
            'mean': round(mean_val, 2),
            'median': round(median_val, 2),
            'std': round(std_val, 2)
        })

    return plots

# ===== ===== Affichage de Violin Plots par colonne ===== =====
def generate_violinplots(numeric_data):
    plots = []

    for col in numeric_data.columns:
        data = numeric_data[col].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))

        # Violin plot principal
        sns.violinplot(y=data, ax=ax, color='cornflowerblue', inner=None, linewidth=1.2)

        # Ajouter points individuels avec jitter pour visualiser chaque donnée
        sns.stripplot(y=data, ax=ax, color='darkblue', size=6, jitter=0.15, alpha=0.8, edgecolor='white', zorder=2)

        # Statistiques
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        q1, q3 = data.quantile([0.25, 0.75])

        # Lignes moyenne et médiane
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Moyenne: {mean_val:.2f}", zorder=3)
        ax.axhline(median_val, color='green', linestyle='-', linewidth=1.5, label=f"Médiane: {median_val:.2f}", zorder=3)

        # Annotations des stats
        ax.annotate(f"Écart-type: {std_val:.2f}", xy=(0.98, 0.92), xycoords='axes fraction',
                    ha='right', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
        ax.annotate(f"Q1: {q1:.2f}", xy=(0.98, 0.85), xycoords='axes fraction', ha='right', fontsize=9)
        ax.annotate(f"Q3: {q3:.2f}", xy=(0.98, 0.78), xycoords='axes fraction', ha='right', fontsize=9)

        # Style final
        ax.set_title(f"Violin Plot : {col}", fontsize=14, fontweight='bold')
        ax.set_ylabel(col)
        ax.set_xticks([])  # pas besoin d'axe X
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)

        # Encoder en base64 pour Django
        plots.append({
            'col_name': col,
            'image': encode_plot_to_base64(fig),
            'mean': round(mean_val, 2),
            'median': round(median_val, 2),
            'std': round(std_val, 2),
            'Q1': round(q1, 2),
            'Q3': round(q3, 2)
        })

    return plots

# ===== ===== Affichage de Pair Plots pour 2 colonnes ===== =====
def generate_pairplot(numeric_data, col1, col2):
    clean_data = numeric_data[[col1, col2]].dropna()

    if clean_data.empty:
        raise ValueError("Pas assez de données valides pour générer un pair plot avec les colonnes sélectionnées.")

    sns.set(style="whitegrid")
    pairplot = sns.pairplot(
        clean_data,
        diag_kind='kde',
        corner=False,
        plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'},
        diag_kws={'shade': True, 'color': 'blue'}
    )
    pairplot.fig.suptitle(f"{col1} vs {col2}", fontsize=16, y=1.02)

    encoded_image = encode_plot_to_base64(pairplot.fig)
    return {'image': encoded_image, 'title': "Pair Plot"}


def generate_single_plot(numeric_data, plot_type, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 8))

    if plot_type == "box":
        # Fonction pour générer un boxplot par colonne
        plot_images = generate_boxplots(numeric_data)
        if len(plot_images) == 1:
            # Si une seule colonne, utiliser plot_image
            return {'plot_image': plot_images[0]['image'], 'title': "Box Plot"}
        else:
            # Si Plusieurs colonnes, utiliser plot_images
            return {'plot_images': plot_images, 'title': "Box Plots"}


    elif plot_type == "heatmap":
        corr_matrix = numeric_data.corr()

        # --- Heatmap seaborn ---
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, center=0,
                    fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})

        # Récupérer le nom du fichier depuis le contexte
        raw_file_name = kwargs.get("file_name", "dataset")
        # Récupérer le nom du fichier sans extension
        file_name = os.path.splitext(raw_file_name)[0]

        title = f"Matrice de corrélation de fichier {raw_file_name}"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Nom du fichier pour téléchargement
        download_filename = f"heatmap_{file_name}.png"

        # --- Détection multicolinéarité ---
        high_corr_pairs = []
        threshold = 0.6  # seuil de forte corrélation
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "corr": round(corr_value, 2)
                    })

        encoded_image = encode_plot_to_base64(fig)
        return {
            'plot_image': encoded_image,
            'title': title,
            'high_corr_pairs': high_corr_pairs,
            'download_filename': download_filename
        }

    elif plot_type == "bar":
        # Calcul des moyennes
        means = numeric_data.mean().sort_values(ascending=False)

        # Palette dégradée (seaborn ou matplotlib colormap)
        colors = plt.cm.viridis(np.linspace(0, 1, len(means)))

        # Bar plot avec tri décroissant
        bars = ax.bar(means.index, means.values, color=colors, edgecolor="black", linewidth=0.7)

        # Ajout des valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6),  # espace au-dessus
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight="bold", color="#333")

        # Récupérer le nom du fichier depuis le contexte
        raw_file_name = kwargs.get("file_name", "dataset")

        # Récupérer le nom du fichier sans extension
        file_name = os.path.splitext(raw_file_name)[0]
        
        title = f"Bar Plot des moyennes par variable - fichier {raw_file_name}"

        # Nom du fichier pour téléchargement
        download_filename = f"bar_{file_name}.png"

        # Style
        ax.set_ylabel("Valeur moyenne", fontsize=12, fontweight="bold")
        ax.set_xlabel("Variables", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Grille horizontale discrète
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
        encoded_image = encode_plot_to_base64(fig)
        return {
            'plot_image': encoded_image,
            'title': title,
            'download_filename': download_filename
        }

    elif plot_type == "line":
        plot_images = generate_lineplots(numeric_data)

        if len(plot_images) == 1:
            return {'plot_image': plot_images[0]['image'], 'title': "Line Plot"}
        else:
            return {'plot_images': plot_images, 'title': "Line Plots"}

    elif plot_type == "violin":
        plot_images = generate_violinplots(numeric_data)

        if len(plot_images) == 1:
            return {'plot_image': plot_images[0]['image'], 'title': "Violin Plot"}
        else:
            return {'plot_images': plot_images, 'title': "Violin Plots"}
    
    elif plot_type == "pair":
        # Récupérer le nom du fichier depuis le contexte
        raw_file_name = kwargs.get("file_name", "dataset")

        # Récupérer le nom du fichier sans extension
        file_name = os.path.splitext(raw_file_name)[0]

        # Nom du fichier pour téléchargement
        download_filename = f"pair_{file_name}.png"

        # Style seaborn
        sns.set(style="whitegrid")

        # Création du pairplot avec options
        pairplot = sns.pairplot(
            numeric_data,
            diag_kind="kde",   # distribution lissée sur la diagonale
            corner=True,       # évite la duplication inutile
            plot_kws={
                "alpha": 0.6,
                "s": 40,
                "edgecolor": "k"
            },
            diag_kws={
                "fill": True,
                "color": "cornflowerblue"
            }
        )

        # Titre global
        pairplot.fig.suptitle(
            f"Pair Plot - fichier {raw_file_name}\n(affichage de {numeric_data.shape[1]} variables)",
            fontsize=14,
            fontweight="bold",
            y=1.02
        )

        fig = pairplot.fig
        title = f"Pair Plot - fichier {raw_file_name}"
    

    else:
        raise ValueError("Type de graphe non reconnu")

    
    encoded_image = encode_plot_to_base64(fig)
    return {
        'plot_image': encoded_image,
        'title': title,
        'download_filename': download_filename
    }

# -------------------- Vues --------------------

def visualisations_view(request: HttpRequest):
    try:
        df, _ = load_latest_dataset()
    except Exception:
        return redirect("dashboard_view")

    preview = df.head().to_html(classes="table table-bordered table-striped", index=False)
    return render(request, "pages/visualisations.html", {"preview": preview})


def select_pairplot_columns_view(request: HttpRequest):
    try:
        df, _ = load_latest_dataset()
    except Exception:
        return redirect("dashboard_view")

    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    return render(request, "pages/visualisations/pair.html", {
        "columns": numeric_columns,
        "plot_image": None,
        "col1": None,
        "col2": None,
        "title": "Sélection des colonnes pour Pair Plot"
    })


def plot_result_view(request: HttpRequest, plot_type: str):
    try:
        # Charger le dernier dataset
        df, _ = load_latest_dataset()
        numeric_data = df.select_dtypes(include='number')

        if numeric_data.empty:
            return HttpResponse("Aucune colonne numérique détectée.", status=400)

        # Histogrammes
        if plot_type == "histogram":
            plots = generate_histograms(numeric_data)
            return render(request, "pages/visualisations/histogram.html", {"plot_images": plots, "title": "Histogrammes"})

        # Box
        if plot_type == "box":
            plots = generate_boxplots(numeric_data)
            return render(request, "pages/visualisations/histogram.html", {"plot_images": plots, "title": "Box Plots"})
        
        # Line
        if plot_type == "line":
            plots = generate_lineplots(numeric_data)
            return render(request, "pages/visualisations/histogram.html", {"plot_images": plots, "title": "Line Plot"})
        
        # Violin
        if plot_type == "violin":
            plots = generate_violinplots(numeric_data)
            return render(request, "pages/visualisations/histogram.html", {"plot_images": plots, "title": "Violin Plot"})

        # Pair plot sélection
        if plot_type == "pair1":
            col1 = request.GET.get('col1')
            col2 = request.GET.get('col2')
            if col1 not in numeric_data.columns or col2 not in numeric_data.columns:
                return HttpResponse("Les colonnes sélectionnées n'existent pas.", status=400)

            result = generate_pairplot(numeric_data, col1, col2)
            return render(request, "pages/visualisations/pair.html", {
                "plot_image": result['image'],
                "columns": numeric_data.columns,
                "col1": col1,
                "col2": col2,
                "title": result['title']
            })
        
        # Scatter plot sélection
        if plot_type == "scatter":
            col_x = request.GET.get('col_x')
            col_y = request.GET.get('col_y')

            # --- Premier affichage : juste montrer le formulaire sans erreur ---
            if not col_x or not col_y:
                return render(request, "pages/visualisations/scatter.html", {
                    "plot_image": None,
                    "columns": numeric_data.columns,
                    "col_x": col_x,
                    "col_y": col_y,
                    "title": "Scatter Plot"
                })
            
            # --- Vérification après soumission ---
            if col_x not in numeric_data.columns or col_y not in numeric_data.columns:
                return render(request, "pages/visualisations/scatter.html", {
                    "plot_image": None,
                    "columns": numeric_data.columns,
                    "col_x": col_x,
                    "col_y": col_y,
                    "title": "Scatter Plot",
                    "error": "Les colonnes sélectionnées n'existent pas."
                }, status=400)
            
            # --- Génération du scatter ---
            result = generate_scatterplot(numeric_data, col_x, col_y)
            return render(request, "pages/visualisations/scatter.html", {
                "plot_image": result['image'],
                "columns": numeric_data.columns,
                "col_x": col_x,
                "col_y": col_y,
                "title": result['title'],
                "corr": result['corr']
            })

        if plot_type == "heatmap":
            result = generate_single_plot(numeric_data, plot_type, file_name=_.original_filename)
            return render(request, "pages/visualisations/histogram.html", {
                "plot_image": result['plot_image'],
                "title": result['title'],
                "high_corr_pairs": result['high_corr_pairs'],
                "download_filename": result['download_filename']
            })

        else:
            result = generate_single_plot(numeric_data, plot_type, file_name=_.original_filename)
            context = {
                "title": result['title'],
                # Nom du fichier pour téléchargement
                "download_filename": result['download_filename']
            }

            # Gérer les box plots (1 ou plusieurs colonnes)
            if 'plot_images' in result:
                context['plot_images'] = result['plot_images']
            elif 'plot_image' in result:
                context['plot_image'] = result['plot_image']

            return render(request, "pages/visualisations/histogram.html", context)

    except Exception as e:
        return HttpResponse(f"Erreur : {str(e)}", status=500)
