from django.urls import path
from Back_Front.views import views_importation, data_informations, data_cleaning, data_statistiques, data_visualisations, data_preprocessing, pdf_report

urlpatterns = [
    # Page d'accueil (redirection vers dashboard)
    path('', views_importation.dashboard_view, name='home'),
    
    # Dashboard principal
    path('dashboard/', views_importation.dashboard_view, name='dashboard'),


    # === === === === ===  DATASET === === === === ===
    # Affichage du dataset
    path('dataset/', views_importation.dataset_view, name='dataset'),
    # Supprimer des colonnes
    path("supprimer_colonnes/", data_cleaning.supprimer_colonnes_page, name="supprimer_colonnes_page"),
    path("supprimer_colonnes/action/", data_cleaning.supprimer_colonnes_action, name="supprimer_colonnes_action"),

    

    # === === === === ===  Informations DATASET === === === === ===
    # Informations générales sur le dataset
    path('informations_dataset/', data_informations.information_view, name='informations_dataset'),
    # Nettoyage des données
    path('nettoyage/', data_cleaning.nettoyer_dataset_view, name='nettoyage'),
    # Supprimer les doublons
    path('supprimer_doublons/', data_cleaning.supprimer_doublons_view, name='supprimer_doublons'),


    # === === === === ===  Statistiques DATASET === === === === ===
    # Statistiques descriptives du dataset
    path('statistiques/', data_statistiques.statistique_view, name='statistiques'),
    # Nettoyage des outliers
    path('nettoyer_outliers/', data_cleaning.remove_outliers_view, name='nettoyer_outliers'),


    # === === === === === Visualisations DATASET === === === === ===
    # Choix du type de visualisation
    path('visualisations/', data_visualisations.visualisations_view, name='visualisations'),
    # Génération plots
    path('select_pairplot/', data_visualisations.select_pairplot_columns_view, name='select_pairplot_columns_view'),
    # Sélection colonnes pour pair plot
    path('generate_plot/<str:plot_type>/', data_visualisations.plot_result_view, name='generate_plot'),


    # === === === === === Machine Learning === === === === ===
    # Prétraitement complet avant ML (doublons → valeurs manquantes → outliers)
    path('pretraitement_complet/', data_cleaning.pretraitement_ml_view, name='pretraitement_complet'),
    # Choisir Target 
    path('choix_cible/', data_preprocessing.choix_cible_view, name='choix_cible'),
    path('transformation_colonnes/', data_preprocessing.transformer_colonnes_view, name='transformation_colonnes'),
    # Télécharger dataset transforme
    path('telecharger_dataset_transforme/', data_preprocessing.telecharger_dataset_transforme, name='telecharger_dataset_transforme'),
    # Entraînement du Modèle 
    path('resultats_entrainement/', data_preprocessing.afficher_resultats_entrainement_view, name='resultats_entrainement'),
    # Prediction
    path('prediction/', data_preprocessing.prediction_view, name='prediction'),


    # Explication AI - Dataset -
    path('stream_explanation/', views_importation.stream_explanation_view, name='stream_explanation'),

    # Téléchargement du dataset nettoyé
    path('download_dataset/', data_cleaning.download_dataset_view, name='download_dataset'),

    # Téléchargement du dataset nettoyé
    path('rapport_pdf/', pdf_report.generate_pdf_report, name='rapport_pdf'),

]
