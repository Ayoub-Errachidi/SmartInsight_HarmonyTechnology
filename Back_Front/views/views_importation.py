import logging
import io
import os
import re
import uuid
import pandas as pd
from django.shortcuts import render
from django.core.exceptions import ValidationError
from django.http import HttpRequest, HttpResponse
from django.db import connection, transaction
import chardet


from .clean_imported_data import clean_dataset
from ..models import DataOriginal, DataNettoyer, DataTransform


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.json')
FORBIDDEN_EXTENSIONS = ('.xlsm',)
MAX_FILE_SIZE_MB = 80  # Taille max (Mo)

# -------------------- Validation fichier --------------------
def validate_uploaded_file(file) -> None:
    '''
    Vérifie que le fichier uploadé est valide :
    - existe
    - ne dépasse pas la taille max
    - extension correcte
    '''
    if not file:
        raise ValidationError("Aucun fichier sélectionné.")
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValidationError(f"Le fichier dépasse {MAX_FILE_SIZE_MB} Mo.")
    
    # Validation extension simple
    if file.name.endswith(FORBIDDEN_EXTENSIONS):
        raise ValidationError("Les fichiers Excel avec macros (.xlsm) sont interdits.")
    if not file.name.lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValidationError("Format non supporté (CSV, Excel, JSON).")

# -------------------- Encodage --------------------
def detect_encoding(file):
    ''' Détecte l’encodage d’un fichier via chardet '''
    raw_data = file.read()
    detected = chardet.detect(raw_data)
    encoding_used = detected['encoding'] or 'utf-8'
    file.seek(0)
    return encoding_used, raw_data

# -------------------- Lecture fichier --------------------
def parse_uploaded_file(file) -> tuple[pd.DataFrame, str]:
    '''
    Lit le fichier en un DataFrame selon son type, et nettoie les faux NaN.
    '''
    try:
        encoding_used = 'utf-8'
        file_name = file.name.lower()

        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8', na_values=["", " ", "NA", "NaN", "null", "None"])
            except UnicodeDecodeError:
                file.seek(0)
                encoding_used, raw_data = detect_encoding(file)
                df = pd.read_csv(io.StringIO(raw_data.decode(encoding_used)),
                                 na_values=["", " ", "NA", "NaN", "null", "None"])

        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=["", " ", "NA", "NaN", "null", "None"])
            encoding_used = 'binary'

        elif file_name.endswith('.json'):
            df = pd.read_json(file)
            encoding_used = 'utf-8'
            if not isinstance(df, pd.DataFrame):
                raise ValidationError("Le JSON n'est pas un tableau valide.")

        else:
            raise ValidationError("Type de fichier non supporté.")

        df = clean_dataset(df)
        return df, encoding_used

    except Exception:
        logger.exception("Erreur lecture : %s", file.name)
        raise ValidationError("Le fichier est corrompu ou mal structuré.")

# -------------------- Sauvegarde en base PostgreSQL --------------------

# Supprime les plus anciennes Tables -> table PostgreSQL
def cleanup_old_datasets(max_tables=2):
    '''
    Supprime physiquement dans PostgreSQL les tables dataset_* les plus anciennes,
    pour les tables originales et nettoyées, mais garde les métadonnées en base.
    '''
    # ------------------- Tables originales -------------------
    datasets_original = list(
        DataOriginal.objects.filter(table_name__startswith="dataset_")
        .order_by('-uploaded_at')
    )

    if len(datasets_original) > max_tables:
        to_delete_original = datasets_original[max_tables:]
        with connection.cursor() as cursor:
            for entry in to_delete_original:
                try:
                    cursor.execute(f'DROP TABLE IF EXISTS "{entry.table_name}" CASCADE;')
                    logger.info(f"Table originale {entry.table_name} supprimée de PostgreSQL")
                except Exception as e:
                    logger.warning(f"Erreur suppression table originale {entry.table_name}: {e}")

    # ------------------- Tables nettoyées -------------------
    datasets_nettoyees = list(
        DataNettoyer.objects.select_related('table_originale')
        .order_by('-table_originale__uploaded_at')
    )

    if len(datasets_nettoyees) > max_tables:
        to_delete_nettoyees = datasets_nettoyees[max_tables:]
        with connection.cursor() as cursor:
            for entry in to_delete_nettoyees:
                try:
                    cursor.execute(f'DROP TABLE IF EXISTS "{entry.table_name_nettoyee}" CASCADE;')
                    logger.info(f"Table nettoyée {entry.table_name_nettoyee} supprimée de PostgreSQL")
                except Exception as e:
                    logger.warning(f"Erreur suppression table nettoyée {entry.table_name_nettoyee}: {e}")
    
    # -------------------- Nettoyage des tables transformées --------------------
    datasets_transformes = list(
        DataTransform.objects.select_related('table_nettoyee')
        .order_by('-uploaded_at')
    )

    if len(datasets_transformes) > max_tables:
        to_delete_transformes = datasets_transformes[max_tables:]
        with connection.cursor() as cursor:
            for entry in to_delete_transformes:
                try:
                    cursor.execute(f'DROP TABLE IF EXISTS "{entry.table_name_transformee}" CASCADE;')
                    logger.info(f"Table transformée {entry.table_name_transformee} supprimée de PostgreSQL")
                except Exception as e:
                    logger.warning(f"Erreur suppression table transformée {entry.table_name_transformee}: {e}")


# types d'origine des colonnes → PostgreSQL
def map_dtype_to_pg(dtype):
    ''' Mappe un dtype Pandas vers un type PostgreSQL '''
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"

# -------------------- Création table optimisée COPY FROM --------------------
@transaction.atomic
def recreate_table_from_df_copy(df: pd.DataFrame, table_name: str):
    """
    Remplace entièrement une table PostgreSQL par un DataFrame via COPY FROM.
    - Très rapide pour les gros datasets.
    - Tout est atomique : rollback si erreur.
    """
    # Nettoyer et préparer les colonnes
    columns_clean = [re.sub(r'\W+', '_', col.lower()) for col in df.columns]
    columns_sql_parts = [f'"{col}" {map_dtype_to_pg(df[col_name])}' 
                         for col, col_name in zip(df.columns, columns_clean)]
    columns_sql = ", ".join(columns_sql_parts)

    create_sql = f'CREATE TABLE "{table_name}" ({columns_sql});'

    # Convertir DataFrame en CSV en mémoire
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, sep=',', header=False, index=False, na_rep='\\N')  # '\\N' = NULL pour PostgreSQL
    csv_buffer.seek(0)

    with connection.cursor() as cursor:
        # Supprimer table si existe et créer nouvelle
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
        cursor.execute(create_sql)

        # COPY FROM en mémoire
        cursor.copy_expert(f'COPY "{table_name}" ({", ".join(columns_clean)}) FROM STDIN WITH CSV NULL AS \'\\N\'', csv_buffer)

# -------------------- Sauvegarde dataset --------------------
@transaction.atomic
def save_dataset_to_db(df, file_name, encoding):
    base_name = re.sub(r'\W+', '_', file_name.rsplit('.', 1)[0].lower())
    table_name = f"dataset_{base_name}_{uuid.uuid4().hex[:2]}"
    recreate_table_from_df_copy(df, table_name)
    
    DataOriginal.objects.create(
        table_name=table_name,
        original_filename=file_name,
        file_type=file_name.split('.')[-1].lower(),
        encoding=encoding
    )

    # Nettoyage automatique
    cleanup_old_datasets(max_tables=2)

# -------------------- Récupération depuis PostgreSQL --------------------
def load_latest_dataset():
    '''
    Récupère le dernier dataset depuis PostgreSQL.
    - Si une version nettoyée existe, elle est prioritaire.
    - Sinon, retourne le dataset original.
    '''
    latest_entry = DataOriginal.objects.order_by('-uploaded_at').first()
    if not latest_entry:
        raise ValidationError("Aucun dataset trouvé en base.")

    # Vérifier s’il existe une table nettoyée associée
    nettoyee = DataNettoyer.objects.filter(table_originale=latest_entry).first()
    if nettoyee:
        table_name = nettoyee.table_name_nettoyee
    else:
        table_name = latest_entry.table_name

    query = f'SELECT * FROM "{table_name}"'
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)

    # Retourne aussi la référence (originale) pour récupérer métadonnées (encoding, type, filename)
    return df, latest_entry

# -------------------- Contexte HTML --------------------
def build_dataset_context(df: pd.DataFrame, filename, file_type, encoding):
    sample = df.head(3000)
    return {
        "data": sample.to_html(classes="table table-striped table-hover"),
        "head_data": df.head().to_json(),
        "columns": df.shape[1],
        "lignes": df.shape[0],
        "filename": filename,
        "file_type": file_type,
        "encoding": encoding
    }

# -------------------- Vues --------------------
def dashboard_view(request: HttpRequest) -> HttpResponse:
    ''' Vue d'accueil : permet d'uploader un fichier '''
    if request.method == "POST":
        uploaded_file = request.FILES.get("type_file")
        try:
            # Vérification et parsing
            validate_uploaded_file(uploaded_file)
            df, encoding = parse_uploaded_file(uploaded_file)

            # Sauvegarde dans la base
            save_dataset_to_db(df, uploaded_file.name, encoding)

            # Séparation nom / extension
            filename_without_ext, file_ext = os.path.splitext(uploaded_file.name)
            file_ext = file_ext.lstrip(".")  # enlève le point

            return render(request, "pages/Dataset.html",
                          build_dataset_context(df, filename_without_ext, file_ext, encoding))

        except ValidationError as ve:
            return render(request, "pages/accueil.html", {"error": str(ve)})
        except Exception as e:
            logger.exception("Erreur upload")
            return render(request, "pages/accueil.html", {"error": str(e)})

    return render(request, "pages/accueil.html")

def dataset_view(request: HttpRequest) -> HttpResponse:
    ''' Vue dédiée à l'affichage du dernier dataset importé '''
    try:
        df, latest_entry = load_latest_dataset()
        return render(request, "pages/Dataset.html",
                      build_dataset_context(df, latest_entry.original_filename, latest_entry.file_type, latest_entry.encoding))
    
    except ValidationError as ve:
        return render(request, "pages/Dataset.html", {"error": str(ve)})
    
    except Exception as e:
        logger.exception("Erreur interne : dataset_view")
        return render(request, "pages/Dataset.html", {"error": f"{type(e).__name__} : {e}"})



# ------------------ Explication AI ------------------
from django.http import StreamingHttpResponse
from django.views.decorators.http import require_GET
from Utils.ollama_ai import stream_ai_explanation

@require_GET
def stream_explanation_view(request):
    prompt = request.GET.get("prompt", "")
    if not prompt:
        return StreamingHttpResponse("[Erreur] Aucun prompt fourni.", content_type="text/plain")

    def event_stream():
        for chunk in stream_ai_explanation(prompt):
            yield chunk

    return StreamingHttpResponse(event_stream(), content_type="text/plain")
