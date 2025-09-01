import pandas as pd
import re
import unicodedata

# --------- Nettoyage de Dataset après importation ---------

# --- Standardisation valeurs manquantes ---
def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les chaînes considérées comme nulles par des NaN réels.
    """
    return df.replace(
        to_replace=["", " ", "NA", "NaN", "nan", "NULL", "null", "None"],
        value=pd.NA
    )

# --- Standardisation noms colonnes ---
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Met les noms de colonnes en minuscules et en snake_case.
    """
    def to_snake_case(name):
        name = str(name).strip().lower()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name

    df.columns = [to_snake_case(col) for col in df.columns]
    return df

# --- Normalisation catégories ---
def normalize_word_variants(value: str) -> str:
    """
    Corrige les variantes d'un mot :
    - supprime les accents
    - met en minuscules
    """
    if not isinstance(value, str):
        return value
    
    # Supprimer les accents
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(c for c in value if not unicodedata.combining(c))
    
    return value.lower().strip()

def uniformize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Met toutes les valeurs textuelles (catégorielles) en minuscules
    et applique normalize_word_variants.
    """
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str).apply(normalize_word_variants)
    return df

"""

# -------------------- Correction orthographique --------------------
from textblob import TextBlob

def correct_spelling_textblob(df: pd.DataFrame, threshold_len: int = 3) -> pd.DataFrame:
    '''
    Corrige automatiquement les fautes de frappe dans les colonnes textuelles
    en utilisant TextBlob.
    '''
    text_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in text_cols:
        def correct_word(val):
            if not isinstance(val, str):
                return val
            val = val.strip()
            if len(val) < threshold_len:
                return val
            try:
                return str(TextBlob(val).correct())
            except Exception:
                return val  # En cas d'erreur, conserver la valeur originale
        
        df[col] = df[col].apply(correct_word)
    
    return df

"""

# --- Conversion automatique des dates ---
from dateutil.parser import parse

def detect_date_format(series: pd.Series) -> str | None:
    """
    Détecte le format dominant dans une série de dates.
    Retourne None si aucun format clair n'est trouvé.
    """
    from datetime import datetime
    import re

    # Supprimer valeurs nulles et convertir en string
    values = series.dropna().astype(str).head(50)  # On échantillonne 50 lignes max

    possible_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y%m%d", "%d.%m.%Y", "%m.%d.%Y"
    ]

    # Compter combien de dates matchent chaque format
    best_format = None
    best_count = 0
    for fmt in possible_formats:
        count = 0
        for val in values:
            try:
                datetime.strptime(val, fmt)
                count += 1
            except ValueError:
                pass
        if count > best_count and count > len(values) / 2:  # au moins la moitié
            best_format = fmt
            best_count = count

    return best_format


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit automatiquement les colonnes contenant 'date' en datetime,
    avec détection automatique du format dominant pour éviter le warning.
    """
    for col in df.columns:
        if 'date' in col.lower():
            fmt = detect_date_format(df[col])
            try:
                if fmt:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')  # fallback
            except Exception:
                pass
    return df


# --- Détection + Normalisation des colonnes date ---
def normalize_dates_to_day_month_year(df: pd.DataFrame, min_valid_fraction: float = 0.8) -> pd.DataFrame:
    """
    Détecte automatiquement les colonnes qui ressemblent à des dates (même en object/texte).
    Convertit et normalise toutes ces colonnes au format YYYY-MM-DD (string).
    """
    for col in df.columns:
        series = df[col]

        # Si déjà datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series.dt.strftime("%Y-%m-%d")
            continue

        # Si numérique → skip
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Essayer de parser comme date
        try:
            converted = pd.to_datetime(series, errors="coerce", utc=True)
            valid_ratio = converted.notna().mean()  # proportion de valeurs valides

            if valid_ratio >= min_valid_fraction:
                df[col] = converted.dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return df



# --- Pipeline global ---
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # Normaliser les noms de colonnes
    df = standardize_column_names(df)

    # Uniformiser les catégories
    df = uniformize_categories(df)

    # Standardiser valeurs manquantes
    df = standardize_missing_values(df)

    # Conversion types
    df = convert_date_columns(df)

    # Normaliser affichage dates (YYYY-MM-DD uniquement)
    df = normalize_dates_to_day_month_year(df)

    # Corrige automatiquement les fautes de frappe
    # df = correct_spelling_textblob(df)

    return df

# --------- Fin Nettoyage de Dataset après importation ---------
