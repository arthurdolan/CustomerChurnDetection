"""
Preprocessing and data loading utilities for churn prediction.
Handles data loading, cleaning, encoding, and feature preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, Union, IO


def _read_excel_smart(source: Union[str, Path, IO]) -> pd.DataFrame:
    """
    Read an Excel file, preferring the actual data sheet.

    For the bundled e-commerce dataset this means:
    - Prefer sheet named like 'E Comm' (the customer-level data)
    - Otherwise, if multiple sheets exist, take the second one
    - Fallback to the first sheet.
    """
    xls = pd.ExcelFile(source)
    sheet_name = None

    # Prefer sheet explicitly named like the data tab
    for name in xls.sheet_names:
        if name.strip().lower() in {"e comm", "ecomm", "e commerce", "e_commerce"}:
            sheet_name = name
            break

    # If not found, and there is more than one sheet, try the second sheet (skip data dict)
    if sheet_name is None:
        if len(xls.sheet_names) > 1:
            sheet_name = xls.sheet_names[1]
        else:
            sheet_name = xls.sheet_names[0]

    df = xls.parse(sheet_name)
    return df


def load_dataset(uploaded_file=None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Centralised dataset loading logic.

    Priority:
    1. If an uploaded file-like object is provided, load it.
    2. Otherwise, try to load a bundled dataset from the local data/ folder.

    Supported formats: CSV, XLSX.

    Returns:
        (dataframe or None, human-readable source description)
    """
    # 1. User-uploaded file (takes precedence)
    if uploaded_file is not None:
        file_name = getattr(uploaded_file, "name", "uploaded_file")
        lower_name = str(file_name).lower()

        try:
            if lower_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif lower_name.endswith((".xlsx", ".xls")):
                df = _read_excel_smart(uploaded_file)
            else:
                # Fallback: try CSV first, then Excel
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file)

            return df, f"Uploaded: {file_name}"
        except Exception as exc:
            raise RuntimeError(f"Failed to read uploaded file '{file_name}': {exc}") from exc

    # 2. Bundled default dataset from /data
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        # Preferred internal demo dataset
        base_dir / "data" / "E Commerce Dataset.xlsx",
        base_dir / "data" / "E Commerce Dataset.csv",
        # Generic fallbacks
        base_dir / "data" / "ecommerce_churn.csv",
        base_dir / "data" / "ecommerce_churn.xlsx",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                else:
                    df = _read_excel_smart(path)
                return df, f"Default: {path.name}"
            except Exception as exc:
                # If one candidate fails to load, try the next
                continue

    # Nothing could be loaded
    return None, "No dataset found (no upload and no default data file in /data)"


def identify_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], list, list]:
    """
    Identify key columns in the dataset.
    
    Returns:
        Tuple of (customer_id_col, churn_col, numeric_cols, categorical_cols)
    """
    # Find customer ID column (case-insensitive)
    customer_id_col = None
    for col in df.columns:
        if 'customerid' in col.lower() or 'customer_id' in col.lower():
            customer_id_col = col
            break
    
    # Find churn column (case-insensitive)
    churn_col = None
    for col in df.columns:
        if col.lower() == 'churn':
            churn_col = col
            break
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col == customer_id_col or col == churn_col:
            continue
        
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (low cardinality integers)
            if df[col].nunique() < 20 and df[col].nunique() < len(df) * 0.1:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return customer_id_col, churn_col, numeric_cols, categorical_cols


def preprocess_data(df: pd.DataFrame, 
                    customer_id_col: Optional[str],
                    churn_col: Optional[str],
                    numeric_cols: list,
                    categorical_cols: list,
                    fit_encoders: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Preprocess the dataset for modeling.
    
    Args:
        df: Input dataframe
        customer_id_col: Name of customer ID column
        churn_col: Name of churn column
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        fit_encoders: Optional dict of fitted encoders/scalers (for inference)
    
    Returns:
        Tuple of (processed_df, encoders_dict)
    """
    df_processed = df.copy()
    
    # Initialize encoders if not provided (training mode)
    if fit_encoders is None:
        fit_encoders = {
            'label_encoders': {},
            'scaler': StandardScaler()
        }
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        if col in df_processed.columns:
            if df_processed[col].isna().any():
                median_val = df_processed[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_processed[col].fillna(median_val, inplace=True)
    
    # Handle missing values in categorical columns
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col].fillna('Unknown', inplace=True)
            # Convert to string for consistency
            df_processed[col] = df_processed[col].astype(str)
    
    # Encode categorical features
    for col in categorical_cols:
        if col in df_processed.columns:
            if col not in fit_encoders['label_encoders']:
                # Create new encoder
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                fit_encoders['label_encoders'][col] = le
            else:
                # Use existing encoder
                le = fit_encoders['label_encoders'][col]
                # Handle unseen categories
                unique_vals = set(df_processed[col].astype(str).unique())
                known_classes = set(le.classes_)
                if unique_vals - known_classes:
                    # Add unknown category
                    df_processed[col] = df_processed[col].astype(str).apply(
                        lambda x: x if x in known_classes else 'Unknown'
                    )
                df_processed[col] = le.transform(df_processed[col].astype(str))
    
    # Scale numeric features
    if numeric_cols:
        numeric_cols_present = [col for col in numeric_cols if col in df_processed.columns]
        if numeric_cols_present:
            # Check if scaler needs to be fitted (training mode)
            is_training = fit_encoders['scaler'] is None or not hasattr(fit_encoders['scaler'], 'mean_')
            if is_training:
                # Fit scaler (training mode)
                fit_encoders['scaler'].fit(df_processed[numeric_cols_present])
            
            # Transform features
            df_processed[numeric_cols_present] = fit_encoders['scaler'].transform(
                df_processed[numeric_cols_present]
            )
    
    # Convert churn to binary if it exists
    if churn_col and churn_col in df.columns:
        churn_series = df[churn_col].copy()
        if churn_series.dtype == 'object':
            # Convert Yes/No, True/False, etc. to 0/1
            churn_series = churn_series.astype(str).str.lower()
            churn_series = churn_series.replace({
                'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
            })
            churn_series = pd.to_numeric(churn_series, errors='coerce').fillna(0)
        df_processed['_churn_target'] = churn_series.astype(int)
    
    return df_processed, fit_encoders


def prepare_features(df_processed: pd.DataFrame,
                    customer_id_col: Optional[str],
                    churn_col: Optional[str],
                    numeric_cols: list,
                    categorical_cols: list) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare feature matrix and target vector.
    
    Returns:
        Tuple of (X_features, y_target)
    """
    # Select feature columns
    feature_cols = [col for col in numeric_cols + categorical_cols if col in df_processed.columns]
    X = df_processed[feature_cols].copy()
    
    # Get target if available
    y = None
    if '_churn_target' in df_processed.columns:
        y = df_processed['_churn_target'].copy()
    
    return X, y
