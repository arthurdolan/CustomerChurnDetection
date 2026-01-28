"""
Churn prediction model logic.
Handles model training and heuristic fallback scoring.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def train_model(X: pd.DataFrame, y: pd.Series) -> Optional[LogisticRegression]:
    """
    Train a Logistic Regression model on the provided data.
    
    Args:
        X: Feature matrix
        y: Target vector (binary churn labels)
    
    Returns:
        Trained model or None if training fails
    """
    if len(X) < 2:
        return None
    
    if y.nunique() < 2:
        # Need at least 2 classes
        return None
    
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Model training failed: {e}")
        return None


def compute_heuristic_score(df: pd.DataFrame,
                           numeric_cols: list,
                           categorical_cols: list) -> pd.Series:
    """
    Compute churn probability using a simple heuristic.
    
    Heuristic factors:
    - High monthly charges (if available)
    - Short tenure (if available)
    - Month-to-month contracts (if available)
    
    Args:
        df: Original dataframe (before preprocessing)
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
    
    Returns:
        Series of churn probabilities (0-1)
    """
    scores = np.zeros(len(df))
    
    # Factor 1: Short tenure (lower tenure = higher churn risk)
    tenure_cols = [col for col in numeric_cols if 'tenure' in col.lower()]
    if tenure_cols:
        tenure_col = tenure_cols[0]
        if tenure_col in df.columns:
            tenure_values = pd.to_numeric(df[tenure_col], errors='coerce').fillna(df[tenure_col].median())
            max_tenure = tenure_values.max()
            if max_tenure > 0:
                # Normalize: lower tenure = higher score
                tenure_score = 1 - (tenure_values / max_tenure)
                scores += tenure_score * 0.4
    
    # Factor 2: High spend / order activity (if available)
    # Tailored for the e-commerce dataset: OrderCount, OrderAmountHikeFromlastYear, CashbackAmount
    charge_cols = [
        col
        for col in numeric_cols
        if any(
            term in col.lower()
            for term in [
                "monthly",
                "charge",
                "amount",
                "order",
                "cashback",
                "coupon",
                "spend",
            ]
        )
    ]
    if charge_cols:
        charge_col = charge_cols[0]
        if charge_col in df.columns:
            charge_values = pd.to_numeric(df[charge_col], errors='coerce').fillna(0)
            max_charge = charge_values.max()
            if max_charge > 0:
                # Normalize: higher charges = higher score
                charge_score = charge_values / max_charge
                scores += charge_score * 0.25
    
    # Factor 3: Engagement / recency (e-commerce specific)
    # More hours on app -> more engagement -> slightly LOWER risk
    hours_cols = [col for col in numeric_cols if "hourspendonapp" in col.lower()]
    if hours_cols:
        hours_col = hours_cols[0]
        if hours_col in df.columns:
            hours_values = pd.to_numeric(df[hours_col], errors="coerce").fillna(
                df[hours_col].median()
            )
            max_hours = hours_values.max()
            if max_hours > 0:
                engagement_score = 1 - (hours_values / max_hours)
                scores += engagement_score * 0.1
    
    # Days since last order: more days -> higher risk
    recency_cols = [col for col in numeric_cols if "daysincelastorder" in col.lower()]
    if recency_cols:
        recency_col = recency_cols[0]
        if recency_col in df.columns:
            recency_values = pd.to_numeric(df[recency_col], errors="coerce").fillna(
                df[recency_col].median()
            )
            max_days = recency_values.max()
            if max_days > 0:
                recency_score = recency_values / max_days
                scores += recency_score * 0.25
    
    # Factor 4: Low satisfaction score (if available)
    satisfaction_cols = [col for col in numeric_cols if 'satisfaction' in col.lower()]
    if satisfaction_cols:
        satisfaction_col = satisfaction_cols[0]
        if satisfaction_col in df.columns:
            sat_values = pd.to_numeric(df[satisfaction_col], errors='coerce').fillna(df[satisfaction_col].median())
            max_sat = sat_values.max()
            if max_sat > 0:
                # Normalize: lower satisfaction = higher score
                sat_score = 1 - (sat_values / max_sat)
                scores += sat_score * 0.2
    
    # Factor 5: Recent complaints (if available)
    complain_cols = [col for col in categorical_cols + numeric_cols if 'complain' in col.lower()]
    if complain_cols:
        complain_col = complain_cols[0]
        if complain_col in df.columns:
            complain_values = df[complain_col].astype(str).str.lower()
            has_complaint = complain_values.str.contains('yes|1|true', na=False)
            scores += has_complaint.astype(int) * 0.2
    
    # Normalize scores to 0-1 range
    if scores.max() > 0:
        scores = scores / scores.max()
    
    # Apply sigmoid-like transformation to make it more probability-like
    scores = 1 / (1 + np.exp(-5 * (scores - 0.5)))
    
    return pd.Series(scores, index=df.index)


def predict_churn(df: pd.DataFrame,
                  customer_id_col: Optional[str],
                  churn_col: Optional[str],
                  numeric_cols: list,
                  categorical_cols: list,
                  model: Optional[LogisticRegression],
                  X_processed: pd.DataFrame,
                  encoders: dict) -> pd.DataFrame:
    """
    Predict churn probabilities for all customers.
    
    Args:
        df: Original dataframe
        customer_id_col: Name of customer ID column
        churn_col: Name of churn column
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        model: Trained model (or None)
        X_processed: Processed feature matrix
        encoders: Fitted encoders/scalers
    
    Returns:
        DataFrame with customer IDs and churn probabilities
    """
    results = df.copy()
    
    # Try to use model first
    if model is not None:
        try:
            churn_probs = model.predict_proba(X_processed)[:, 1]
            results['churn_probability'] = churn_probs
        except Exception:
            # Fall back to heuristic
            results['churn_probability'] = compute_heuristic_score(
                df, numeric_cols, categorical_cols
            )
    else:
        # Use heuristic
        results['churn_probability'] = compute_heuristic_score(
            df, numeric_cols, categorical_cols
        )
    
    # Ensure probabilities are in [0, 1] range
    results['churn_probability'] = results['churn_probability'].clip(0, 1)
    
    # Add risk level
    results['risk_level'] = results['churn_probability'].apply(
        lambda x: 'High' if x > 0.6 else ('Medium' if x >= 0.3 else 'Low')
    )
    
    # Keep only essential columns in results
    output_cols = ['churn_probability', 'risk_level']
    if customer_id_col:
        output_cols.insert(0, customer_id_col)
    
    return results[output_cols]
