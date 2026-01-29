"""
Streamlit web app for customer churn prediction and retention.
Main application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import load_dataset, identify_columns, preprocess_data, prepare_features
from model import train_model, predict_churn
from actions import (
    send_retention_email,
    offer_discount,
    flag_for_call,
    prioritize_success_call,
    send_personalized_retention_email,
    flag_short_term_monitoring,
    send_feature_education_email,
    send_usage_reminder,
    add_to_experiment_cohort,
    issue_loyalty_reward,
    send_upsell_recommendation,
    send_referral_incentive,
    request_feedback_review,
    get_action_history,
)

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# No custom CSS per requirements

# Initialize session state
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "model" not in st.session_state:
    st.session_state.model = None
if "encoders" not in st.session_state:
    st.session_state.encoders = None
if "customer_id_col" not in st.session_state:
    st.session_state.customer_id_col = None
if "churn_col" not in st.session_state:
    st.session_state.churn_col = None
if "numeric_cols" not in st.session_state:
    st.session_state.numeric_cols = []
if "categorical_cols" not in st.session_state:
    st.session_state.categorical_cols = []
if "high_risk_threshold" not in st.session_state:
    st.session_state.high_risk_threshold = 0.6
if "dataset_source_name" not in st.session_state:
    st.session_state.dataset_source_name = None
if "dataset_row_count" not in st.session_state:
    st.session_state.dataset_row_count = 0


def compute_churn_scores(df: pd.DataFrame, show_messages: bool = False) -> pd.DataFrame:
    """Run the full churn pipeline."""
    customer_id_col, churn_col, numeric_cols, categorical_cols = identify_columns(df)
    st.session_state.customer_id_col = customer_id_col
    st.session_state.churn_col = churn_col
    st.session_state.numeric_cols = numeric_cols
    st.session_state.categorical_cols = categorical_cols

    df_processed, encoders = preprocess_data(
        df, customer_id_col, churn_col, numeric_cols, categorical_cols
    )
    st.session_state.encoders = encoders

    X, y = prepare_features(
        df_processed, customer_id_col, churn_col, numeric_cols, categorical_cols
    )

    model = None
    if churn_col and y is not None and y.nunique() >= 2:
        model = train_model(X, y)
        if show_messages:
            if model:
                st.write("Model trained successfully using Logistic Regression.")
            else:
                st.caption("Model training failed — using heuristic scoring instead.")
    else:
        if show_messages:
            st.caption("No churn labels found — using heuristic scoring.")

    st.session_state.model = model

    df_results = predict_churn(
        df, customer_id_col, churn_col, numeric_cols, categorical_cols,
        model, X, encoders,
    )
    st.session_state.df_results = df_results
    st.session_state.dataset_row_count = len(df)

    return df_results


# Preload default dataset
if st.session_state.df_raw is None:
    df_default, default_source = load_dataset()
    if df_default is not None:
        st.session_state.df_raw = df_default
        st.session_state.dataset_source_name = default_source
        try:
            compute_churn_scores(df_default, show_messages=False)
        except Exception:
            st.session_state.df_results = None


def render_upload_page():
    """Render the Upload & Dataset Overview page."""
    st.title("Dataset Overview")
    st.write(
        "Upload a dataset to analyze customer churn. The app accepts CSV and Excel files. "
        "Each row should represent one customer with features like tenure, charges, and demographics."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file with customer data"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing dataset..."):
                df, source_label = load_dataset(uploaded_file)
                st.session_state.df_raw = df
                st.session_state.dataset_source_name = source_label
                compute_churn_scores(df, show_messages=True)
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        source = st.session_state.dataset_source_name or "Dataset"

        st.write(
            f"Active dataset: {source.split(':')[-1].strip()} | "
            f"Rows: {len(df):,} | Columns: {len(df.columns)}"
        )

        customer_id_col = st.session_state.customer_id_col
        churn_col = st.session_state.churn_col
        numeric_cols = st.session_state.numeric_cols
        categorical_cols = st.session_state.categorical_cols

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Features", len(numeric_cols) + len(categorical_cols))
        with col3:
            if churn_col and churn_col in df.columns:
                churn_count = df[churn_col].astype(str).str.lower().str.contains("yes|1|true").sum()
                st.metric("Churn Rate", f"{churn_count/len(df):.1%}")
            else:
                st.metric("Churn Column", "Not found")
        with col4:
            st.metric("ID Column", "Found" if customer_id_col else "Not found")

        st.markdown("### Dataset Structure")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": ["Numeric" if c in numeric_cols else ("Categorical" if c in categorical_cols else "ID/Target") for c in df.columns],
            "Missing": [df[c].isna().sum() for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

        with st.expander("Preview Data"):
            st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    else:
        st.write("Upload a dataset to get started, or place a default file in the data/ folder.")


def render_churn_dashboard():
    """Render the Churn Prediction Dashboard page."""
    st.title("Churn Scoring")

    if st.session_state.df_results is None:
        st.write("No data available. Please upload a dataset first.")
        return

    df_results = st.session_state.df_results.copy()

    st.markdown("### Threshold Settings")
    threshold = st.slider(
        "High Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.high_risk_threshold,
        step=0.05,
        format="%.2f"
    )
    st.session_state.high_risk_threshold = threshold

    st.caption(
        f"Risk categories — Low: < 0.30 | Medium: 0.30–{threshold:.2f} | High: > {threshold:.2f}"
    )

    df_results['risk_level'] = df_results['churn_probability'].apply(
        lambda x: 'High' if x > threshold else ('Medium' if x >= 0.3 else 'Low')
    )

    st.markdown("### Risk Summary")
    high_count = (df_results['risk_level'] == 'High').sum()
    medium_count = (df_results['risk_level'] == 'Medium').sum()
    low_count = (df_results['risk_level'] == 'Low').sum()
    avg_prob = df_results['churn_probability'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High Risk", f"{high_count:,}", f"{high_count/len(df_results)*100:.1f}%")
    with col2:
        st.metric("Medium Risk", f"{medium_count:,}", f"{medium_count/len(df_results)*100:.1f}%")
    with col3:
        st.metric("Low Risk", f"{low_count:,}", f"{low_count/len(df_results)*100:.1f}%")
    with col4:
        st.metric("Avg Probability", f"{avg_prob:.3f}")

    st.markdown("### Distribution")
    order = ["Low", "Medium", "High"]
    risk_counts = df_results["risk_level"].value_counts().reindex(order, fill_value=0)
    chart_data = pd.DataFrame(
        {"Risk level": order, "Customers": risk_counts.values}
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Risk level:N", sort=order),
            y=alt.Y("Customers:Q"),
        )
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Customer Scores")

    col_filter, col_sort = st.columns(2)
    with col_filter:
        filter_risk = st.multiselect(
            "Filter by Risk",
            ['Low', 'Medium', 'High'],
            default=['Low', 'Medium', 'High']
        )
    with col_sort:
        sort_desc = st.checkbox("Sort by highest risk first", value=True)

    df_filtered = df_results[df_results['risk_level'].isin(filter_risk)].copy()
    df_sorted = df_filtered.sort_values('churn_probability', ascending=not sort_desc)

    display_cols = ['churn_probability', 'risk_level']
    if st.session_state.customer_id_col and st.session_state.customer_id_col in df_sorted.columns:
        display_cols.insert(0, st.session_state.customer_id_col)

    df_display = df_sorted[display_cols].copy()
    df_display['churn_probability'] = df_display['churn_probability'].apply(lambda x: f"{x:.4f}")

    st.dataframe(df_display, use_container_width=True, hide_index=True, height=450)
    st.caption(f"Showing {len(df_display):,} of {len(df_results):,} customers")

    st.write("")

    csv = df_results.to_csv(index=False)
    st.download_button(
        "Download Results",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )


def render_retention_actions():
    """Render the Retention Actions page."""
    st.title("Retention Actions")

    if st.session_state.df_results is None:
        st.write("No data available. Please upload a dataset first.")
        return

    df_results = st.session_state.df_results.copy()
    threshold = st.session_state.high_risk_threshold

    df_results['risk_level'] = df_results['churn_probability'].apply(
        lambda x: 'High' if x > threshold else ('Medium' if x >= 0.3 else 'Low')
    )

    customer_id_col = st.session_state.customer_id_col or "CustomerID"
    if customer_id_col not in df_results.columns:
        st.error("Customer ID column not found.")
        return

    def get_actions_for_risk(level: str):
        if level == "High":
            return [
                {
                    "key": "priority_call",
                    "name": "Priority customer success call",
                    "desc": "Escalate to a senior retention specialist.",
                    "cost": "High cost",
                    "handler": prioritize_success_call,
                },
                {
                    "key": "high_discount",
                    "name": "High-value discount",
                    "desc": "Offer a stronger incentive to prevent churn.",
                    "cost": "High cost",
                    "handler": offer_discount,
                    "discount": {"min": 15, "max": 30, "default": 20},
                },
                {
                    "key": "personalized_email",
                    "name": "Personalized retention email",
                    "desc": "Send a tailored message based on recent activity.",
                    "cost": "Medium cost",
                    "handler": send_personalized_retention_email,
                },
                {
                    "key": "monitoring",
                    "name": "Short-term monitoring",
                    "desc": "Flag account for monitoring over the next 14 days.",
                    "cost": "Medium cost",
                    "handler": flag_short_term_monitoring,
                },
            ]
        if level == "Medium":
            return [
                {
                    "key": "targeted_discount",
                    "name": "Targeted discount",
                    "desc": "Small incentive to encourage another purchase.",
                    "cost": "Medium cost",
                    "handler": offer_discount,
                    "discount": {"min": 5, "max": 10, "default": 7},
                },
                {
                    "key": "feature_email",
                    "name": "Feature education email",
                    "desc": "Highlight useful features to increase engagement.",
                    "cost": "Low cost",
                    "handler": send_feature_education_email,
                },
                {
                    "key": "usage_nudge",
                    "name": "Usage reminder / inactivity nudge",
                    "desc": "Remind customers to return and complete an order.",
                    "cost": "Low cost",
                    "handler": send_usage_reminder,
                },
                {
                    "key": "experiment",
                    "name": "Retention experiment cohort",
                    "desc": "Add to a targeted retention experiment.",
                    "cost": "Medium cost",
                    "handler": add_to_experiment_cohort,
                },
            ]
        return [
            {
                "key": "loyalty_reward",
                "name": "Loyalty reward / points",
                "desc": "Reinforce loyalty with points or credits.",
                "cost": "Low cost",
                "handler": issue_loyalty_reward,
            },
            {
                "key": "upsell",
                "name": "Upsell or cross-sell recommendation",
                "desc": "Recommend a complementary product.",
                "cost": "Medium cost",
                "handler": send_upsell_recommendation,
            },
            {
                "key": "referral",
                "name": "Referral incentive",
                "desc": "Invite customers to refer others.",
                "cost": "Medium cost",
                "handler": send_referral_incentive,
            },
            {
                "key": "feedback",
                "name": "Feedback or review request",
                "desc": "Collect feedback to improve retention.",
                "cost": "Low cost",
                "handler": request_feedback_review,
            },
        ]

    def run_action(action, customer_ids, discount_value=None):
        for customer_id in customer_ids:
            if discount_value is not None:
                action["handler"](customer_id, discount_value)
            else:
                action["handler"](customer_id)

    # Bulk actions by risk category
    st.markdown("### Recommended actions for this risk level")
    risk_category = st.selectbox(
        "Risk category",
        ["High", "Medium", "Low"],
        help="Apply actions to all customers in a selected risk category",
    )

    category_df = df_results[df_results["risk_level"] == risk_category].copy()
    st.caption(f"{len(category_df):,} customers in {risk_category.lower()} risk.")

    actions = get_actions_for_risk(risk_category)
    for action in actions:
        st.write(f"**{action['name']}** — {action['desc']}")
        st.caption(action["cost"])

        if "discount" in action:
            discount_value = st.number_input(
                "Discount %",
                min_value=action["discount"]["min"],
                max_value=action["discount"]["max"],
                value=action["discount"]["default"],
                step=1,
                key=f"bulk_{action['key']}_discount",
            )
            if st.button(
                f"Apply to all {risk_category.lower()} risk",
                key=f"bulk_{action['key']}_btn",
            ):
                run_action(action, category_df[customer_id_col].astype(str).tolist(), discount_value)
                st.success("Action applied to selected risk category.")
        else:
            if st.button(
                f"Apply to all {risk_category.lower()} risk",
                key=f"bulk_{action['key']}_btn",
            ):
                run_action(action, category_df[customer_id_col].astype(str).tolist())
                st.success("Action applied to selected risk category.")

        st.write("")

    st.markdown("### Individual customer actions")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Select Customer")
        customer_options = df_results.sort_values("churn_probability", ascending=False)[
            customer_id_col
        ].tolist()
        selected = st.selectbox(
            "Customer",
            customer_options,
            format_func=lambda x: f"{x} — {df_results[df_results[customer_id_col]==x]['churn_probability'].iloc[0]:.3f}",
        )

        if selected:
            data = df_results[df_results[customer_id_col] == selected].iloc[0]

            st.markdown("### Customer Details")
            st.metric("Churn Probability", f"{data['churn_probability']:.4f}")
            st.metric("Customer ID", str(selected))

            percentile = (df_results['churn_probability'] < data['churn_probability']).sum() / len(df_results) * 100
            st.metric("Risk Percentile", f"{percentile:.1f}%")

    with col_right:
        if selected:
            risk_level = data["risk_level"]
            st.markdown("### Recommended actions for this risk level")
            actions = get_actions_for_risk(risk_level)

            for action in actions:
                st.write(f"**{action['name']}** — {action['desc']}")
                st.caption(action["cost"])

                if "discount" in action:
                    discount_value = st.number_input(
                        "Discount %",
                        min_value=action["discount"]["min"],
                        max_value=action["discount"]["max"],
                        value=action["discount"]["default"],
                        step=1,
                        key=f"ind_{action['key']}_discount",
                    )
                    if st.button(action["name"], key=f"ind_{action['key']}_btn"):
                        run_action(action, [str(selected)], discount_value)
                        st.success("Action logged.")
                else:
                    if st.button(action["name"], key=f"ind_{action['key']}_btn"):
                        run_action(action, [str(selected)])
                        st.success("Action logged.")

                st.write("")

            st.markdown("### Action History")
            history = get_action_history(str(selected))
            if len(history) > 0:
                st.dataframe(history, use_container_width=True, hide_index=True)
            else:
                st.caption("No actions recorded for this customer.")


# Sidebar
st.sidebar.markdown("## Churn Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Upload & Overview", "Churn Scoring", "Retention Actions"],
    label_visibility="collapsed"
)

if st.session_state.df_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset loaded**")
    st.sidebar.markdown(f"{st.session_state.dataset_row_count:,} customers")

# Render page
if page == "Upload & Overview":
    render_upload_page()
elif page == "Churn Scoring":
    render_churn_dashboard()
elif page == "Retention Actions":
    render_retention_actions()
