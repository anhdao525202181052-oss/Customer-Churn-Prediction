import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Olist Customer Intelligence", page_icon="🛒", layout="wide")

FEATURES = [
    "Frequency", "Monetary", "Avg_Freight", "Avg_Items_Per_Order",
    "Avg_Products_Per_Order", "Avg_Category_Diversity",
    "Avg_Review_Score", "Avg_Installments"
]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#f7f8fa;border:1px solid #e3e7ed;border-radius:12px;padding:14px;}
.risk-high {padding:16px;border-radius:12px;background:#fff0f0;border:1px solid #ef9a9a;color:#9b1c1c;font-weight:700;}
.risk-low {padding:16px;border-radius:12px;background:#eefbf3;border:1px solid #8bd1a6;color:#146c3a;font-weight:700;}
.small-note {color:#667085;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)


def clean_cols(df):
    return df.copy()


def aggregate_olist(customers, orders, items, payments, reviews, products):
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()

    item_x = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    item_agg = item_x.groupby("order_id", as_index=False).agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        item_rows=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
        category_diversity=("product_category_name", "nunique"),
    )
    pay_agg = payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        avg_installments=("payment_installments", "mean"),
    )
    review_agg = reviews.groupby("order_id", as_index=False).agg(review_score=("review_score", "mean"))

    master = (delivered
              .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
              .merge(item_agg, on="order_id", how="left")
              .merge(pay_agg, on="order_id", how="left")
              .merge(review_agg, on="order_id", how="left"))

    reference_date = master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    customer = master.groupby("customer_unique_id", as_index=False).agg(
        Last_Purchase=("order_purchase_timestamp", "max"),
        Frequency=("order_id", "nunique"),
        Monetary=("payment_value", "sum"),
        Avg_Freight=("total_freight", "mean"),
        Avg_Items_Per_Order=("item_rows", "mean"),
        Avg_Products_Per_Order=("unique_products", "mean"),
        Avg_Category_Diversity=("category_diversity", "mean"),
        Avg_Review_Score=("review_score", "mean"),
        Avg_Installments=("avg_installments", "mean"),
    )
    customer["Recency"] = (reference_date - customer["Last_Purchase"]).dt.days
    customer["Churn_Risk"] = (customer["Recency"] > 180).astype(int)
    return customer, master, reference_date


def train_models(customer):
    X = customer[FEATURES].copy()
    y = customer["Churn_Risk"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )),
    ])
    logistic.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    metrics = []
    for name, model in [("Logistic Regression", logistic), ("XGBoost", xgb)]:
        p = model.predict(X_test)
        pr = model.predict_proba(X_test)[:, 1]
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, p),
            "Precision": precision_score(y_test, p, zero_division=0),
            "Recall": recall_score(y_test, p, zero_division=0),
            "F1": f1_score(y_test, p, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, pr),
        })
    pred = xgb.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    rfm = customer[["Recency", "Frequency", "Monetary"]].copy()
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler().fit(rfm_log)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaler.transform(rfm_log))
    customer = customer.copy()
    customer["Cluster"] = kmeans.labels_
    profile = customer.groupby("Cluster").agg(
        Customers=("customer_unique_id", "count"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    ).reset_index()
    sorted_rec = profile.sort_values("Recency")
    recent_cluster = int(sorted_rec.iloc[0]["Cluster"])
    loyal_cluster = int(profile.sort_values(["Frequency", "Monetary"], ascending=False).iloc[0]["Cluster"])
    remaining = [int(c) for c in profile.Cluster if int(c) not in {recent_cluster, loyal_cluster}]
    high_value = int(profile.loc[profile.Cluster.isin(remaining)].sort_values("Monetary", ascending=False).iloc[0]["Cluster"])
    low_value = next(c for c in remaining if c != high_value)
    segment_map = {
        recent_cluster: "Recent Customers",
        loyal_cluster: "Loyal / Repeat Customers",
        high_value: "High-Value Inactive",
        low_value: "Low-Value Inactive",
    }
    customer["Segment"] = customer["Cluster"].map(segment_map)

    artifacts = {
        "logistic": logistic, "xgb": xgb, "rfm_scaler": scaler,
        "kmeans": kmeans, "segment_map": segment_map,
        "metrics": pd.DataFrame(metrics), "cm": cm,
    }
    return customer, artifacts


def save_artifacts(customer, artifacts):
    joblib.dump(artifacts["xgb"], MODEL_DIR / "xgb_pipeline.joblib")
    joblib.dump(artifacts["logistic"], MODEL_DIR / "logistic_pipeline.joblib")
    joblib.dump(artifacts["rfm_scaler"], MODEL_DIR / "rfm_scaler.joblib")
    joblib.dump(artifacts["kmeans"], MODEL_DIR / "kmeans.joblib")
    (MODEL_DIR / "segment_map.json").write_text(json.dumps({str(k): v for k, v in artifacts["segment_map"].items()}, ensure_ascii=False), encoding="utf-8")
    customer.to_pickle(MODEL_DIR / "customer_scored.pkl")
    artifacts["metrics"].to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    np.savetxt(MODEL_DIR / "confusion_matrix.csv", artifacts["cm"], delimiter=",", fmt="%d")


def load_saved():
    required = ["xgb_pipeline.joblib", "rfm_scaler.joblib", "kmeans.joblib", "segment_map.json", "customer_scored.pkl"]
    if not all((MODEL_DIR / f).exists() for f in required):
        return None
    return {
        "xgb": joblib.load(MODEL_DIR / "xgb_pipeline.joblib"),
        "rfm_scaler": joblib.load(MODEL_DIR / "rfm_scaler.joblib"),
        "kmeans": joblib.load(MODEL_DIR / "kmeans.joblib"),
        "segment_map": {int(k): v for k, v in json.loads((MODEL_DIR / "segment_map.json").read_text(encoding="utf-8")).items()},
        "customer": pd.read_pickle(MODEL_DIR / "customer_scored.pkl"),
    }


def predict_one(values, recency, art):
    row = pd.DataFrame([values], columns=FEATURES)
    prob = float(art["xgb"].predict_proba(row)[0, 1])
    rfm = np.log1p(pd.DataFrame([[recency, values[0], values[1]]], columns=["Recency", "Frequency", "Monetary"]))
    cluster = int(art["kmeans"].predict(art["rfm_scaler"].transform(rfm))[0])
    segment = art["segment_map"].get(cluster, f"Cluster {cluster}")
    return prob, segment


st.title("🛒 Olist Customer Intelligence Demo")
st.caption("RFM + K-Means segmentation • XGBoost churn-risk scoring • Action recommendations")

setup, single, portfolio, method = st.tabs(["1. Setup model", "2. Predict customer", "3. Portfolio dashboard", "4. Method notes"])

with setup:
    st.subheader("Train the real pipeline from Olist CSV files")
    st.info("Upload the six original CSV files from Kaggle. The app aggregates data at order/customer level and trains the same model family and parameters used in the report.")
    labels = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
    }
    uploaded = {}
    cols = st.columns(2)
    for i, (key, filename) in enumerate(labels.items()):
        uploaded[key] = cols[i % 2].file_uploader(filename, type="csv", key=key)
    if st.button("Train and save demo model", type="primary", use_container_width=True):
        if not all(uploaded.values()):
            st.error("Please upload all six CSV files.")
        else:
            with st.status("Building customer dataset and training models...", expanded=True) as status:
                dfs = {k: pd.read_csv(v) for k, v in uploaded.items()}
                st.write("Aggregating orders and customer-level features...")
                customer, master, ref_date = aggregate_olist(**dfs)
                st.write(f"Delivered orders: {master.order_id.nunique():,}; customers: {customer.customer_unique_id.nunique():,}")
                customer, artifacts = train_models(customer)
                save_artifacts(customer, artifacts)
                status.update(label="Model saved. Demo is ready.", state="complete")
            st.dataframe(artifacts["metrics"].style.format({c:"{:.4f}" for c in ["Accuracy","Precision","Recall","F1","ROC_AUC"]}), use_container_width=True)
            st.success("Open tab 2 for live prediction or tab 3 for the dashboard.")

art = load_saved()

with single:
    if art is None:
        st.warning("Train the model in tab 1 first.")
    else:
        st.subheader("Single customer prediction")
        defaults = art["customer"][FEATURES + ["Recency"]].median(numeric_only=True)
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Recency (days, segmentation only)", min_value=0, value=int(defaults["Recency"]), help="Not sent to XGBoost. Used only for RFM segment assignment.")
        frequency = c2.number_input("Frequency", min_value=1, value=max(1, int(round(defaults["Frequency"]))))
        monetary = c3.number_input("Monetary", min_value=0.0, value=float(defaults["Monetary"]), step=10.0)
        a, b, c = st.columns(3)
        freight = a.number_input("Average freight", min_value=0.0, value=float(defaults["Avg_Freight"]), step=1.0)
        review = b.slider("Average review score", 1.0, 5.0, float(np.clip(defaults["Avg_Review_Score"], 1, 5)), 0.1)
        installments = c.number_input("Average installments", min_value=0.0, value=float(defaults["Avg_Installments"]), step=0.5)
        d, e, f = st.columns(3)
        items = d.number_input("Average item rows/order", min_value=1.0, value=float(defaults["Avg_Items_Per_Order"]), step=0.1)
        products = e.number_input("Average products/order", min_value=1.0, value=float(defaults["Avg_Products_Per_Order"]), step=0.1)
        diversity = f.number_input("Average category diversity", min_value=0.0, value=float(defaults["Avg_Category_Diversity"]), step=0.1)

        values = [frequency, monetary, freight, items, products, diversity, review, installments]
        threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, 0.05)
        if st.button("Predict churn risk", type="primary", use_container_width=True):
            prob, segment = predict_one(values, recency, art)
            high = prob >= threshold
            x1, x2, x3 = st.columns(3)
            x1.metric("Churn-risk probability", f"{prob:.1%}")
            x2.metric("Risk label", "HIGH RISK" if high else "LOW RISK")
            x3.metric("Customer segment", segment)
            st.progress(min(max(prob, 0.0), 1.0))
            if high:
                st.markdown('<div class="risk-high">Priority recommendation: include this customer in a controlled retention experiment.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">Recommendation: standard nurturing; avoid expensive blanket discounts.</div>', unsafe_allow_html=True)
            if segment == "High-Value Inactive":
                action = "Personalized win-back, service recovery check, and a capped shipping or product incentive."
            elif segment == "Low-Value Inactive":
                action = "Low-cost automated remarketing and a minimum-spend voucher."
            elif segment == "Recent Customers":
                action = "Second-purchase journey: post-purchase content and time-limited cross-sell."
            else:
                action = "Loyalty treatment, cross-sell, referral and experience protection."
            st.write("**Suggested business action:**", action)
            st.caption("Important: Avg_Freight may act as a temporal proxy. Treat the prediction as a screening signal, not causal proof.")

with portfolio:
    if art is None:
        st.warning("Train the model in tab 1 first.")
    else:
        st.subheader("Customer portfolio dashboard")
        scored = art["customer"].copy()
        scored["Risk_Probability"] = art["xgb"].predict_proba(scored[FEATURES])[:, 1]
        scored["Predicted_Risk"] = np.where(scored["Risk_Probability"] >= 0.5, "High", "Low")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers", f"{len(scored):,}")
        m2.metric("Predicted high risk", f"{(scored.Predicted_Risk=='High').mean():.1%}")
        m3.metric("Average probability", f"{scored.Risk_Probability.mean():.1%}")
        m4.metric("Segments", scored.Segment.nunique())
        left, right = st.columns(2)
        seg = scored.groupby("Segment", as_index=False).agg(Customers=("customer_unique_id","count"), Avg_Risk=("Risk_Probability","mean"))
        left.plotly_chart(px.bar(seg, x="Segment", y="Customers", color="Avg_Risk", color_continuous_scale="Reds", title="Customers and average risk by segment"), use_container_width=True)
        right.plotly_chart(px.histogram(scored, x="Risk_Probability", nbins=30, color="Predicted_Risk", title="Risk probability distribution"), use_container_width=True)
        preview = scored[["customer_unique_id", "Segment", "Risk_Probability", "Predicted_Risk", "Frequency", "Monetary", "Avg_Freight"]].sort_values("Risk_Probability", ascending=False)
        st.dataframe(preview.head(500), use_container_width=True, hide_index=True)
        st.download_button("Download scored customer list", preview.to_csv(index=False).encode("utf-8"), "olist_customer_risk_scores.csv", "text/csv", use_container_width=True)

with method:
    st.subheader("Method alignment with the report")
    st.markdown("""
- Only `delivered` orders are used.
- Items, payments and reviews are aggregated by `order_id` before joining.
- RFM segmentation uses log1p + StandardScaler + K-Means (K=4).
- `Churn_Risk = 1` when `Recency > 180 days`.
- Recency, Cluster and Segment are not input features for XGBoost.
- Train/test split: 80/20, stratified, random_state=42.
- XGBoost: 300 trees, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8.
- The app is for educational demonstration; out-of-time validation and controlled A/B testing are required before real deployment.
""")
