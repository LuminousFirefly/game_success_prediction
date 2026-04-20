import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import NUMERIC_UI, RATING_5_COLOR, RATING_COLOR, UI_TAGS, TargetEncoderCV

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Game Success Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_READY = (MODELS_DIR / "clf3_pipeline.joblib").exists()

# ── Loaders (cached) ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf  = joblib.load(MODELS_DIR / "clf3_pipeline.joblib")
    reg  = joblib.load(MODELS_DIR / "reg_pipeline.joblib")
    le3  = joblib.load(MODELS_DIR / "le3.joblib")
    with open(MODELS_DIR / "feature_cols.json")    as f: feat_cols = json.load(f)
    with open(MODELS_DIR / "feature_defaults.json") as f: defaults  = json.load(f)
    return clf, reg, le3, feat_cols, defaults

@st.cache_data
def load_eda():
    with open(MODELS_DIR / "eda_data.json") as f:
        return json.load(f)

@st.cache_data
def load_eval():
    with open(MODELS_DIR / "eval_data.json") as f:
        return json.load(f)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://store.steampowered.com/favicon.ico", width=32)
    st.title("🎮 Steam Success")
    st.caption("CSC 240 · Game Success Prediction")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Dataset Explorer", "📈 Model Evaluation", "🔮 Single Game Predictor", "📂 Batch Predictor"],
        label_visibility="collapsed",
    )
    st.divider()
    if MODELS_READY:
        st.success("Models loaded ✓", icon="✅")
    else:
        st.warning("Run `train_models.py` first", icon="⚠️")
    st.caption("Best CV accuracy (3-class): **~68%**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.caption("Steam store data · ~27,000 games · 499 raw features")

    if not MODELS_READY:
        st.info("Run `python streamlit_app/train_models.py` to generate model files, then restart.")
        st.stop()

    eda = load_eda()

    # ── Top metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Games",    "26,821")
    c2.metric("Raw Features",   "499")
    c3.metric("Model Features", "40")
    c4.metric("Best Accuracy",  "~68%", "3-class")

    st.divider()

    # ── Row 1: class distributions ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("5-Class Rating Distribution")
        d5 = eda["dist_5class"]
        order = ["Positive", "Mostly Positive", "Mixed", "Mostly Negative", "Negative"]
        fig = px.bar(
            x=[k for k in order if k in d5],
            y=[d5[k] for k in order if k in d5],
            color=[k for k in order if k in d5],
            color_discrete_map=RATING_5_COLOR,
            labels={"x": "Rating", "y": "Games"},
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("3-Class Rating Distribution")
        d3 = eda["dist_3class"]
        order3 = ["Good", "Mixed", "Bad"]
        fig = px.bar(
            x=[k for k in order3 if k in d3],
            y=[d3[k] for k in order3 if k in d3],
            color=[k for k in order3 if k in d3],
            color_discrete_map=RATING_COLOR,
            labels={"x": "Rating", "y": "Games"},
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Wilson score + Price distributions ──
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Wilson Score Distribution")
        h = eda["wilson_hist"]
        midpoints = [(h["edges"][i] + h["edges"][i+1]) / 2 for i in range(len(h["counts"]))]
        fig = px.bar(x=midpoints, y=h["counts"],
                     labels={"x": "Wilson Score (0–1)", "y": "Games"},
                     color_discrete_sequence=["#66c0f4"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Price Distribution (capped at $60)")
        hp = eda["price_hist"]
        mids = [(hp["edges"][i] + hp["edges"][i+1]) / 2 for i in range(len(hp["counts"]))]
        fig = px.bar(x=mids, y=hp["counts"],
                     labels={"x": "Price (USD)", "y": "Games"},
                     color_discrete_sequence=["#c6d4df"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Feature importances + Tag frequency ──
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Top 30 Feature Importances")
        fi = pd.DataFrame(eda["feature_importances"])
        fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale="Blues",
                     labels={"importance": "Importance", "feature": ""})
        fig.update_layout(height=550, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Most Common Tags")
        tf = eda["tag_frequency"]
        tf_df = pd.DataFrame({"tag": list(tf.keys()), "count": list(tf.values())})
        fig = px.bar(tf_df.sort_values("count"), x="count", y="tag",
                     orientation="h", color="count",
                     color_continuous_scale="Teal",
                     labels={"count": "# Games", "tag": ""})
        fig.update_layout(height=550, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Scatter ──
    st.subheader("Owners (log scale) vs Wilson Score")
    scatter_df = pd.DataFrame(eda["scatter_sample"])
    has_color = "rating_category" in scatter_df.columns
    fig = px.scatter(
        scatter_df, x="owners_log", y="wilson_score",
        color="rating_category" if has_color else None,
        color_discrete_map=RATING_5_COLOR if has_color else None,
        opacity=0.6,
        labels={"owners_log": "log(Owners)", "wilson_score": "Wilson Score"},
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Game Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Game Predictor":
    st.title("🔮 Single Game Predictor")
    st.caption("Enter game attributes to predict rating category and Wilson score.")

    if not MODELS_READY:
        st.info("Run `python streamlit_app/train_models.py` first, then restart.")
        st.stop()

    clf, reg, le3, feat_cols, defaults = load_models()

    # ── Input form ──────────────────────────────────────────────────────────
    with st.form("predictor_form"):
        st.subheader("Game Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Pricing & Scale**")
            price        = st.slider("Price (USD)",       *NUMERIC_UI["price"][:3],       step=NUMERIC_UI["price"][3])
            achievements = st.slider("Achievements",      *NUMERIC_UI["achievements"][:3], step=NUMERIC_UI["achievements"][3])
            owners_log   = st.slider("Owners log₁₀",      *NUMERIC_UI["owners_log"][:3],  step=NUMERIC_UI["owners_log"][3],
                                     help="log10 of estimated owner count. 10 ≈ 10k owners, 14 ≈ 1M owners.")

        with c2:
            st.markdown("**Release & Platform**")
            release_year  = st.slider("Release Year",    *NUMERIC_UI["release_year"][:3],  step=NUMERIC_UI["release_year"][3])
            release_month = st.slider("Release Month",   *NUMERIC_UI["release_month"][:3], step=NUMERIC_UI["release_month"][3])
            mac           = st.checkbox("macOS support",   value=False)
            mac_sup       = st.checkbox("Mac featured",    value=False)

        with c3:
            st.markdown("**Tech Specs**")
            processor_Ghz = st.slider("CPU (GHz)",       *NUMERIC_UI["processor_Ghz"][:3], step=NUMERIC_UI["processor_Ghz"][3])
            RAM_mb        = st.slider("RAM (MB)",         *NUMERIC_UI["RAM_mb"][:3],        step=NUMERIC_UI["RAM_mb"][3])
            GPU_mb        = st.slider("GPU VRAM (MB)",    *NUMERIC_UI["GPU_mb"][:3],        step=NUMERIC_UI["GPU_mb"][3])
            storage_mb    = st.slider("Storage (MB)",     *NUMERIC_UI["storage_mb"][:3],    step=NUMERIC_UI["storage_mb"][3])

        st.markdown("**Playtime (minutes)**")
        pt1, pt2 = st.columns(2)
        avg_playtime = pt1.slider("Avg Playtime", *NUMERIC_UI["average_playtime"][:3], step=NUMERIC_UI["average_playtime"][3])
        med_playtime = pt2.slider("Median Playtime", *NUMERIC_UI["median_playtime"][:3], step=NUMERIC_UI["median_playtime"][3])

        st.markdown("**Genre Tags**")
        selected_tags = st.multiselect(
            "Select all that apply",
            options=UI_TAGS,
            default=["indie", "singleplayer"],
        )

        st.markdown("**Steam Features**")
        sf1, sf2, sf3 = st.columns(3)
        steam_cloud    = sf1.checkbox("Steam Cloud",            value=True)
        trading_cards  = sf2.checkbox("Steam Trading Cards",    value=False)
        full_ctrl      = sf3.checkbox("Full Controller Support", value=False)
        partial_ctrl   = sf1.checkbox("Partial Controller Support", value=False)
        early_access   = sf2.checkbox("Early Access",           value=False)

        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    # ── Prediction ──────────────────────────────────────────────────────────
    if submitted:
        row = {col: defaults[col] for col in feat_cols}

        # Numeric overrides
        row.update({
            "price": price, "achievements": achievements, "owners_log": owners_log,
            "release_year": release_year, "release_month": release_month,
            "processor_Ghz": processor_Ghz, "RAM_mb": RAM_mb,
            "GPU_mb": GPU_mb, "storage_mb": storage_mb,
            "average_playtime": avg_playtime, "median_playtime": med_playtime,
            "mac": int(mac), "mac_sup": int(mac_sup),
        })
        # Tag overrides
        for tag in UI_TAGS:
            if tag in row:
                row[tag] = 1 if tag in selected_tags else 0
        # Steam feature overrides
        for col, val in [
            ("Steam Cloud_cat",                steam_cloud),
            ("Steam Trading Cards_cat",         trading_cards),
            ("Full controller support_cat",     full_ctrl),
            ("Partial Controller Support_cat",  partial_ctrl),
            ("early_access",                    early_access),
        ]:
            if col in row:
                row[col] = int(val)

        X_input = pd.DataFrame([row])[feat_cols]

        clf_label  = le3.inverse_transform(clf.predict(X_input))[0]
        clf_proba  = clf.predict_proba(X_input)[0]
        wilson_hat = float(np.clip(reg.predict(X_input)[0], 0, 1))

        st.divider()
        st.subheader("Results")
        rc1, rc2, rc3 = st.columns(3)

        color = RATING_COLOR.get(clf_label, "#c6d4df")
        rc1.markdown(
            f"<div style='background:{color}22;border:2px solid {color};"
            f"border-radius:12px;padding:20px;text-align:center'>"
            f"<p style='font-size:0.9rem;margin:0;color:{color}'>Predicted Rating</p>"
            f"<p style='font-size:2.2rem;font-weight:700;margin:0;color:{color}'>{clf_label}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        rc2.metric("Wilson Score", f"{wilson_hat:.3f}", help="0 = very negative · 1 = overwhelmingly positive")
        rc3.metric("Confidence", f"{max(clf_proba)*100:.1f}%")

        # Probability breakdown
        st.markdown("**Class Probabilities**")
        prob_df = pd.DataFrame({
            "Class": le3.classes_,
            "Probability": clf_proba,
        }).sort_values("Probability", ascending=True)
        fig = px.bar(prob_df, x="Probability", y="Class", orientation="h",
                     color="Class", color_discrete_map=RATING_COLOR,
                     range_x=[0, 1])
        fig.update_layout(showlegend=False, height=200,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.caption("Confusion matrices, ROC curves, and regression diagnostics — evaluated on a held-out 20% test set.")

    if not MODELS_READY or not (MODELS_DIR / "eval_data.json").exists():
        st.info("Run `python streamlit_app/train_models.py` first, then restart.")
        st.stop()

    ev = load_eval()

    tab_clf3, tab_clf5, tab_reg = st.tabs(["3-Class Classification", "5-Class Classification", "Regression"])

    # ── helpers ──
    BG = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

    def confusion_heatmap(cm, classes, title):
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=classes, y=classes,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title=title,
        )
        fig.update_layout(**BG, margin=dict(l=0, r=0, t=40, b=0),
                          coloraxis_showscale=False)
        return fig

    def roc_figure(roc_data, title):
        fig = go.Figure()
        colors = ["#66c0f4", "#4ade80", "#facc15", "#fb923c", "#f87171"]
        for i, (cls, d) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"], mode="lines",
                name=f"{cls} (AUC={d['auc']:.2f})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="#666"))
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.05), **BG,
        )
        return fig

    # ── 3-Class tab ──
    with tab_clf3:
        models3 = list(ev["clf3"].keys())
        sel3 = st.selectbox("Model", models3, key="sel3")
        d = ev["clf3"][sel3]

        m1, m2 = st.columns(2)
        m1.metric("Test Accuracy", f"{d['test_accuracy']*100:.1f}%")

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.plotly_chart(
                confusion_heatmap(d["confusion_matrix"], d["classes"],
                                  f"Confusion Matrix — {sel3}"),
                use_container_width=True,
            )
        with col_roc:
            st.plotly_chart(
                roc_figure(d["roc"], f"ROC Curves — {sel3}"),
                use_container_width=True,
            )

    # ── 5-Class tab ──
    with tab_clf5:
        models5 = list(ev["clf5"].keys())
        sel5 = st.selectbox("Model", models5, key="sel5")
        d = ev["clf5"][sel5]

        st.metric("Test Accuracy", f"{d['test_accuracy']*100:.1f}%")

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.plotly_chart(
                confusion_heatmap(d["confusion_matrix"], d["classes"],
                                  f"Confusion Matrix — {sel5}"),
                use_container_width=True,
            )
        with col_roc:
            st.plotly_chart(
                roc_figure(d["roc"], f"ROC Curves — {sel5}"),
                use_container_width=True,
            )

    # ── Regression tab ──
    with tab_reg:
        models_r = list(ev["reg"].keys())
        sel_r = st.selectbox("Model", models_r, key="sel_r")
        d = ev["reg"][sel_r]

        rm1, rm2 = st.columns(2)
        rm1.metric("Test R²",   f"{d['r2']:.4f}")
        rm2.metric("Test RMSE", f"{d['rmse']:.4f}")

        col_sc, col_res = st.columns(2)

        with col_sc:
            sc = d["scatter"]
            fig = px.scatter(
                x=sc["actual"], y=sc["predicted"],
                labels={"x": "Actual Wilson Score", "y": "Predicted Wilson Score"},
                title=f"Actual vs Predicted — {sel_r}",
                opacity=0.4, color_discrete_sequence=["#66c0f4"],
            )
            mn = min(min(sc["actual"]), min(sc["predicted"]))
            mx = max(max(sc["actual"]), max(sc["predicted"]))
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(dash="dash", color="#f87171", width=2))
            fig.update_layout(**BG)
            st.plotly_chart(fig, use_container_width=True)

        with col_res:
            res = d["residuals"]
            fig = px.scatter(
                x=res["predicted"], y=res["residuals"],
                labels={"x": "Predicted Wilson Score", "y": "Residual (Actual − Predicted)"},
                title=f"Residual Plot — {sel_r}",
                opacity=0.4, color_discrete_sequence=["#66c0f4"],
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#f87171", line_width=2)
            fig.update_layout(**BG)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batch Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Predictor":
    st.title("📂 Batch Predictor")
    st.caption("Upload a CSV of games and get predictions on every row.")

    if not MODELS_READY:
        st.info("Run `python streamlit_app/train_models.py` first, then restart.")
        st.stop()

    clf, reg, le3, feat_cols, defaults = load_models()

    st.info(
        "Your CSV should contain any subset of the model's feature columns. "
        "Missing columns are filled with training-set medians automatically.",
        icon="ℹ️",
    )

    with st.expander("Expected columns (click to expand)"):
        st.code(", ".join(feat_cols))

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.subheader(f"Preview — {len(raw):,} rows")
        st.dataframe(raw.head(10), use_container_width=True)

        if st.button("Run Predictions", type="primary"):
            with st.spinner("Predicting..."):
                # Fill in all required columns, use defaults for anything missing
                X_batch = pd.DataFrame(
                    {col: raw[col] if col in raw.columns else defaults[col]
                     for col in feat_cols}
                )

                raw["predicted_rating"]       = le3.inverse_transform(clf.predict(X_batch))
                raw["predicted_wilson_score"] = np.clip(reg.predict(X_batch), 0, 1).round(4)
                proba = clf.predict_proba(X_batch)
                for i, cls in enumerate(le3.classes_):
                    raw[f"prob_{cls}"] = proba[:, i].round(4)

            st.subheader("Results")

            # Summary metrics
            dist = raw["predicted_rating"].value_counts()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Good",  dist.get("Good",  0))
            mc2.metric("Mixed", dist.get("Mixed", 0))
            mc3.metric("Bad",   dist.get("Bad",   0))

            # Distribution chart
            fig = px.pie(
                names=dist.index, values=dist.values,
                color=dist.index, color_discrete_map=RATING_COLOR,
                hole=0.4,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                              margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Full results table
            st.dataframe(raw, use_container_width=True)

            # Download
            csv_out = raw.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
