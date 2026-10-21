
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

st.set_page_config(
    page_title="Data Drift Early Warning System",
    page_icon="📊",
    layout="wide"
)

# ---------------- CSS: SUNSET MAUVE THEME ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #f8edf1 0%, #e8cfdb 48%, #f9e8df 100%
    );
    color: #382334;
}

.block-container {
    max-width: 100%;
    padding: 2rem 2rem 3rem;
}

h1, h2, h3 {
    color: #382334 !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
}

.main-title {
    font-size: clamp(28px, 3vw, 40px);
    font-weight: 850;
    line-height: 1.3;
    color: #382334;
    margin-bottom: 8px;
    overflow-wrap: anywhere;
}

.subtitle {
    color: #79566b;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 16px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #382334;
    margin-top: 22px;
    margin-bottom: 12px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #25121f 0%, #493044 58%, #79576d 100%
    );
}

[data-testid="stSidebar"] * {
    color: #fff7f4;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 10px;
    margin-bottom: 5px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255, 177, 145, 0.20);
}

.online {
    background: #d9f5df;
    color: #176b42;
    padding: 12px 14px;
    border-radius: 16px;
    font-weight: 750;
    text-align: center;
    margin-top: 8px;
}

/* Dashboard cards */
.card {
    background: linear-gradient(145deg, #fffaf6, #ffe6d7);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #efd0c5;
    box-shadow: 0 7px 20px rgba(73, 48, 62, 0.10);
    min-height: 125px;
    height: 100%;
}

.card-title {
    color: #79566b;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 12px;
}

.card-value {
    color: #382334;
    font-size: 30px;
    font-weight: 850;
    overflow-wrap: anywhere;
}

.workflow-card {
    background: linear-gradient(145deg, #fffaf6, #f4dce4);
    padding: 20px;
    border-radius: 17px;
    border: 1px solid #e6c2d0;
    box-shadow: 0 6px 18px rgba(73, 48, 62, 0.09);
    min-height: 135px;
    height: 100%;
    color: #493044;
}

.workflow-card h3 {
    margin-top: 0;
    font-size: 19px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #fffaf6, #ffe6d7);
    padding: 18px;
    border: 1px solid #efd0c5;
    border-radius: 17px;
    box-shadow: 0 6px 18px rgba(73, 48, 62, 0.09);
}

[data-testid="stMetricLabel"] {
    color: #79566b !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #382334 !important;
    font-weight: 850 !important;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #d98a78, #a95f7a);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 750;
    min-height: 42px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #79516b;
    color: white !important;
    transform: translateY(-1px);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #fffaf7;
    border: 1px solid #e7cbd5;
    border-radius: 14px;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 250, 247, 0.72);
    border: 1px dashed #c48ba0;
    border-radius: 14px;
    padding: 12px;
}

hr {
    border-color: #d8b6c6;
}

.devops-box {
    background: rgba(255, 255, 255, 0.09);
    padding: 16px;
    border-radius: 13px;
    margin-top: 10px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DRIFT FUNCTIONS ----------------

def drift_status(score):
    if score < 0.30:
        return "🟢 Normal"
    elif score < 0.60:
        return "🟡 Warning"
    return "🔴 Drift"


def numeric_drift(old, new):
    old = pd.to_numeric(old, errors="coerce").dropna()
    new = pd.to_numeric(new, errors="coerce").dropna()

    if old.empty or new.empty:
        return 0.0

    score, _ = ks_2samp(old, new)
    return round(float(score), 3)


def categorical_drift(old, new):
    old = old.astype("string").fillna("<MISSING>")
    new = new.astype("string").fillna("<MISSING>")

    old_dist = old.value_counts(normalize=True)
    new_dist = new.value_counts(normalize=True)

    categories = set(old_dist.index).union(new_dist.index)

    score = sum(
        abs(old_dist.get(category, 0) - new_dist.get(category, 0))
        for category in categories
    )

    return round(float(score / 2), 3)


def calculate_drift(train, new):
    rows = []

    common_columns = [
        col for col in train.columns
        if col in new.columns
    ]

    for col in common_columns:
        numeric = (
            pd.api.types.is_numeric_dtype(train[col])
            and pd.api.types.is_numeric_dtype(new[col])
        )

        if numeric:
            score = numeric_drift(train[col], new[col])
            kind = "Numerical"
        else:
            score = categorical_drift(train[col], new[col])
            kind = "Categorical"

        rows.append({
            "Feature": col,
            "Type": kind,
            "Drift Score": score,
            "Status": drift_status(score)
        })

    return pd.DataFrame(
        rows,
        columns=["Feature", "Type", "Drift Score", "Status"]
    )


def correct_dataset(
    train,
    new,
    selected_columns,
    numeric_method,
    fill_categorical
):
    corrected = new.copy(deep=True)
    changes = []

    for col in selected_columns:
        if col not in train.columns or col not in corrected.columns:
            continue

        before_result = calculate_drift(
            train[[col]], new[[col]]
        )

        before_score = (
            float(before_result["Drift Score"].iloc[0])
            if not before_result.empty else 0.0
        )

        old_missing = int(corrected[col].isna().sum())
        changed = False
        action = "No change applied"

        is_numeric = (
            pd.api.types.is_numeric_dtype(train[col])
            and pd.api.types.is_numeric_dtype(new[col])
        )

        if is_numeric:
            reference = pd.to_numeric(
                train[col], errors="coerce"
            ).dropna().to_numpy()

            values = pd.to_numeric(
                corrected[col], errors="coerce"
            )

            if len(reference):
                median = float(np.median(reference))

                if numeric_method == "Fill missing values only":
                    corrected[col] = values.fillna(median)
                    changed = old_missing > 0
                    action = (
                        f"Filled missing values with training median ({median:.4g})"
                        if changed else "No missing values to fill"
                    )

                elif numeric_method == "Align numeric distribution":
                    values = values.fillna(median)
                    ranks = values.rank(method="average", pct=True)

                    sorted_reference = np.sort(reference)
                    positions = np.clip(
                        np.rint(
                            ranks.to_numpy() * (len(sorted_reference) - 1)
                        ).astype(int),
                        0,
                        len(sorted_reference) - 1
                    )

                    corrected[col] = sorted_reference[positions]
                    changed = True
                    action = (
                        "Aligned numeric values to training quantiles; "
                        "filled missing values with training median"
                    )

                else:
                    action = "Numeric correction disabled"

        else:
            if fill_categorical and corrected[col].isna().any():
                modes = train[col].dropna().mode()
                fill_value = (
                    modes.iloc[0] if not modes.empty else "Unknown"
                )

                corrected[col] = corrected[col].fillna(fill_value)
                changed = old_missing > 0
                action = (
                    f"Filled missing values with training mode ({fill_value})"
                    if changed else "No missing values to fill"
                )
            else:
                action = (
                    "Categorical values left unchanged "
                    "to preserve their original meaning"
                )

        new_missing = int(corrected[col].isna().sum())
        changed_count = int(
            (new[col].astype(str) != corrected[col].astype(str)).sum()
        )

        after_result = calculate_drift(
            train[[col]], corrected[[col]]
        )

        after_score = (
            float(after_result["Drift Score"].iloc[0])
            if not after_result.empty else 0.0
        )

        changes.append({
            "Feature": col,
            "Type": "Numerical" if is_numeric else "Categorical",
            "Drift Before": before_score,
            "Drift After": after_score,
            "Missing Before": old_missing,
            "Missing After": new_missing,
            "Values Changed": changed_count,
            "Correction Applied": action
        })

    return corrected, pd.DataFrame(changes)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        "<h2>📊 Data Drift</h2>",
        unsafe_allow_html=True
    )
    st.markdown("Early Warning System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📤 Upload & Analyze",
            "📈 Feature Analysis",
            "🛠️ Drift Correction",
            "ℹ️ About"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚙️ DevOps Pipeline")

    st.markdown("""
    <div class="devops-box">
    🔧 Git<br>
    🐙 GitHub<br>
    🔄 Jenkins<br>
    🐳 Docker<br>
    ☸️ Kubernetes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Streamlit • Python • Pandas • SciPy")


# ---------------- COMMON HEADER ----------------

if page != "ℹ️ About":
    header_col, status_col = st.columns([5, 1])

    with header_col:
        st.markdown(
            '<div class="main-title">Data Drift Early Warning System</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="subtitle">'
            'Monitor changes in incoming data before they affect ML model performance.'
            '</div>',
            unsafe_allow_html=True
        )

    with status_col:
        st.markdown(
            '<div class="online">● System Online</div>',
            unsafe_allow_html=True
        )


# ---------------- UPLOAD AND ANALYZE ----------------

if page == "📤 Upload & Analyze":
    st.markdown(
        '<div class="section-title">📤 Upload & Analyze Data</div>',
        unsafe_allow_html=True
    )
    st.write("Upload your training CSV and new incoming CSV.")

    left, right = st.columns(2)

    with left:
        st.markdown("### 📁 Training Data")
        train_file = st.file_uploader(
            "Upload training CSV",
            type=["csv"],
            key="train_upload"
        )

    with right:
        st.markdown("### 📥 New Data")
        new_file = st.file_uploader(
            "Upload new CSV",
            type=["csv"],
            key="new_upload"
        )

    if train_file is not None and new_file is not None:
        try:
            train_data = pd.read_csv(train_file)
            new_data = pd.read_csv(new_file)

            st.session_state["train_data"] = train_data
            st.session_state["new_data"] = new_data

            st.success("Both datasets uploaded successfully!")

            p1, p2 = st.columns(2)

            with p1:
                st.markdown("### Training Data Preview")
                st.dataframe(train_data.head(), use_container_width=True)

            with p2:
                st.markdown("### New Data Preview")
                st.dataframe(new_data.head(), use_container_width=True)

            if st.button(
                "🔍 Analyze Data Drift",
                type="primary",
                use_container_width=True
            ):
                st.session_state["result"] = calculate_drift(
                    train_data, new_data
                )
                st.success("Drift analysis completed! Open Dashboard.")

        except Exception as exc:
            st.error(f"Could not read the CSV files: {exc}")

    else:
        st.info("Upload both CSV files to start the analysis.")


# ---------------- DASHBOARD ----------------

elif page == "🏠 Dashboard":
    if (
        "train_data" not in st.session_state
        or "new_data" not in st.session_state
    ):
        st.info("👈 Go to Upload & Analyze and upload your two CSV files.")

        st.markdown(
            '<div class="section-title">📌 How the System Works</div>',
            unsafe_allow_html=True
        )

        workflow = st.columns(4, gap="medium")

        cards = [
            ("1️⃣ Upload", "Training and incoming data"),
            ("2️⃣ Compare", "Compare common features"),
            ("3️⃣ Detect", "Calculate feature drift scores"),
            ("4️⃣ Alert", "Show warnings and recommendations")
        ]

        for column, (title, description) in zip(workflow, cards):
            with column:
                st.markdown(
                    f'<div class="workflow-card">'
                    f'<h3>{title}</h3>'
                    f'<p>{description}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    else:
        train_data = st.session_state["train_data"]
        new_data = st.session_state["new_data"]

        result = calculate_drift(train_data, new_data)
        st.session_state["result"] = result

        overall = (
            float(result["Drift Score"].max())
            if not result.empty else 0.0
        )

        st.markdown(
            '<div class="section-title">📊 System Overview</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4, gap="medium")

        metrics = [
            ("📁 Training Records", f"{len(train_data):,}"),
            ("📥 New Records", f"{len(new_data):,}"),
            ("🔎 Features Checked", f"{len(result):,}"),
            ("⚠️ Maximum Drift", f"{overall:.3f}")
        ]

        for column, (label, value) in zip([c1, c2, c3, c4], metrics):
            with column:
                st.markdown(
                    f'<div class="card">'
                    f'<div class="card-title">{label}</div>'
                    f'<div class="card-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.write("")

        if result.empty:
            st.warning("No common features found between the datasets.")

        elif overall >= 0.60:
            st.error(
                "🚨 DATA DRIFT DETECTED — Significant changes were found."
            )

        elif overall >= 0.30:
            st.warning(
                "⚠️ WARNING — Some features show noticeable changes."
            )

        else:
            st.success(
                "✅ DATA NORMAL — Measured drift scores are below the warning threshold."
            )

        st.markdown(
            '<div class="section-title">📊 Feature Analysis Results</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        if not result.empty:
            st.markdown(
                '<div class="section-title">📈 Feature Drift Comparison</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            result.set_index("Feature")["Drift Score"].plot(
                kind="bar", ax=ax
            )

            ax.axhline(0.30, linestyle="--", label="Warning 0.30")
            ax.axhline(0.60, linestyle="--", label="Drift 0.60")
            ax.set_ylabel("Drift Score")
            ax.set_ylim(0, 1)
            ax.legend()

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            '<div class="section-title">🎯 Drift Thresholds</div>',
            unsafe_allow_html=True
        )

        t1, t2, t3 = st.columns(3)

        with t1:
            st.success("🟢 0.00–0.29\n\nNormal")

        with t2:
            st.warning("🟡 0.30–0.59\n\nWarning")

        with t3:
            st.error("🔴 0.60–1.00\n\nDrift")

        st.markdown("### 💡 Recommendations")

        if overall >= 0.60:
            st.error(
                "Investigate affected features and validate the incoming data. "
                "Consider checking model performance."
            )
        elif overall >= 0.30:
            st.warning(
                "Monitor the affected features and investigate the source of change."
            )
        else:
            st.success(
                "Continue monitoring incoming data for future distribution changes."
            )


# ---------------- FEATURE ANALYSIS ----------------

elif page == "📈 Feature Analysis":
    st.markdown(
        '<div class="main-title">📈 Feature Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Explore the training and incoming distribution for each feature.</div>',
        unsafe_allow_html=True
    )

    if (
        "train_data" not in st.session_state
        or "new_data" not in st.session_state
    ):
        st.info("Upload both datasets first from Upload & Analyze.")

    else:
        train_data = st.session_state["train_data"]
        new_data = st.session_state["new_data"]
        result = calculate_drift(train_data, new_data)

        if result.empty:
            st.warning("No common features are available for analysis.")

        else:
            feature = st.selectbox(
                "Select a feature",
                result["Feature"].tolist()
            )

            score_row = result[result["Feature"] == feature]
            score = float(score_row["Drift Score"].iloc[0])

            st.markdown(
                f"### {feature} — Drift Score: {score:.3f}"
            )
            st.caption(f"Status: {drift_status(score)}")

            old = train_data[feature]
            new = new_data[feature]

            numeric = (
                pd.api.types.is_numeric_dtype(old)
                and pd.api.types.is_numeric_dtype(new)
            )

            if numeric:
                fig, ax = plt.subplots(figsize=(10, 5))

                ax.hist(
                    pd.to_numeric(old, errors="coerce").dropna(),
                    bins=20,
                    alpha=0.65,
                    label="Training Data"
                )
                ax.hist(
                    pd.to_numeric(new, errors="coerce").dropna(),
                    bins=20,
                    alpha=0.65,
                    label="New Data"
                )

                ax.set_xlabel(feature)
                ax.set_ylabel("Frequency")
                ax.set_title("Training vs New Data")
                ax.legend()
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:
                old_dist = (
                    old.astype("string")
                    .fillna("<MISSING>")
                    .value_counts(normalize=True)
                    .rename("Training")
                )
                new_dist = (
                    new.astype("string")
                    .fillna("<MISSING>")
                    .value_counts(normalize=True)
                    .rename("New Data")
                )

                compare = pd.concat(
                    [old_dist, new_dist], axis=1
                ).fillna(0)

                st.bar_chart(compare)


# ---------------- DRIFT CORRECTION ----------------

elif page == "🛠️ Drift Correction":
    st.markdown(
        '<div class="main-title">🛠️ Automatic Drift Correction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Generate a corrected copy, review the changes, and download the CSV.</div>',
        unsafe_allow_html=True
    )

    if (
        "train_data" not in st.session_state
        or "new_data" not in st.session_state
    ):
        st.info("Upload the training CSV and new CSV first.")

    else:
        train_data = st.session_state["train_data"].copy()
        new_data = st.session_state["new_data"].copy()
        common = [
            col for col in train_data.columns
            if col in new_data.columns
        ]

        if not common:
            st.error("The datasets have no common columns.")

        else:
            st.warning(
                "Correction modifies a copy of incoming data. "
                "A lower drift score does not prove that the original cause "
                "of drift is fixed or that model accuracy improves. "
                "Exclude target, ID, timestamp, and sensitive columns where appropriate."
            )

            before_result = calculate_drift(train_data, new_data)

            st.markdown("### 1. Drift Before Correction")
            st.dataframe(
                before_result,
                use_container_width=True,
                hide_index=True
            )

            exclude_text = st.text_input(
                "Columns to exclude (comma-separated)",
                placeholder="target, id, timestamp"
            )

            excluded = {
                x.strip() for x in exclude_text.split(",") if x.strip()
            }

            candidates = [col for col in common if col not in excluded]

            selected = st.multiselect(
                "Select features to process",
                options=candidates,
                default=[
                    row["Feature"]
                    for _, row in before_result.iterrows()
                    if row["Drift Score"] >= 0.30
                    and row["Feature"] in candidates
                    and row["Type"] == "Numerical"
                ]
            )

            numeric_method = st.selectbox(
                "Numeric feature correction",
                [
                    "Align numeric distribution",
                    "Fill missing values only",
                    "Do not modify numeric features"
                ]
            )

            fill_categorical = st.checkbox(
                "Fill missing categorical values using training mode",
                value=True
            )

            if st.button(
                "🛠️ Generate Corrected Dataset",
                type="primary",
                use_container_width=True
            ):
                corrected, log = correct_dataset(
                    train_data,
                    new_data,
                    selected,
                    numeric_method,
                    fill_categorical
                )

                st.session_state["corrected_data"] = corrected
                st.session_state["correction_log"] = log
                st.session_state["corrected_for_upload"] = True

                st.success("Corrected copy generated successfully.")

            if st.session_state.get("corrected_for_upload", False):
                corrected = st.session_state["corrected_data"]
                log = st.session_state.get(
                    "correction_log", pd.DataFrame()
                )

                after_result = calculate_drift(
                    train_data, corrected
                )

                st.markdown("### 2. Correction Change Log")

                if log.empty:
                    st.info(
                        "No features were selected, or no changes were required."
                    )
                else:
                    st.dataframe(
                        log,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "📄 Download Correction Report",
                        data=log.to_csv(index=False).encode("utf-8"),
                        file_name="drift_correction_report.csv",
                        mime="text/csv"
                    )

                st.markdown("### 3. Drift After Correction")

                comparison = before_result[
                    ["Feature", "Drift Score"]
                ].merge(
                    after_result[["Feature", "Drift Score"]],
                    on="Feature",
                    suffixes=(" Before", " After")
                )

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                before_max = (
                    float(before_result["Drift Score"].max())
                    if not before_result.empty else 0.0
                )
                after_max = (
                    float(after_result["Drift Score"].max())
                    if not after_result.empty else 0.0
                )

                m1, m2, m3 = st.columns(3)

                m1.metric("Maximum Drift Before", f"{before_max:.3f}")
                m2.metric("Maximum Drift After", f"{after_max:.3f}")
                m3.metric(
                    "Score Change",
                    f"{before_max - after_max:+.3f}"
                )

                if after_max < before_max:
                    st.success(
                        "Maximum measured drift decreased. "
                        "Review feature-level results before using the data."
                    )
                elif after_max > before_max:
                    st.warning(
                        "Maximum measured drift increased. Review the selected settings."
                    )
                else:
                    st.info("Maximum measured drift did not change.")

                st.markdown("### 4. Corrected Dataset Preview")
                st.dataframe(
                    corrected.head(10),
                    use_container_width=True
                )

                st.download_button(
                    "📥 Download Corrected Dataset CSV",
                    data=corrected.to_csv(index=False).encode("utf-8"),
                    file_name="corrected_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ---------------- ABOUT ----------------

elif page == "ℹ️ About":
    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 📊 Data Drift Early Warning System

    This project compares historical training data with new incoming data
    to identify changes in feature distributions.

    ### 🎯 Project Objectives

    - Compare common features between training and incoming datasets.
    - Calculate numerical and categorical drift scores.
    - Display Normal, Warning, and Drift statuses.
    - Explore individual feature distributions.
    - Generate an optional corrected copy and a change report.

    ### 🔄 System Workflow
    """)

    st.code("""
Training Data + Incoming Data
             ↓
      Compare Features
             ↓
       Calculate Drift
             ↓
      Normal / Warning / Drift
             ↓
      Review and Investigate
             ↓
    Optional Correction + Report
""")

    st.markdown("### 🛠️ Technologies Used")

    a, b, c, d = st.columns(4)

    a.info("🐍 Python")
    b.info("📊 Pandas / NumPy")
    c.info("🎨 Streamlit")
    d.info("📈 SciPy / Matplotlib")

    st.markdown("### 🚀 DevOps Workflow")

    st.markdown("""
    **Git → GitHub → Jenkins → Docker → Kubernetes → Streamlit**

    - **Git:** Tracks code changes.
    - **GitHub:** Stores the source code.
    - **Jenkins:** Automates build and test steps.
    - **Docker:** Packages the application.
    - **Kubernetes:** Manages containers.
    - **Streamlit:** Displays the monitoring dashboard.
    """)

    st.info(
        "Drift is a statistical warning, not a direct measurement of model accuracy. "
        "Always validate corrected data before using it in production."
    )