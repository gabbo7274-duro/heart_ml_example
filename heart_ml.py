# heart_ml_app.py
"""
Heart Disease – ML Demo

Piccola applicazione Streamlit che:
carica un dataset pubblico su malattia cardiaca
allena un modello binario (malattia sì/no)
mostra alcune metriche di performance
permette di fare una previsione per un singolo paziente
"""
import streamlit as st  
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# *PAGE CONFIG 
st.set_page_config(page_title="Heart Disease – ML Demo", layout="wide", page_icon="❤️")

# *VARS  -----------------------------------------------------------
# Data path
DATA_PATH = Path("data/heart.csv")

# Cols selezionate come feature
FEATURE_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Etichette leggibili in italiano per l'UI
FEATURE_LABELS = {
    "age": "Età (anni)",
    "trestbps": "Pressione a riposo (mm Hg)",
    "chol": "Colesterolo (mg/dl)",
    "thalch": "Freq. cardiaca max (bpm)",
    "oldpeak": "Depressione ST (oldpeak)",
}
TARGET_LABEL = "Presenza di malattia (target)"

# *UTILS -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Legge il csv e crea la colonna target binaria."""
    df = pd.read_csv(DATA_PATH)

    # num: 0 = sano, 1–4 = malattia
    df["target"] = (df["num"] > 0).astype(int)

    cols = FEATURE_COLS + ["target"]
    return df[cols]


@st.cache_resource
def train_model(df: pd.DataFrame, max_depth: int, min_samples_leaf: int):
    """
    Allena un RandomForest e calcola:
    - accuracy su train e test
    - baseline (classe più frequente)
    - importanza delle feature
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=max_depth,             # <--- iperparametro dalla sidebar
        min_samples_leaf=min_samples_leaf,  # <--- iperparametro dalla sidebar
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # baseline: predire sempre la classe più frequente
    majority_class = int(y_test.value_counts().idxmax())
    baseline_pred = [majority_class] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "baseline_acc": baseline_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "overfit_gap": acc_train - acc_test,
    }

    # importanza delle feature
    fi = pd.Series(
        model.feature_importances_,
        index=[FEATURE_LABELS[c] for c in FEATURE_COLS],
    ).sort_values(ascending=True)

    return model, metrics, fi


######################################################################
# ------------------------------ ST APP ---------------------------- #
######################################################################

df = load_data()

# *SIDEBAR – IPERPARAMETRI RF ---------------------------------------

st.sidebar.header("⚙️ Iperparametri RandomForest")

max_depth = st.sidebar.slider(
    "Profondità massima (max_depth)",
    min_value=1,
    max_value=20,
    value=5,
    help="Limita la profondità degli alberi: valori più bassi riducono l'overfitting."
)

min_samples_leaf = st.sidebar.slider(
    "Min campioni per foglia (min_samples_leaf)",
    min_value=1,
    max_value=20,
    value=1,
    help="Numero minimo di campioni in una foglia: valori più alti rendono il modello più semplice."
)

st.title("❤️ Heart Disease – ML Demo")
st.caption(
    "Esempio didattico: modello binario (malattia sì/no) su 5 feature "
    "numeriche. Non è uno strumento medico reale."
)

# *PANORAMICA --------------------------------------------------------

st.subheader("🔍 Panoramica del dataset")

col_a, col_b, col_c = st.columns(3)

n_patients = len(df)
positive_rate = df["target"].mean()

col_a.metric("Numero pazienti", n_patients)
col_b.metric("Con malattia (%)", f"{positive_rate:.1%}")
col_c.metric(
    "Sani vs malati",
    f"{(1 - positive_rate):.1%} sani / {positive_rate:.1%} malati",
)

with st.expander("Mostra prime righe del dataset"):
    st.dataframe(df.head())

# *ESPLORAZIONE VARIABILE SINGOLA -----------------------------------

st.markdown("### 📊 Esplorazione di una singola variabile")

selected_feature = st.selectbox(
    "Seleziona una variabile",
    FEATURE_COLS,
    format_func=lambda c: FEATURE_LABELS[c],
)

selected_label = FEATURE_LABELS[selected_feature]

plot_col1, plot_col2 = st.columns(2)

# Istogramma distribuzione variabile selezionata
with plot_col1:
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax_hist)
    ax_hist.set_title(f"Distribuzione di {selected_label}")
    ax_hist.set_xlabel(selected_label)
    st.pyplot(fig_hist)

# Violin plot sani vs malati
with plot_col2:
    fig_violin, ax_violin = plt.subplots()
    df_violin = df.copy()
    df_violin["stato"] = df_violin["target"].map({0: "Sano", 1: "Malato"})
    sns.violinplot(
        data=df_violin,
        x="stato",
        y=selected_feature,
        ax=ax_violin,
    )
    ax_violin.set_xlabel("Stato")
    ax_violin.set_ylabel(selected_label)
    ax_violin.set_title(f"{selected_label} – sani vs malati")
    st.pyplot(fig_violin)

# Tabella con statistiche descrittive
st.markdown("**Statistiche descrittive**")
desc = df[selected_feature].describe().to_frame(name=selected_label)
st.table(desc)

# * RF PERFORMANCE ---------------------------------------------------

# <-- ORA il modello usa gli iperparametri della sidebar
model, metrics, feature_importances = train_model(df, max_depth, min_samples_leaf)

st.subheader("📏 Performance del modello")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")
col3.metric(
    "Baseline (classe più frequente)",
    f"{metrics['baseline_acc']:.2%}",
)

st.metric(
    "Gap train–test (overfitting)",
    f"{metrics['overfit_gap']:.2%}",
)

gap = metrics["acc_train"] - metrics["acc_test"]
if gap > 0.2:
    st.warning(
        "Possibile overfitting: il modello va molto meglio su train "
        f"({metrics['acc_train']:.0%}) che su test "
        f"({metrics['acc_test']:.0%}). Prova ad abbassare la profondità "
        "degli alberi o ad aumentare il minimo di pazienti per foglia."
    )
elif gap > 0.1:
    st.info(
        "Leggero overfitting: il modello è più preciso su train "
        f"({metrics['acc_train']:.0%}) che su test "
        f"({metrics['acc_test']:.0%}), ma il gap è ancora accettabile."
    )
else:
    st.success(
        "Train e test hanno performance simili: il modello sembra "
        "generalizzare bene."
    )



st.caption(
    "Un gap train–test più basso indica minore overfitting. "
    "Gioca con max_depth e min_samples_leaf nella sidebar per vedere come cambiano le metriche."
)

# *CORR & FEATURE IMPORTANCE -----------------------------------------

st.subheader("📈 Correlazioni e importanza delle variabili")

col_corr, col_imp = st.columns(2)

with col_corr:
    st.markdown("**Correlazione tra variabili e target**")

    # Rename cols for corr matrix plot
    df_corr = df.copy()
    rename_map = {col: FEATURE_LABELS[col] for col in FEATURE_COLS}
    rename_map["target"] = TARGET_LABEL
    df_corr = df_corr.rename(columns=rename_map)

    # compute corr matrix
    corr = df_corr.corr()

    # corr matrix heatmap with sns
    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax_corr,
    )
    ax_corr.set_title("Matrice di correlazione")
    st.pyplot(fig_corr)

with col_imp:
    st.markdown("**Importanza delle variabili (RandomForest)**")

    # Plot feature importances barchart with matplotlib
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    feature_importances.plot(kind="barh", ax=ax_imp)
    ax_imp.set_xlabel("Importanza (Gini)")
    ax_imp.set_ylabel("Variabile")
    ax_imp.set_title("Importanza delle variabili")
    plt.tight_layout()
    st.pyplot(fig_imp)

    # --- VARIABILE PIÙ IMPORTANTE + BOXPLOT VS TARGET -----------------
    st.markdown("---")

    # label leggibile (es. "Depressione ST (oldpeak)")
    top_feature_label = feature_importances.idxmax()
    # mappa inversa per tornare al nome colonna originale (es. "oldpeak")
    inv_labels = {v: k for k, v in FEATURE_LABELS.items()}
    top_feature_col = inv_labels[top_feature_label]

    st.markdown(f"**Variabile più importante:** {top_feature_label}")

    # Boxplot sani vs malati per la variabile più importante
    df_top = df.copy()
    df_top["stato"] = df_top["target"].map({0: "Sano", 1: "Malato"})

    fig_box, ax_box = plt.subplots(figsize=(4, 3))
    sns.boxplot(
        data=df_top,
        x="stato",
        y=top_feature_col,
        ax=ax_box,
    )
    ax_box.set_xlabel("Stato")
    ax_box.set_ylabel(top_feature_label)
    ax_box.set_title(f"{top_feature_label} – distribuzione per stato")
    st.pyplot(fig_box)

    # confronto media malati vs sani
    mean_sani = df_top.loc[df_top["stato"] == "Sano", top_feature_col].mean()
    mean_malati = df_top.loc[df_top["stato"] == "Malato", top_feature_col].mean()

    trend = "più alta" if mean_malati > mean_sani else "più bassa"
    st.markdown(
        f"Nei **malati** la media di **{top_feature_label}** è **{trend}** "
        f"rispetto ai sani.\n\n"
        f"- Media sani: {mean_sani:.2f}\n"
        f"- Media malati: {mean_malati:.2f}"
    )


# *FORM PAZIENTE ----------------------------------------------------

st.subheader("🧪 Inserisci i dati del paziente")

cols = st.columns(3)
user_input: dict[str, float] = {}

for i, col_name in enumerate(FEATURE_COLS):
    serie = df[col_name]
    min_val = float(serie.min())
    max_val = float(serie.max())
    default = float(serie.median())

    label = FEATURE_LABELS[col_name]

    with cols[i % 3]:
        user_input[col_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
        )

if st.button("Predici rischio"):
    input_df = pd.DataFrame([user_input])
    proba = model.predict_proba(input_df)[0]
    pred = int(proba[1] > 0.5)

    col_res1, col_res2 = st.columns(2)
    label_risk = "ALTO" if pred == 1 else "BASSO"
    col_res1.metric("Rischio stimato", label_risk)
    col_res2.metric("Probabilità di malattia", f"{proba[1]:.1%}")

    st.write("Valori inseriti:")
    pretty_input = {FEATURE_LABELS[k]: v for k, v in user_input.items()}
    st.json(pretty_input)

    st.info(
        "⚠️ Esempio didattico su un dataset pubblico. "
        "Non è uno strumento clinico e non va usato per decisioni reali."
    )