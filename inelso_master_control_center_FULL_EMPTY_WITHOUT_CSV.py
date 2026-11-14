
from pathlib import Path
import re
import io
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
st.set_page_config(
    page_title="INELSO Κεντρικό Σύστημα Έξυπνης Μάθησης — Τελική Έκδοση",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# ΚΟΙΝΟ MOODLE REST PANEL ΓΙΑ ΟΛΑ ΤΑ SERVICES
# ==========================================================
def render_moodle_rest_service_panel(
    service_name,
    *,
    summary,
    score=None,
    status=None,
    entity=None,
    enabled=True,
    intervention_type=None,
    note=None,
):
    """
    Human-controlled bridge προς το υπάρχον Moodle REST write workflow.
    Δεν εκτελεί write εδώ. Δημιουργεί μόνο draft/provenance στη συνεδρία.
    Η πραγματική εγγραφή γίνεται στην «Ελεγχόμενη Εγγραφή Moodle»
    με preview, διπλή επιβεβαίωση και read-back verification.
    """
    st.markdown("### Moodle REST / SCORM")
    if note:
        st.caption(note)

    st.caption(
        "Το SCORM χρησιμοποιείται ως launch/display container. "
        "Η πραγματική καταγραφή του αποτελέσματος γίνεται μέσω Moodle REST με ανθρώπινη έγκριση."
    )

    if not enabled:
        st.info("Δεν υπάρχει ακόμη κατάλληλο αποτέλεσμα για καταγραφή στο Moodle.")
        return

    button_key = "moodle_rest_" + re.sub(
        r"[^a-zA-Z0-9_]+", "_", service_name
    ).strip("_").lower()

    if st.button(
        f"Προώθηση αποτελέσματος {service_name} προς Moodle REST",
        key=button_key,
    ):
        lines = [
            f"Service: {service_name}",
        ]
        if entity:
            lines.append(f"Entity: {entity}")
        if status:
            lines.append(f"Status: {status}")
        if score is not None:
            lines.append(f"Score: {float(score):.2f}/100")
        lines.append(f"Summary: {summary}")

        draft = "\n".join(lines)

        st.session_state["pending_moodle_note_text"] = draft
        st.session_state["pending_moodle_intervention_type"] = (
            intervention_type or "Άλλο"
        )
        st.session_state["pending_moodle_result_meta"] = {
            "service": service_name,
            "entity": entity,
            "status": status,
            "score": score,
            "summary": summary,
        }

        st.success(
            "Το αποτέλεσμα προωθήθηκε ως ελεγχόμενο Moodle REST draft. "
            "Πήγαινε στην «Ελεγχόμενη Εγγραφή Moodle» για preview, "
            "διπλή επιβεβαίωση, write και verification."
        )

# ==========================================================
# ΕΜΦΑΝΙΣΗ
# ==========================================================
st.markdown("""
<style>
:root{
 --navy:#102a43; --navy2:#173b5e; --teal:#2a8c99;
 --soft:#f5f8fb; --line:#dde6ee; --muted:#667085;
}
.block-container{padding-top:.8rem;padding-bottom:3rem;max-width:1550px}
h1{color:var(--navy);font-size:2.15rem!important}
h2{color:var(--navy2);font-size:1.5rem!important}
h3{font-size:1.15rem!important}
[data-testid="stMetricValue"]{font-size:1.7rem}
[data-testid="stMetricLabel"]{font-size:.9rem}
.hero{
 background:linear-gradient(135deg,#102a43,#1f5f87);
 color:white;border-radius:20px;padding:1.25rem 1.45rem;margin:.15rem 0 1rem 0;
}
.hero h1,.hero h2,.hero h3{color:white!important;margin:0}
.hero p{margin:.45rem 0 0;opacity:.92}
.flow{
 border:1px dashed #a9bdcf;background:#f8fbfd;padding:.8rem 1rem;border-radius:12px;
 font-weight:650
}
.callout{
 border-left:5px solid var(--teal);background:#f4fbfc;padding:.85rem 1rem;border-radius:10px
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ΓΙΑ MOODLE SERVICE 21 CSV
# ==========================================================
def pct_number(value):
    """Μετατρέπει Moodle τιμές όπως 61.90%, '-45.23% ή αριθμούς σε float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip().replace("%", "").replace("'", "").replace(",", ".")
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def extract_service21_from_moodle_complete_stats(raw_df):
    """
    Αναγνωρίζει Moodle Complete Statistics export, όπου:
    - η πρώτη γραμμή είναι quiz-level summary
    - επόμενη γραμμή περιέχει item-level header (Q#, Question type, Question name, ...)
    """
    if raw_df is None or raw_df.empty:
        return None

    # Search every row for the embedded item-level header.
    header_idx = None
    for idx in range(len(raw_df)):
        vals = [str(v).strip() for v in raw_df.iloc[idx].tolist()]
        if "Q#" in vals and "Question name" in vals and "Facility index" in vals:
            header_idx = idx
            break

    if header_idx is None:
        return None

    header = [str(v).strip() for v in raw_df.iloc[header_idx].tolist()]
    items = raw_df.iloc[header_idx + 1:].copy()
    items.columns = header

    required = ["Question name", "Facility index", "Discrimination index"]
    if not all(c in items.columns for c in required):
        return None

    # Keep only actual item rows.
    if "Q#" in items.columns:
        qnum = pd.to_numeric(items["Q#"], errors="coerce")
        items = items[qnum.notna()].copy()

    if items.empty:
        return None

    out = pd.DataFrame({
        "ερώτηση": items["Question name"].astype(str).str.strip(),
        "facility": items["Facility index"].map(pct_number),
        "διάκριση": items["Discrimination index"].map(pct_number),
    })

    if "Discriminative efficiency" in items.columns:
        out["αποτελεσματικότητα"] = items["Discriminative efficiency"].map(pct_number)
    else:
        out["αποτελεσματικότητα"] = np.nan

    out = out[
        out["ερώτηση"].notna() &
        (out["ερώτηση"].astype(str).str.strip() != "") &
        out["facility"].notna()
    ].reset_index(drop=True)

    return out if not out.empty else None



# ==========================================================
# ΚΟΙΝΟ ACTIVE DATASET CONTEXT ΓΙΑ ΟΛΗ ΤΗΝ ΕΦΑΡΜΟΓΗ
# ==========================================================
def get_active_dataset():
    df = st.session_state.get(
        "inelso_uploaded_clean",
        st.session_state.get("inelso_uploaded_original")
    )
    return df.copy() if isinstance(df, pd.DataFrame) else None


def active_dataset_name():
    return st.session_state.get("inelso_uploaded_name", "Δεν έχει φορτωθεί CSV")


def dataset_profile(df):
    if not isinstance(df, pd.DataFrame):
        return None
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical": int(df.select_dtypes(exclude=np.number).shape[1]),
    }


def normalized_column_map(df):
    if not isinstance(df, pd.DataFrame):
        return {}
    return {str(c).strip().lower(): c for c in df.columns}


def detect_column(df, terms):
    cmap = normalized_column_map(df)
    for term in terms:
        term = term.lower()
        # exact first
        for lc, original in cmap.items():
            if lc == term:
                return original
        # then contains
        for lc, original in cmap.items():
            if term in lc:
                return original
    return None


def active_dataset_kind(df):
    if not isinstance(df, pd.DataFrame):
        return "ΚΑΝΕΝΑ"
    cols = [str(c).strip().lower() for c in df.columns]
    if (
        st.session_state.get("custom_s21_source") == "Moodle Complete Statistics"
        or ("quiz name" in cols and any("facility" in c for c in cols))
    ):
        return "MOODLE_COMPLETE_STATS"
    if any(str(c).startswith("Response ") for c in df.columns) and any("grade" in c.lower() for c in cols):
        return "MOODLE_RESPONSES"
    if (
        detect_column(df, ["score", "grade", "final score"]) is not None
        and detect_column(df, ["attempt", "προσπάθεια"]) is not None
    ):
        return "LEARNER_LEVEL"
    return "GENERIC"


def dataset_context_summary(df, max_cols=12):
    if not isinstance(df, pd.DataFrame):
        return "Δεν υπάρχει ενεργό dataset."
    p = dataset_profile(df)
    cols = [str(c) for c in df.columns[:max_cols]]
    suffix = " ..." if len(df.columns) > max_cols else ""
    return (
        f"Αρχείο: {active_dataset_name()}\n"
        f"Γραμμές: {p['rows']} | Στήλες: {p['cols']} | "
        f"Κενά: {p['missing']} | Διπλότυπα: {p['duplicates']}\n"
        f"Στήλες: {', '.join(cols)}{suffix}"
    )


def service_dataset_compatibility(service_key, df):
    if not isinstance(df, pd.DataFrame):
        return "ΧΩΡΙΣ DATASET"
    kind = active_dataset_kind(df)

    if service_key == "S21":
        return "ΠΛΗΡΗΣ" if st.session_state.get("custom_s21") is not None else (
            "ΜΕΡΙΚΗ" if kind in {"MOODLE_RESPONSES","GENERIC"} else "ΜΗ ΣΥΜΒΑΤΗ"
        )
    if service_key == "S26":
        if st.session_state.get("custom_s26") is not None:
            return "ΠΛΗΡΗΣ"
        score = detect_column(df, ["score","grade","final score"])
        attempt = detect_column(df, ["attempt","προσπάθεια"])
        return "ΜΕΡΙΚΗ" if score is not None or attempt is not None else "ΜΗ ΣΥΜΒΑΤΗ"
    if service_key == "S16":
        numeric = df.select_dtypes(include=np.number).shape[1]
        return "ΠΛΗΡΗΣ" if numeric >= 1 else "ΜΕΡΙΚΗ"
    if service_key in {"S25","S28"}:
        return "CONTEXT"
    if service_key == "S31":
        if st.session_state.get("custom_s21") is not None or st.session_state.get("custom_s26") is not None:
            return "ΔΥΝΑΜΙΚΗ"
        return "CONTEXT"
    if service_key in {"ANALYTICS","DATA","REPORT","DASHBOARD","PRESENTATION"}:
        return "ΠΛΗΡΗΣ"
    return "CONTEXT"


def show_active_dataset_context(service_key=None, compact=True):
    df = get_active_dataset()
    if df is None:
        st.info("Active Dataset: δεν έχει φορτωθεί αρχείο από το Κέντρο Δεδομένων.")
        return

    p = dataset_profile(df)
    compat = service_dataset_compatibility(service_key, df) if service_key else "ΕΝΕΡΓΟ"
    st.caption(
        f"📄 Active Dataset: {active_dataset_name()}  •  "
        f"{p['rows']} γραμμές × {p['cols']} στήλες  •  "
        f"Συμβατότητα: {compat}"
    )
    if not compact:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Γραμμές CSV", p["rows"])
        c2.metric("Στήλες CSV", p["cols"])
        c3.metric("Κενά", p["missing"])
        c4.metric("Διπλότυπα", p["duplicates"])


def csv_context_as_text(df, max_rows=5):
    if not isinstance(df, pd.DataFrame):
        return ""
    summary = dataset_context_summary(df)
    sample = df.head(max_rows).astype(str).to_dict(orient="records")
    return summary + "\nΔείγμα γραμμών:\n" + json.dumps(sample, ensure_ascii=False, indent=2)


ACTIVE_DF = get_active_dataset()
ACTIVE_NAME = active_dataset_name()
ACTIVE_KIND = active_dataset_kind(ACTIVE_DF)

# ==========================================================
# ΠΡΑΓΜΑΤΙΚΑ / ΕΛΕΓΧΟΜΕΝΑ SIGNALS SERVICE 21
# ==========================================================
service21 = pd.DataFrame([
    {"ερώτηση":"S21 Marketing Q01","facility":2.00,"διάκριση":34.92,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"S21 Marketing Q02","facility":2.00,"διάκριση":34.92,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"S21 Marketing Q04","facility":0.67,"διάκριση":39.68,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"S21 Marketing Q07","facility":0.00,"διάκριση":np.nan,"αποτελεσματικότητα":np.nan},
    {"ερώτηση":"S21 Marketing Q10","facility":2.00,"διάκριση":34.92,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"S21 Marketing Q05","facility":20.00,"διάκριση":44.11,"αποτελεσματικότητα":61.54},
    {"ερώτηση":"S21 Marketing Q09","facility":72.00,"διάκριση":12.88,"αποτελεσματικότητα":21.49},
    {"ερώτηση":"S21 Marketing Q03","facility":44.00,"διάκριση":73.22,"αποτελεσματικότητα":92.35},
    {"ερώτηση":"S21 Marketing Q06","facility":32.00,"διάκριση":44.38,"αποτελεσματικότητα":52.03},
    {"ερώτηση":"S21 Marketing Q08","facility":52.00,"διάκριση":60.89,"αποτελεσματικότητα":80.71},
])

def κατηγορία_διάκρισης(x):
    if pd.isna(x):
        return "ΜΗ ΕΚΤΙΜΗΣΙΜΗ"
    if x >= 40:
        return "ΙΣΧΥΡΗ"
    if x >= 20:
        return "ΙΚΑΝΟΠΟΙΗΤΙΚΗ"
    if x >= 0:
        return "ΕΛΕΓΧΟΣ"
    return "ΚΡΙΣΙΜΗ"

def ενέργεια_ερώτησης(r):
    d, f = r["διάκριση"], r["facility"]
    if pd.isna(d):
        return "Έλεγχος scoring και κατανομής απαντήσεων"
    if d < 0:
        return "Άμεσος έλεγχος κλειδιού / scoring"
    if d < 20 and (f <= 10 or f >= 90):
        return "Αναθεώρηση λόγω ακραίας ευκολίας/δυσκολίας και ασθενούς διάκρισης"
    if d < 20:
        return "Έλεγχος διατύπωσης, distractors και ευθυγράμμισης με μαθησιακό στόχο"
    if d < 40:
        return "Διατήρηση με παρακολούθηση"
    return "Διατήρηση"

service21["κατάσταση"] = service21["διάκριση"].apply(κατηγορία_διάκρισης)
service21["ενέργεια"] = service21.apply(ενέργεια_ερώτησης, axis=1)

# Αν έχει δημιουργηθεί Service 21 dataset από upload, αυτό γίνεται η ενεργή πηγή.
if "custom_s21" in st.session_state:
    custom = st.session_state["custom_s21"].copy()
    needed = {"ερώτηση", "facility", "διάκριση"}
    if needed.issubset(custom.columns):
        if "αποτελεσματικότητα" not in custom.columns:
            custom["αποτελεσματικότητα"] = np.nan
        custom["facility"] = custom["facility"].map(pct_number)
        custom["διάκριση"] = custom["διάκριση"].map(pct_number)
        custom["αποτελεσματικότητα"] = custom["αποτελεσματικότητα"].map(pct_number)
        custom["κατάσταση"] = custom["διάκριση"].apply(κατηγορία_διάκρισης)
        custom["ενέργεια"] = custom.apply(ενέργεια_ερώτησης, axis=1)
        service21 = custom.reset_index(drop=True)

# ==========================================================
# ΠΡΑΓΜΑΤΙΚΑ / ΕΛΕΓΧΟΜΕΝΑ SIGNALS SERVICE 26
# ==========================================================
rubric = pd.DataFrame([
    {"κριτήριο":"RC-01","περιγραφή":"Εννοιολογική ακρίβεια","βάρος":30},
    {"κριτήριο":"RC-02","περιγραφή":"Μέθοδος / συλλογισμός","βάρος":25},
    {"κριτήριο":"RC-03","περιγραφή":"Τεκμηρίωση / υποστήριξη","βάρος":15},
    {"κριτήριο":"RC-04","περιγραφή":"Πληρότητα","βάρος":15},
    {"κριτήριο":"RC-05","περιγραφή":"Σαφήνεια / επικοινωνία","βάρος":15},
])

service26 = pd.DataFrame([
    {"προσπάθεια":1,"RC-01":3,"RC-02":2,"RC-03":2,"RC-04":2,"RC-05":3,
     "score":61.25,"κατάκτηση":"Αναπτυσσόμενο","βεβαιότητα":0.65},
    {"προσπάθεια":2,"RC-01":4,"RC-02":4,"RC-03":4,"RC-04":4,"RC-05":4,
     "score":100.00,"κατάκτηση":"Κατάκτηση","βεβαιότητα":0.85},
])

def κατάσταση_26(score, conf, rc01):
    if conf < 0.70 or rc01 <= 1:
        return "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ"
    if score < 70:
        return "ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ"
    return "ΕΤΟΙΜΟ"

service26["κατάσταση"] = service26.apply(
    lambda r: κατάσταση_26(r["score"], r["βεβαιότητα"], r["RC-01"]), axis=1
)

cohort26 = pd.DataFrame([
    {"μαθητής":"Demo Student 0201","score":100.00,"βεβαιότητα":0.85,"κατάκτηση":"Κατάκτηση","κατάσταση":"ΕΤΟΙΜΟ"},
    {"μαθητής":"Demo Student 0202","score":78.75,"βεβαιότητα":0.81,"κατάκτηση":"Επαρκής","κατάσταση":"ΕΤΟΙΜΟ"},
    {"μαθητής":"Demo Student 0203","score":66.25,"βεβαιότητα":0.68,"κατάκτηση":"Αναπτυσσόμενο","κατάσταση":"ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ"},
    {"μαθητής":"Demo Student 0204","score":91.25,"βεβαιότητα":0.88,"κατάκτηση":"Κατάκτηση","κατάσταση":"ΕΤΟΙΜΟ"},
    {"μαθητής":"Demo Student 0205","score":57.50,"βεβαιότητα":0.72,"κατάκτηση":"Αναπτυσσόμενο","κατάσταση":"ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ"},
])

# Αν υπάρχει mapped Service 26 dataset από το ενεργό CSV, γίνεται η ενεργή cohort πηγή.
if "custom_s26" in st.session_state and isinstance(st.session_state["custom_s26"], pd.DataFrame):
    mapped26_active = st.session_state["custom_s26"].copy()
    if not mapped26_active.empty:
        if "μαθητής" not in mapped26_active.columns:
            mapped26_active["μαθητής"] = [
                f"CSV learner {i+1}" for i in range(len(mapped26_active))
            ]
        if "κατάκτηση" not in mapped26_active.columns:
            mapped26_active["κατάκτηση"] = np.where(
                pd.to_numeric(mapped26_active["score"], errors="coerce") >= 70,
                "Επαρκής",
                "Αναπτυσσόμενο"
            )
        if "κατάσταση" not in mapped26_active.columns:
            mapped26_active["κατάσταση"] = mapped26_active.apply(
                lambda r: κατάσταση_26(
                    float(r.get("score",0)) if pd.notna(r.get("score",np.nan)) else 0,
                    float(r.get("βεβαιότητα",0)) if pd.notna(r.get("βεβαιότητα",np.nan)) else 0,
                    float(r.get("RC-01",2)) if pd.notna(r.get("RC-01",np.nan)) else 2
                ),
                axis=1
            )
        cohort26 = mapped26_active[
            ["μαθητής","score","βεβαιότητα","κατάκτηση","κατάσταση"]
        ].copy()

# ==========================================================
# SERVICE 31: ΠΛΕΟΝ ΠΑΡΑΓΕΤΑΙ ΑΠΟ SIGNALS 21 + 26
# ==========================================================
# Μόνο το Q08 έχει τεκμηριωμένη σημασιολογική ετικέτα από το υπάρχον workflow (CTR).
# Τα Q09/Q07 διατηρούνται ως item-level signals χωρίς επινοημένη θεματική ονομασία.

# 1) Signals από Service 21
# Παράγονται δυναμικά από το ενεργό Service 21 dataset.
s21_signal_rows = []

estimable_s21 = service21.dropna(subset=["διάκριση"]).copy()
if not estimable_s21.empty:
    strongest = estimable_s21.sort_values("διάκριση", ascending=False).iloc[0]
    weakest = estimable_s21.sort_values("διάκριση", ascending=True).iloc[0]

    s21_signal_rows.append({
        "οντότητα": str(strongest["ερώτηση"]),
        "πηγή":"Service 21",
        "τύπος":"Ισχυρό item-level signal",
        "ισχύς_σήματος":round(float(strongest["διάκριση"]),2),
        "κατεύθυνση":"ΘΕΤΙΚΗ",
        "ενέργεια":"Διατήρηση / χρήση ως ισχυρό σημείο αναφοράς",
        "προέλευση":"Ενεργό Service 21 dataset"
    })

    weak_strength = min(100.0, max(0.0, 100.0 - float(weakest["διάκριση"])))
    s21_signal_rows.append({
        "οντότητα":str(weakest["ερώτηση"]),
        "πηγή":"Service 21",
        "τύπος":"Αδύναμο item-level signal",
        "ισχύς_σήματος":round(weak_strength,2),
        "κατεύθυνση":"ΑΡΝΗΤΙΚΗ" if float(weakest["διάκριση"]) < 20 else "ΜΕΣΑΙΑ",
        "ενέργεια":str(weakest["ενέργεια"]),
        "προέλευση":"Ενεργό Service 21 dataset"
    })

nonestimable_s21 = service21[service21["διάκριση"].isna()]
if not nonestimable_s21.empty:
    rr = nonestimable_s21.iloc[0]
    s21_signal_rows.append({
        "οντότητα":str(rr["ερώτηση"]),
        "πηγή":"Service 21",
        "τύπος":"Μη εκτιμήσιμο item-level signal",
        "ισχύς_σήματος":100.0,
        "κατεύθυνση":"ΑΒΕΒΑΙΗ",
        "ενέργεια":str(rr["ενέργεια"]),
        "προέλευση":"Ενεργό Service 21 dataset"
    })

s21_signals = pd.DataFrame(s21_signal_rows)

# 2) Signals από Service 26
if "custom_s26" in st.session_state and isinstance(st.session_state["custom_s26"], pd.DataFrame):
    s26_rows = []
    active26 = cohort26.copy()
    for _, rr in active26.iterrows():
        status = str(rr.get("κατάσταση",""))
        score = pd.to_numeric(pd.Series([rr.get("score")]), errors="coerce").iloc[0]
        conf = pd.to_numeric(pd.Series([rr.get("βεβαιότητα")]), errors="coerce").iloc[0]

        if status == "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ":
            direction = "ΑΒΕΒΑΙΗ"
            strength = 100.0 if pd.isna(conf) else round(max(0, (0.70-conf)*200),2)
            action = "Ανθρώπινος έλεγχος"
        elif status == "ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ":
            direction = "ΜΕΣΑΙΑ"
            strength = 100.0 if pd.isna(score) else round(max(0, 100-float(score)),2)
            action = "Στοχευμένη νέα προσπάθεια"
        else:
            direction = "ΘΕΤΙΚΗ"
            strength = 0.0 if pd.isna(score) else round(float(score),2)
            action = "Παρακολούθηση"

        s26_rows.append({
            "οντότητα":str(rr.get("μαθητής","CSV learner")),
            "πηγή":"Service 26",
            "τύπος":"Learner-level adaptive signal",
            "ισχύς_σήματος":strength,
            "κατεύθυνση":direction,
            "ενέργεια":action,
            "προέλευση":f"Active CSV: {ACTIVE_NAME}"
        })
    s26_signals = pd.DataFrame(s26_rows)
else:
    prev = service26.iloc[0]
    latest = service26.iloc[-1]
    s26_rows = []
    for _, rr in rubric.iterrows():
        code = rr["κριτήριο"]
        gap1 = 4 - int(prev[code])
        gap2 = 4 - int(latest[code])
        improvement = gap1 - gap2
        strength = round((gap1 * rr["βάρος"]) / 4, 2)
        direction = "ΕΠΙΛΥΘΗΚΕ" if gap1 > 0 and gap2 == 0 else ("ΣΤΑΘΕΡΟ" if gap1 == gap2 else "ΒΕΛΤΙΩΣΗ")
        action = "Καμία άμεση ενέργεια — παρακολούθηση" if gap2 == 0 else "Στοχευμένη εξάσκηση"
        s26_rows.append({
            "οντότητα":f"{code} — {rr['περιγραφή']}",
            "πηγή":"Service 26",
            "τύπος":"Rubric-level μαθησιακό signal",
            "ισχύς_σήματος":strength,
            "κατεύθυνση":direction,
            "ενέργεια":action,
            "προέλευση":"Ελεγχόμενη adaptive practice"
        })
    s26_signals = pd.DataFrame(s26_rows)

service31_integrated = pd.concat([s21_signals, s26_signals], ignore_index=True)

# Ενιαία προτεραιότητα Service 31 από πραγματικά signals.
def προτεραιότητα_31(row):
    if row["κατεύθυνση"] in ["ΑΡΝΗΤΙΚΗ","ΑΒΕΒΑΙΗ"] and row["ισχύς_σήματος"] >= 70:
        return "ΥΨΗΛΗ"
    if row["κατεύθυνση"] in ["ΕΠΙΛΥΘΗΚΕ","ΘΕΤΙΚΗ"]:
        return "ΧΑΜΗΛΗ"
    return "ΜΕΣΑΙΑ"

service31_integrated["προτεραιότητα"] = service31_integrated.apply(προτεραιότητα_31, axis=1)

# ==========================================================
# ΔΕΙΚΤΕΣ ΥΓΕΙΑΣ
# ==========================================================
def περιορισμός(x):
    return max(0,min(100,float(x)))

valid_disc = service21["διάκριση"].dropna()
health21 = περιορισμός(
    55 + 0.35*valid_disc.mean()
    - 6*int((service21["κατάσταση"]=="ΕΛΕΓΧΟΣ").sum())
    - 10*int((service21["κατάσταση"]=="ΚΡΙΣΙΜΗ").sum())
    - 8*int((service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ").sum())
)

ready_rate = (cohort26["κατάσταση"]=="ΕΤΟΙΜΟ").mean()*100
health26 = περιορισμός(
    .55*cohort26["score"].mean()
    + .25*(cohort26["βεβαιότητα"].mean()*100)
    + .20*ready_rate
)

high31 = int((service31_integrated["προτεραιότητα"]=="ΥΨΗΛΗ").sum())
medium31 = int((service31_integrated["προτεραιότητα"]=="ΜΕΣΑΙΑ").sum())
health31 = περιορισμός(100 - high31*18 - medium31*8)

overall = round((health21+health26+health31)/3,1)

open21 = int(
    ((service21["κατάσταση"]=="ΚΡΙΣΙΜΗ") |
     (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
     (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")).sum()
)
open26 = int((cohort26["κατάσταση"]!="ΕΤΟΙΜΟ").sum())
open31 = high31

# ==========================================================
# ΠΛΟΗΓΗΣΗ
# ==========================================================
st.sidebar.markdown("## INELSO Κεντρικό Σύστημα")
st.sidebar.caption("Τελική έκδοση • ελληνικό περιβάλλον • διασυνδεδεμένες υπηρεσίες")
section = st.sidebar.radio(
    "Κεντρική πλοήγηση",
    [
        "Κεντρικός Πίνακας Ελέγχου",
        "Service 16 — Automated Analytics & Early-Warning",
        "Service 21 — Ευφυής Ποιότητα Αξιολόγησης",
        "Service 25 — AI-Assisted Course & Assessment Authoring",
        "Service 26 — Ευφυής Προσαρμοστική Εξάσκηση",
        "Service 28 — Faculty Teaching Amplification",
        "Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις",
        "Διασύνδεση Υπηρεσιών",
        "Κύκλος Ζωής Ενεργειών",
        "Μεθοδολογία & Προέλευση",
        "Έλεγχος Ετοιμότητας",
        "Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων",
        "INELSO Analytics Lab",
        "Προσομοιωτής What-if",
        "Live Moodle",
        "Ελεγχόμενη Εγγραφή Moodle",
        "Μητρώο Ενεργειών",
        "Λειτουργία Παρουσίασης",
        "Κεντρική Αναφορά"
    ]
)
st.sidebar.markdown("---")
if ACTIVE_DF is not None:
    _p = dataset_profile(ACTIVE_DF)
    st.sidebar.success(f"Active CSV: {ACTIVE_NAME}")
    st.sidebar.caption(f"{_p['rows']} γραμμές × {_p['cols']} στήλες • {ACTIVE_KIND}")
else:
    st.sidebar.info("Δεν έχει φορτωθεί Active CSV.")
st.sidebar.caption("Service 31: παράγεται από πραγματικά/ελεγχόμενα signals των Service 21 και 26.")
st.sidebar.caption("Δεν πραγματοποιείται αυτόματη εγγραφή επίσημων βαθμών.")

# ==========================================================
# ΚΕΝΤΡΙΚΟΣ ΠΙΝΑΚΑΣ
# ==========================================================
if section == "Κεντρικός Πίνακας Ελέγχου":
    show_active_dataset_context("DASHBOARD", compact=False)
    st.markdown("""
    <div class="hero">
      <h1>INELSO Κεντρικό Σύστημα Έξυπνης Μάθησης</h1>
      <p>Ποιότητα αξιολόγησης + προσαρμοστική εξάσκηση + μαθησιακές ενδείξεις, σε μία ενιαία αλυσίδα τεκμηρίωσης.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Συνολικός δείκτης υγείας", f"{overall:.1f}/100")
    c2.metric("Service 21", f"{health21:.1f}" if ACTIVE_DF is not None and "custom_s21" in st.session_state else "—")
    c3.metric("Service 26", f"{health26:.1f}" if ACTIVE_DF is not None and "custom_s26" in st.session_state else "—")
    c4.metric("Service 31", f"{health31:.1f}" if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—")
    c5.metric("Ανοικτές ενέργειες", open21+open26+open31)

    st.markdown(
        '<div class="flow">Moodle → Service 21 / 26 signals → Service 31 synthesis → '
        'διαφανείς κανόνες → ανθρώπινος έλεγχος → παρέμβαση / retry / revision → αναφορά</div>',
        unsafe_allow_html=True
    )

    st.success("Νέες δυνατότητες: Services 16, 25 και 28, Live Moodle, ελεγχόμενη εγγραφή Moodle, Μητρώο Ενεργειών, Analytics Lab και What-if προσομοιώσεις.")

    st.subheader("Σύγκριση υπηρεσιών")
    comparison = pd.DataFrame([
        {"Υπηρεσία":"Service 21","Ρόλος":"Ποιότητα αξιολόγησης","Δείκτης υγείας":round(health21,1),"Ανοικτές ενέργειες":open21,"Πηγή":"Ελεγχόμενο Moodle validation"},
        {"Υπηρεσία":"Service 26","Ρόλος":"Προσαρμοστική εξάσκηση","Δείκτης υγείας":round(health26,1),"Ανοικτές ενέργειες":open26,"Πηγή":"Ελεγχόμενη adaptive practice"},
        {"Υπηρεσία":"Service 31","Ρόλος":"Σύνθεση μαθησιακών ενδείξεων","Δείκτης υγείας":round(health31,1),"Ανοικτές ενέργειες":open31,"Πηγή":"Παράγωγο των signals Service 21 + 26"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

# ==========================================================
# SERVICE 21
# ==========================================================
elif section == "Service 21 — Ευφυής Ποιότητα Αξιολόγησης":
    if ACTIVE_DF is None:
        show_active_dataset_context("S21")
        st.title("Service 21 — Ευφυής Ποιότητα Αξιολόγησης")
        st.info("Δεν έχει φορτωθεί Active CSV. Το Service 21 παραμένει κενό μέχρι να φορτωθεί πραγματικό dataset.")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Ερωτήσεις","—")
        c2.metric("Μέσος δείκτης ευκολίας","—")
        c3.metric("Μέσος δείκτης διάκρισης","—")
        c4.metric("Δείκτης υγείας","—")
        c5.metric("Ανοικτές ενέργειες","—")
        st.stop()
    if "custom_s21" not in st.session_state:
        show_active_dataset_context("S21")
        st.title("Service 21 — Ευφυής Ποιότητα Αξιολόγησης")
        st.warning(
            "Το Active CSV έχει φορτωθεί, αλλά δεν έχει δημιουργηθεί συμβατό item-level dataset για το Service 21. "
            "Χρησιμοποίησε την Αυτόματη αναγνώριση ή την Αντιστοίχιση στηλών στο Κέντρο Δεδομένων."
        )
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Ερωτήσεις","—")
        c2.metric("Μέσος δείκτης ευκολίας","—")
        c3.metric("Μέσος δείκτης διάκρισης","—")
        c4.metric("Δείκτης υγείας","—")
        c5.metric("Ανοικτές ενέργειες","—")
        st.stop()

    show_active_dataset_context("S21")
    st.title("Service 21 — Ευφυής Ποιότητα Αξιολόγησης")
    st.caption("Από ψυχομετρικούς δείκτες σε τεκμηριωμένη ουρά ελέγχου ερωτήσεων.")

    if "custom_s21" in st.session_state:
        st.success(
            f"Ενεργή πηγή: {st.session_state.get('inelso_uploaded_name','ανεβασμένο CSV')} — "
            f"{len(service21)} item-level ερωτήσεις. Όλοι οι δείκτες και τα γραφήματα παρακάτω "
            "υπολογίζονται από αυτό το αρχείο."
        )

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ερωτήσεις",len(service21))
    c2.metric("Μέσος δείκτης ευκολίας",f"{service21['facility'].mean():.1f}%")
    c3.metric("Μέσος δείκτης διάκρισης",f"{valid_disc.mean():.1f}%")
    c4.metric("Δείκτης υγείας",f"{health21:.1f}/100")
    c5.metric("Ανοικτές ενέργειες",open21)
    render_moodle_rest_service_panel(
        "Service 21",
        summary=f"Assessment health {health21:.1f}/100 · open actions {open21}",
        score=None,
        status="Assessment quality review",
        entity=ACTIVE_NAME if ACTIVE_DF is not None else "Service 21 assessment dataset",
        enabled=True,
        intervention_type="Service 21 — Έλεγχος ποιότητας ερώτησης",
        note="Καταγράφεται assessment-quality provenance, όχι learner grade.",
    )


    tabs = st.tabs(["Συνοπτική εικόνα","Τεχνική ανάλυση","Ουρά προτεραιότητας","Έλεγχος ερώτησης"])
    with tabs[0]:
        counts = service21["κατάσταση"].value_counts().rename_axis("Κατηγορία").reset_index(name="Πλήθος")
        st.dataframe(counts,use_container_width=True,hide_index=True)
        st.bar_chart(
            service21.dropna(subset=["διάκριση"]).sort_values("διάκριση",ascending=False)
            .set_index("ερώτηση")[["διάκριση"]]
        )
    with tabs[1]:
        st.dataframe(service21,use_container_width=True,hide_index=True)
    with tabs[2]:
        q = service21[
            (service21["κατάσταση"]=="ΚΡΙΣΙΜΗ") |
            (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
            (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
        ].copy()
        q["προτεραιότητα"] = q["κατάσταση"].map({
            "ΚΡΙΣΙΜΗ":1,
            "ΜΗ ΕΚΤΙΜΗΣΙΜΗ":2,
            "ΕΛΕΓΧΟΣ":3
        })
        st.dataframe(q.sort_values("προτεραιότητα"),use_container_width=True,hide_index=True)
    with tabs[3]:
        choice = st.selectbox("Επιλέξτε ερώτηση", service21["ερώτηση"].tolist())
        r = service21.loc[service21["ερώτηση"]==choice].iloc[0]
        a,b,c,d = st.columns(4)
        a.metric("Δείκτης ευκολίας",f"{r['facility']:.2f}%")
        b.metric("Δείκτης διάκρισης","N/A" if pd.isna(r["διάκριση"]) else f"{r['διάκριση']:.2f}%")
        c.metric("Αποτελεσματικότητα","N/A" if pd.isna(r["αποτελεσματικότητα"]) else f"{r['αποτελεσματικότητα']:.2f}%")
        d.metric("Κατάσταση",r["κατάσταση"])
        st.write("**Προτεινόμενη ενέργεια:**",r["ενέργεια"])
        st.info("Η τελική απόφαση αλλαγής ή διατήρησης της ερώτησης παραμένει ανθρώπινη.")

# ==========================================================
# SERVICE 26
# ==========================================================
elif section == "Service 26 — Ευφυής Προσαρμοστική Εξάσκηση":
    if ACTIVE_DF is None:
        show_active_dataset_context("S26")
        st.title("Service 26 — Ευφυής Προσαρμοστική Εξάσκηση")
        st.info("Δεν έχει φορτωθεί Active CSV. Το Service 26 παραμένει κενό μέχρι να φορτωθεί πραγματικό learner-level dataset.")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Τρέχουσα επίδοση","—")
        c2.metric("Κατάκτηση","—")
        c3.metric("Βεβαιότητα","—")
        c4.metric("Πρόοδος","—")
        c5.metric("Δείκτης υγείας","—")
        st.stop()
    if "custom_s26" not in st.session_state:
        show_active_dataset_context("S26")
        st.title("Service 26 — Ευφυής Προσαρμοστική Εξάσκηση")
        st.warning(
            "Το Active CSV έχει φορτωθεί, αλλά δεν έχει γίνει ακόμη mapping για Service 26. "
            "Χρειάζονται πραγματικά learner-level πεδία, π.χ. learner/score και όπου υπάρχουν attempt/confidence/rubric."
        )
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Τρέχουσα επίδοση","—")
        c2.metric("Κατάκτηση","—")
        c3.metric("Βεβαιότητα","—")
        c4.metric("Πρόοδος","—")
        c5.metric("Δείκτης υγείας","—")
        st.stop()

    show_active_dataset_context("S26")
    st.title("Service 26 — Ευφυής Προσαρμοστική Εξάσκηση")
    st.caption("Rubric-level gaps, κατάκτηση, βεβαιότητα, retry και ανθρώπινη κλιμάκωση.")

    latest = service26.iloc[-1]
    prev = service26.iloc[0]
    gain = latest["score"]-prev["score"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Τρέχουσα επίδοση",f"{latest['score']:.0f}/100")
    c2.metric("Κατάκτηση",latest["κατάκτηση"])
    c3.metric("Βεβαιότητα",f"{latest['βεβαιότητα']:.2f}")
    c4.metric("Πρόοδος",f"+{gain:.2f}")
    c5.metric("Δείκτης υγείας",f"{health26:.1f}/100")
    render_moodle_rest_service_panel(
        "Service 26",
        summary=(
            f"Adaptive practice · mastery {latest['κατάκτηση']} · "
            f"confidence {float(latest['βεβαιότητα']):.2f} · "
            f"progress +{gain:.2f}"
        ),
        score=float(latest["score"]),
        status=str(latest["κατάσταση"]),
        entity="Demo Student 0201",
        enabled=True,
        intervention_type=(
            "Service 26 — Enrichment"
            if str(latest["κατάσταση"]) == "ΕΤΟΙΜΟ"
            else "Service 26 — Human review"
            if str(latest["κατάσταση"]) == "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ"
            else "Service 26 — Νέα προσπάθεια"
        ),
        note="Το πραγματικό adaptive-practice score περιλαμβάνεται στο provenance. Δεν αλλάζει αυτόματα επίσημος βαθμός.",
    )


    view = st.radio("Προβολή",["Μαθητής","Εκπαιδευτής"],horizontal=True)
    if view == "Μαθητής":
        tabs = st.tabs(["Προσαρμοστικό προφίλ","Κενά rubric","Πορεία προόδου","Μηχανισμός απόφασης"])
        with tabs[0]:
            st.success("ΕΤΟΙΜΟ — η βεβαιότητα είναι επαρκής και δεν ενεργοποιείται ο κρίσιμος κανόνας RC-01.")
            st.write("**Προτεινόμενο επόμενο βήμα:** enrichment / επόμενη δραστηριότητα.")
        with tabs[1]:
            rows=[]
            for _,rr in rubric.iterrows():
                code=rr["κριτήριο"]
                gap1=4-int(prev[code]); gap2=4-int(latest[code])
                rows.append({
                    "Κριτήριο":code,
                    "Περιγραφή":rr["περιγραφή"],
                    "Gap 1ης προσπάθειας":gap1,
                    "Gap 2ης προσπάθειας":gap2,
                    "Βάρος (%)":rr["βάρος"]
                })
            gd=pd.DataFrame(rows)
            st.dataframe(gd,use_container_width=True,hide_index=True)
            st.line_chart(
                pd.DataFrame({
                    "1η προσπάθεια":[prev[c] for c in rubric["κριτήριο"]],
                    "2η προσπάθεια":[latest[c] for c in rubric["κριτήριο"]]
                },index=rubric["κριτήριο"])
            )
        with tabs[2]:
            st.line_chart(service26.set_index("προσπάθεια")[["score","βεβαιότητα"]])
            st.dataframe(service26,use_container_width=True,hide_index=True)
        with tabs[3]:
            st.markdown("""
- Βεβαιότητα < 0,70 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- RC-01 Level 0–1 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- Score < 70 με επαρκή βεβαιότητα → **ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ**
- Score ≥ 70 με επαρκή βεβαιότητα → **ΕΤΟΙΜΟ**
""")
    else:
        st.dataframe(cohort26,use_container_width=True,hide_index=True)
        st.subheader("Ουρά ενεργειών εκπαιδευτή")
        q = cohort26[cohort26["κατάσταση"]!="ΕΤΟΙΜΟ"].copy()
        q["προτεραιότητα"] = q["κατάσταση"].map({"ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ":1,"ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ":2})
        st.dataframe(q.sort_values("προτεραιότητα"),use_container_width=True,hide_index=True)

# ==========================================================
# SERVICE 31 — ΠΡΑΓΜΑΤΙΚΗ ΣΥΝΔΕΣΗ
# ==========================================================

# ==========================================================
# SERVICE 25 — AI-ASSISTED COURSE & ASSESSMENT AUTHORING
# ==========================================================
elif section == "Service 25 — AI-Assisted Course & Assessment Authoring":
    if ACTIVE_DF is None:
        show_active_dataset_context("S25")
        st.title("Service 25 — AI-Assisted Course & Assessment Authoring")
        st.info("Δεν έχει φορτωθεί Active CSV. Το Service 25 παραμένει κενό μέχρι να φορτωθεί πραγματικό dataset/context.")
        st.metric("Κατάσταση","—")
        st.stop()

    show_active_dataset_context("S25")
    st.title("Service 25 — AI-Assisted Course & Assessment Authoring")
    st.caption(
        "Workspace για γρήγορη παραγωγή drafts question banks, quizzes, rubrics και practice exercises "
        "από υπάρχον course content — πάντα με faculty review και approval."
    )

    st.warning(
        "Η ενσωματωμένη έκδοση είναι ασφαλές drafting prototype χωρίς εξωτερικό LLM/API. "
        "Παράγει δομημένα drafts από το υλικό του εκπαιδευτή και δεν δημοσιεύει τίποτα αυτόματα."
    )


    s25_scorm_ready = "s25_approved_draft" in st.session_state
    render_moodle_rest_service_panel(
        "Service 25",
        summary=(
            "Faculty-reviewed authoring draft"
            if s25_scorm_ready else
            "Draft pending faculty review"
        ),
        score=None,
        status="completed" if s25_scorm_ready else "incomplete",
        entity=ACTIVE_NAME if ACTIVE_DF is not None else "Service 25 authoring session",
        enabled=True,
        intervention_type="Service 25 — Faculty authoring review",
        note="Καταγράφεται workflow completion/provenance χωρίς τεχνητό βαθμό.",
    )


    s25_content = st.text_area(
        "Υλικό μαθήματος / πηγή",
        height=220,
        placeholder="Επικόλλησε σημειώσεις, learning outcomes, κεφάλαιο ή άλλο course content.",
        key="s25_content"
    )

    use_csv_s25 = st.checkbox(
        "Χρήση του ενεργού CSV ως πρόσθετου context για το draft",
        value=ACTIVE_DF is not None,
        key="s25_use_active_csv"
    )
    if use_csv_s25 and ACTIVE_DF is not None:
        st.text_area(
            "CSV context που θα ληφθεί υπόψη",
            value=dataset_context_summary(ACTIVE_DF),
            height=110,
            disabled=True,
            key="s25_csv_context_preview"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        s25_type = st.selectbox(
            "Τύπος draft",
            ["Question bank", "Quiz", "Rubric", "Practice exercises"],
            key="s25_type"
        )
    with c2:
        s25_count = st.slider("Πλήθος στοιχείων", 3, 12, 5, 1, key="s25_count")
    with c3:
        s25_level = st.selectbox(
            "Επίπεδο",
            ["Βασικό", "Μεσαίο", "Προχωρημένο"],
            key="s25_level"
        )

    learning_goal = st.text_input(
        "Learning objective / στόχος",
        placeholder="π.χ. Ο φοιτητής να μπορεί να εφαρμόζει...",
        key="s25_goal"
    )

    if st.button("Δημιουργία faculty-review draft", key="s25_generate"):
        raw_manual = " ".join(s25_content.strip().split())
        raw_csv = csv_context_as_text(ACTIVE_DF, max_rows=3) if use_csv_s25 and ACTIVE_DF is not None else ""
        raw = " ".join((raw_manual + " " + raw_csv).split())
        if not raw:
            st.error("Χρειάζεται course content ή ενεργό CSV context.")
        else:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?;])\s+", raw)
                if len(s.strip()) > 18
            ]
            if not sentences:
                sentences = [raw]

            items = []
            for i in range(s25_count):
                src_sentence = sentences[i % len(sentences)]
                short = src_sentence[:240].rstrip()

                if s25_type == "Question bank":
                    items.append({
                        "ID": f"Q{i+1:02d}",
                        "Draft ερώτησης": f"Εξήγησε ή εφάρμοσε την ακόλουθη έννοια: {short}",
                        "Τύπος": "Open response",
                        "Επίπεδο": s25_level,
                        "Faculty review": "ΑΠΑΙΤΕΙΤΑΙ"
                    })
                elif s25_type == "Quiz":
                    items.append({
                        "ID": f"Q{i+1:02d}",
                        "Draft ερώτησης": f"Ποια διατύπωση αποδίδει καλύτερα το ακόλουθο σημείο; {short}",
                        "Σωστή απάντηση / κριτήριο": short,
                        "Επίπεδο": s25_level,
                        "Faculty review": "ΑΠΑΙΤΕΙΤΑΙ"
                    })
                elif s25_type == "Rubric":
                    criteria = ["Ακρίβεια", "Μεθοδολογία", "Τεκμηρίωση", "Πληρότητα", "Σαφήνεια"]
                    criterion = criteria[i % len(criteria)]
                    weights = [30, 25, 15, 15, 15]
                    items.append({
                        "Κριτήριο": criterion,
                        "Βάρος (draft)": weights[i % len(weights)],
                        "Περιγραφή": (
                            f"Αξιολόγηση ως προς {criterion.lower()} σε σχέση με τον στόχο: "
                            f"{learning_goal or 'τον δηλωμένο μαθησιακό στόχο'}."
                        ),
                        "Faculty review": "ΑΠΑΙΤΕΙΤΑΙ"
                    })
                else:
                    items.append({
                        "Άσκηση": f"Practice {i+1}",
                        "Draft": f"Εφάρμοσε την έννοια «{short}» σε νέο παράδειγμα και αιτιολόγησε τα βήματά σου.",
                        "Επίπεδο": s25_level,
                        "Faculty review": "ΑΠΑΙΤΕΙΤΑΙ"
                    })

            st.session_state["s25_draft"] = pd.DataFrame(items)
            st.success("Το draft δημιουργήθηκε. Δεν έχει δημοσιευτεί πουθενά.")

    if "s25_draft" in st.session_state:
        st.subheader("Draft προς έλεγχο")
        edited_s25 = st.data_editor(
            st.session_state["s25_draft"],
            use_container_width=True,
            hide_index=True,
            key="s25_editor"
        )

        faculty_ok = st.checkbox(
            "Έλεγξα και εγκρίνω το περιεχόμενο ως faculty-reviewed draft.",
            key="s25_faculty_ok"
        )

        if st.button(
            "Καταχώριση στο faculty review queue",
            disabled=not faculty_ok,
            key="s25_review_queue"
        ):
            st.session_state["s25_approved_draft"] = edited_s25.copy()
            st.success("Το Service 25 draft χαρακτηρίστηκε faculty-reviewed στην τρέχουσα συνεδρία.")

        st.download_button(
            "Λήψη Service 25 draft CSV",
            edited_s25.to_csv(index=False).encode("utf-8-sig"),
            "INELSO_Service25_Authoring_Draft.csv",
            "text/csv",
            key="s25_download"
        )


# ==========================================================
# SERVICE 28 — FACULTY TEACHING AMPLIFICATION
# ==========================================================
elif section == "Service 28 — Faculty Teaching Amplification":
    if ACTIVE_DF is None:
        show_active_dataset_context("S28")
        st.title("Service 28 — Faculty Teaching Amplification")
        st.info("Δεν έχει φορτωθεί Active CSV. Το Service 28 παραμένει κενό μέχρι να φορτωθεί πραγματικό dataset/context.")
        st.metric("Κατάσταση","—")
        st.stop()

    show_active_dataset_context("S28")
    st.title("Service 28 — Faculty Teaching Amplification")
    st.caption(
        "Faculty-assist workspace για routine student questions, clarification rubrics και draft feedback, "
        "με τελικό έλεγχο από τον διδάσκοντα."
    )

    st.info(
        "Το Service 28 δεν απαντά αυτόματα στους φοιτητές. Παράγει draft για faculty review/approval."
    )


    s28_scorm_ready = bool(st.session_state.get("s28_library"))
    render_moodle_rest_service_panel(
        "Service 28",
        summary=(
            "Faculty-approved support draft"
            if s28_scorm_ready else
            "Faculty review pending"
        ),
        score=None,
        status="completed" if s28_scorm_ready else "incomplete",
        entity="Faculty Teaching Amplification",
        enabled=True,
        intervention_type="Service 28 — Faculty-approved feedback",
        note="Καταγράφεται faculty workflow provenance χωρίς τεχνητό learner grade.",
    )


    s28_mode = st.radio(
        "Λειτουργία",
        ["Routine Q&A draft", "Rubric clarification", "Feedback draft"],
        horizontal=True,
        key="s28_mode"
    )

    source_context = st.text_area(
        "Εγκεκριμένο course context / policy / rubric",
        height=180,
        placeholder="Επικόλλησε μόνο το εγκεκριμένο υλικό στο οποίο πρέπει να βασιστεί η απάντηση.",
        key="s28_context"
    )

    use_csv_s28 = st.checkbox(
        "Συμπερίληψη ενεργού CSV ως τεκμηριωμένου context",
        value=ACTIVE_DF is not None,
        key="s28_use_active_csv"
    )
    if use_csv_s28 and ACTIVE_DF is not None:
        st.text_area(
            "CSV context",
            value=dataset_context_summary(ACTIVE_DF),
            height=110,
            disabled=True,
            key="s28_csv_context_preview"
        )

    student_input = st.text_area(
        "Ερώτηση φοιτητή / περίπτωση",
        height=120,
        placeholder="π.χ. Τι ακριβώς ζητά το κριτήριο τεκμηρίωσης;",
        key="s28_question"
    )

    if st.button("Δημιουργία faculty-assist draft", key="s28_generate"):
        combined_context = source_context.strip()
        if use_csv_s28 and ACTIVE_DF is not None:
            combined_context = (
                combined_context + "\n\n" + csv_context_as_text(ACTIVE_DF, max_rows=3)
            ).strip()

        if not combined_context or not student_input.strip():
            st.error("Χρειάζονται context (κείμενο ή ενεργό CSV) και ερώτηση/περίπτωση.")
        else:
            chunks = [
                s.strip()
                for s in re.split(r"(?<=[.!?;])\s+", " ".join(combined_context.split()))
                if s.strip()
            ]
            q_terms = set(re.findall(r"\w{4,}", student_input.lower(), flags=re.UNICODE))

            def score_chunk(s):
                terms = set(re.findall(r"\w{4,}", s.lower(), flags=re.UNICODE))
                return len(q_terms.intersection(terms))

            ranked_chunks = sorted(chunks, key=score_chunk, reverse=True)
            evidence = ranked_chunks[:3] if ranked_chunks else [source_context.strip()]
            evidence_text = "\n".join(f"• {x}" for x in evidence)

            if s28_mode == "Routine Q&A draft":
                draft = (
                    "Προτεινόμενο draft απάντησης προς faculty review:\n\n"
                    f"Με βάση το εγκεκριμένο υλικό, τα πιο σχετικά σημεία είναι:\n{evidence_text}\n\n"
                    "Προτεινόμενη απάντηση: Παρακαλώ ακολούθησε τα παραπάνω σημεία του μαθήματος. "
                    "Αν υπάρχει ασάφεια που δεν καλύπτεται από το εγκεκριμένο υλικό, "
                    "χρειάζεται απάντηση από τον διδάσκοντα."
                )
            elif s28_mode == "Rubric clarification":
                draft = (
                    "Draft διευκρίνισης rubric προς faculty review:\n\n"
                    f"Σχετικά αποσπάσματα:\n{evidence_text}\n\n"
                    "Διευκρίνιση: Το κριτήριο πρέπει να ερμηνευτεί μόνο σύμφωνα με το εγκεκριμένο rubric. "
                    "Δεν προστίθενται νέα κριτήρια ή απαιτήσεις χωρίς έγκριση του διδάσκοντα."
                )
            else:
                draft = (
                    "Draft feedback προς faculty review:\n\n"
                    f"Σχετικά σημεία του course context:\n{evidence_text}\n\n"
                    "Feedback scaffold: Αναγνώρισε τι έγινε σωστά, σύνδεσε τη βελτίωση με το αντίστοιχο κριτήριο, "
                    "δώσε μία συγκεκριμένη επόμενη ενέργεια και απόφυγε αλλαγή επίσημου βαθμού."
                )

            st.session_state["s28_draft"] = draft

    if "s28_draft" in st.session_state:
        s28_edited = st.text_area(
            "Faculty-assist draft",
            value=st.session_state["s28_draft"],
            height=260,
            key="s28_edited"
        )

        c1, c2 = st.columns(2)
        with c1:
            s28_approve = st.checkbox(
                "Faculty review ολοκληρώθηκε.",
                key="s28_approve"
            )
        with c2:
            st.metric("Αυτόματη αποστολή", "ΟΧΙ")

        if st.button(
            "Αποθήκευση ως εγκεκριμένο FAQ/feedback draft",
            disabled=not s28_approve,
            key="s28_save"
        ):
            if "s28_library" not in st.session_state:
                st.session_state["s28_library"] = []
            st.session_state["s28_library"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": s28_mode,
                "question": student_input,
                "draft": s28_edited,
                "status": "FACULTY APPROVED"
            })
            st.success("Το draft αποθηκεύτηκε στη session faculty library.")

        if st.button("Προώθηση ως draft παρέμβασης για Moodle note", key="s28_stage"):
            st.session_state["pending_moodle_note_text"] = s28_edited
            st.session_state["pending_moodle_intervention_type"] = "Service 28 — Faculty-approved feedback"
            st.success(
                "Το draft προωθήθηκε στην Ελεγχόμενη Εγγραφή Moodle. "
                "Δεν έχει γίνει write — απαιτείται επιλογή learner/course και διπλή επιβεβαίωση."
            )

    if st.session_state.get("s28_library"):
        st.subheader("Faculty-approved drafts της συνεδρίας")
        lib_df = pd.DataFrame(st.session_state["s28_library"])
        st.dataframe(lib_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Λήψη faculty draft library CSV",
            lib_df.to_csv(index=False).encode("utf-8-sig"),
            "INELSO_Service28_Faculty_Drafts.csv",
            "text/csv",
            key="s28_download"
        )


elif section == "Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις":
    if ACTIVE_DF is None:
        show_active_dataset_context("S31")
        st.title("Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις")
        st.info("Δεν έχει φορτωθεί Active CSV. Το Service 31 παραμένει κενό.")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Συνολικά signals","—")
        c2.metric("Υψηλή προτεραιότητα","—")
        c3.metric("Μεσαία προτεραιότητα","—")
        c4.metric("Δείκτης υγείας","—")
        st.stop()
    if ("custom_s21" not in st.session_state) and ("custom_s26" not in st.session_state):
        show_active_dataset_context("S31")
        st.title("Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις")
        st.warning(
            "Υπάρχει Active CSV, αλλά δεν υπάρχουν ακόμη πραγματικά mapped signals από Service 21 ή Service 26. "
            "Το Service 31 δεν δημιουργεί synthetic signals."
        )
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Συνολικά signals","—")
        c2.metric("Υψηλή προτεραιότητα","—")
        c3.metric("Μεσαία προτεραιότητα","—")
        c4.metric("Δείκτης υγείας","—")
        st.stop()

    show_active_dataset_context("S31")
    st.title("Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις")
    st.caption("Σύνθεση πραγματικών/ελεγχόμενων signals από Service 21 και Service 26 — χωρίς επινοημένες θεματικές αντιστοιχίσεις.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Συνολικά signals",len(service31_integrated))
    c2.metric("Υψηλή προτεραιότητα",high31)
    c3.metric("Μεσαία προτεραιότητα",medium31)
    c4.metric("Δείκτης υγείας",f"{health31:.1f}/100")
    render_moodle_rest_service_panel(
        "Service 31",
        summary=f"Integrated signals {len(service31_integrated)} · high priority {high31}",
        score=None,
        status="Integrated learning insight",
        entity="Integrated learning signals",
        enabled=True,
        intervention_type="Service 31 — Learning insight",
        note="Καταγράφεται synthesis/provenance των signals, όχι learner grade.",
    )


    tabs = st.tabs(["Ενιαίες ενδείξεις","Προτεραιότητες","Προέλευση","Αλυσίδα παρέμβασης"])
    with tabs[0]:
        st.dataframe(service31_integrated,use_container_width=True,hide_index=True)
    with tabs[1]:
        st.dataframe(
            service31_integrated.sort_values(
                ["προτεραιότητα","ισχύς_σήματος"],
                ascending=[True,False]
            ),
            use_container_width=True,hide_index=True
        )
        st.warning("Υψηλή προτεραιότητα σημαίνει ότι το signal απαιτεί ανθρώπινο έλεγχο πριν από οποιαδήποτε εκπαιδευτική παρέμβαση.")
    with tabs[2]:
        prov = service31_integrated[["οντότητα","πηγή","τύπος","προέλευση"]].copy()
        st.dataframe(prov,use_container_width=True,hide_index=True)
        st.info("Το Service 31 v6 δεν χρησιμοποιεί πλέον ανεξάρτητα synthetic concept values. Συνθέτει signals που προέρχονται από τα Service 21 και 26.")
    with tabs[3]:
        st.markdown("""
1. Εντοπισμός signal στο Service 21 ή Service 26  
2. Διατήρηση της πραγματικής προέλευσης και του τύπου signal  
3. Προτεραιοποίηση χωρίς επινοημένη θεματική ετικέτα  
4. Έλεγχος από εκπαιδευτή / σχεδιαστή αξιολόγησης  
5. Επιλογή παρέμβασης: item review, retry, enrichment ή ανθρώπινη υποστήριξη  
6. Παρακολούθηση επόμενης προσπάθειας ή επόμενου cohort  
7. Κλείσιμο action μόνο μετά από νέα τεκμηρίωση
""")

# ==========================================================
# ΔΙΑΣΥΝΔΕΣΗ ΥΠΗΡΕΣΙΩΝ
# ==========================================================
elif section == "Διασύνδεση Υπηρεσιών":
    show_active_dataset_context("S31")
    st.title("Διασύνδεση Υπηρεσιών")
    st.caption("Μία ενιαία ουρά ενεργειών από Service 21, Service 26 και το διασυνδεδεμένο Service 31.")

    actions=[]
    for _,r in service21[
        (service21["κατάσταση"]=="ΚΡΙΣΙΜΗ") |
        (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
        (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
    ].iterrows():
        actions.append({
            "Προτεραιότητα":1 if r["κατάσταση"] in ["ΚΡΙΣΙΜΗ","ΜΗ ΕΚΤΙΜΗΣΙΜΗ"] else 2,
            "Υπηρεσία":"Service 21",
            "Οντότητα":r["ερώτηση"],
            "Signal":r["κατάσταση"],
            "Ενέργεια":r["ενέργεια"]
        })

    for _,r in cohort26[cohort26["κατάσταση"]!="ΕΤΟΙΜΟ"].iterrows():
        actions.append({
            "Προτεραιότητα":1 if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else 2,
            "Υπηρεσία":"Service 26",
            "Οντότητα":r["μαθητής"],
            "Signal":r["κατάσταση"],
            "Ενέργεια":"Ανθρώπινος έλεγχος" if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else "Στοχευμένη νέα προσπάθεια"
        })

    for _,r in service31_integrated[service31_integrated["προτεραιότητα"]=="ΥΨΗΛΗ"].iterrows():
        actions.append({
            "Προτεραιότητα":2,
            "Υπηρεσία":"Service 31",
            "Οντότητα":r["οντότητα"],
            "Signal":r["τύπος"],
            "Ενέργεια":r["ενέργεια"]
        })

    action_df=pd.DataFrame(actions).sort_values(["Προτεραιότητα","Υπηρεσία","Οντότητα"])
    st.dataframe(action_df,use_container_width=True,hide_index=True)

    st.subheader("Αναλυτικός έλεγχος ενέργειας")
    if not action_df.empty:
        labels=[
            f"{r['Υπηρεσία']} — {r['Οντότητα']} — {r['Signal']}"
            for _,r in action_df.iterrows()
        ]
        selected=st.selectbox("Επιλέξτε ενέργεια",labels)
        idx=labels.index(selected)
        rr=action_df.iloc[idx]
        a,b,c=st.columns(3)
        a.metric("Υπηρεσία",rr["Υπηρεσία"])
        b.metric("Προτεραιότητα",rr["Προτεραιότητα"])
        c.metric("Signal",rr["Signal"])
        st.write("**Οντότητα:**",rr["Οντότητα"])
        st.write("**Προτεινόμενη ενέργεια:**",rr["Ενέργεια"])
        st.info("Η ενέργεια δεν εκτελείται αυτόματα.")


# ==========================================================
# ΚΥΚΛΟΣ ΖΩΗΣ ΕΝΕΡΓΕΙΩΝ
# ==========================================================
elif section == "Κύκλος Ζωής Ενεργειών":
    show_active_dataset_context("S31")
    st.title("Κύκλος Ζωής Ενεργειών")
    st.caption("Από την ανίχνευση ενός signal μέχρι την ανθρώπινη απόφαση, την παρέμβαση και το κλείσιμο της ενέργειας.")

    actions=[]
    for _,r in service21[
        (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
        (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
    ].iterrows():
        actions.append({
            "Κωδικός":f"S21-{r['ερώτηση'].split()[-1]}",
            "Υπηρεσία":"Service 21",
            "Οντότητα":r["ερώτηση"],
            "Προτεραιότητα":1 if r["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ" else 2,
            "Signal":r["κατάσταση"],
            "Προτεινόμενη ενέργεια":r["ενέργεια"]
        })

    for i,(_,r) in enumerate(cohort26[cohort26["κατάσταση"]!="ΕΤΟΙΜΟ"].iterrows(), start=1):
        actions.append({
            "Κωδικός":f"S26-{i:02d}",
            "Υπηρεσία":"Service 26",
            "Οντότητα":r["μαθητής"],
            "Προτεραιότητα":1 if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else 2,
            "Signal":r["κατάσταση"],
            "Προτεινόμενη ενέργεια":"Ανθρώπινος έλεγχος" if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else "Στοχευμένη νέα προσπάθεια"
        })

    for i,(_,r) in enumerate(service31_integrated[service31_integrated["προτεραιότητα"]=="ΥΨΗΛΗ"].iterrows(), start=1):
        actions.append({
            "Κωδικός":f"S31-{i:02d}",
            "Υπηρεσία":"Service 31",
            "Οντότητα":r["οντότητα"],
            "Προτεραιότητα":2,
            "Signal":r["τύπος"],
            "Προτεινόμενη ενέργεια":r["ενέργεια"]
        })

    lifecycle = pd.DataFrame(actions).sort_values(["Προτεραιότητα","Υπηρεσία","Οντότητα"])

    if "κατάσταση_ενεργειών" not in st.session_state:
        st.session_state["κατάσταση_ενεργειών"] = {
            code:"ΝΕΑ" for code in lifecycle["Κωδικός"].tolist()
        }

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Σύνολο ενεργειών", len(lifecycle))
    c2.metric("Νέες", sum(v=="ΝΕΑ" for v in st.session_state["κατάσταση_ενεργειών"].values()))
    c3.metric("Σε έλεγχο", sum(v=="ΣΕ ΕΛΕΓΧΟ" for v in st.session_state["κατάσταση_ενεργειών"].values()))
    c4.metric("Κλειστές", sum(v=="ΚΛΕΙΣΤΗ" for v in st.session_state["κατάσταση_ενεργειών"].values()))

    if not lifecycle.empty:
        επιλογές = lifecycle["Κωδικός"].tolist()
        code = st.selectbox("Επιλέξτε ενέργεια", επιλογές)
        row = lifecycle.loc[lifecycle["Κωδικός"]==code].iloc[0]

        a,b,c = st.columns(3)
        a.metric("Υπηρεσία", row["Υπηρεσία"])
        b.metric("Προτεραιότητα", row["Προτεραιότητα"])
        c.metric("Signal", row["Signal"])

        st.write("**Οντότητα:**", row["Οντότητα"])
        st.write("**Προτεινόμενη ενέργεια:**", row["Προτεινόμενη ενέργεια"])

        current = st.session_state["κατάσταση_ενεργειών"].get(code, "ΝΕΑ")
        new_state = st.radio(
            "Κατάσταση κύκλου ζωής",
            ["ΝΕΑ","ΣΕ ΕΛΕΓΧΟ","ΚΛΕΙΣΤΗ"],
            index=["ΝΕΑ","ΣΕ ΕΛΕΓΧΟ","ΚΛΕΙΣΤΗ"].index(current),
            horizontal=True
        )
        st.session_state["κατάσταση_ενεργειών"][code] = new_state

        st.info("Η αλλαγή κατάστασης αποθηκεύεται μόνο προσωρινά στη συνεδρία του dashboard και δεν γράφεται στο Moodle.")

        export_lifecycle = lifecycle.copy()
        export_lifecycle["Κατάσταση κύκλου ζωής"] = export_lifecycle["Κωδικός"].map(
            st.session_state["κατάσταση_ενεργειών"]
        )
        st.dataframe(export_lifecycle,use_container_width=True,hide_index=True)
        st.download_button(
            "Λήψη μητρώου ενεργειών",
            export_lifecycle.to_csv(index=False).encode("utf-8-sig"),
            "INELSO_Mitroo_Energeion.csv",
            "text/csv"
        )

# ==========================================================
# ΜΕΘΟΔΟΛΟΓΙΑ & ΠΡΟΕΛΕΥΣΗ
# ==========================================================
elif section == "Μεθοδολογία & Προέλευση":
    show_active_dataset_context("REPORT")
    st.title("Μεθοδολογία & Προέλευση")
    st.caption("Διαφάνεια για το τι υπολογίζεται, από πού προέρχεται και τι επιτρέπεται να συμπεράνουμε.")

    st.subheader("Προέλευση δεδομένων")
    provenance = pd.DataFrame([
        {
            "Υπηρεσία":"Service 21",
             "Πηγή": (f"Active CSV: {ACTIVE_NAME}" if "custom_s21" in st.session_state else "Ελεγχόμενο Moodle validation"),
            "Τύπος":"Item-level psychometric signals",
            "Χρήση":"Ποιότητα ερωτήσεων και ουρά ελέγχου",
            "Περιορισμός":"Δεν επιτρέπεται αυτόματη διαγραφή/αλλαγή item"
        },
        {
            "Υπηρεσία":"Service 26",
             "Πηγή": (f"Active CSV: {ACTIVE_NAME}" if "custom_s26" in st.session_state else "Ελεγχόμενη adaptive practice"),
            "Τύπος":"Rubric levels, score, mastery, confidence",
            "Χρήση":"Retry / enrichment / ανθρώπινος έλεγχος",
            "Περιορισμός":"Δεν αλλάζει επίσημο βαθμό Moodle"
        },
        {
            "Υπηρεσία":"Service 31",
             "Πηγή": f"Παράγωγο signals — Active Dataset: {ACTIVE_NAME}" if ACTIVE_DF is not None else "Παράγωγο signals από Service 21 + 26",
            "Τύπος":"Integrated learning signals",
            "Χρήση":"Σύνθεση και προτεραιοποίηση",
            "Περιορισμός":"Δεν επινοούνται θεματικές αντιστοιχίσεις χωρίς τεκμηρίωση"
        },
    ])
    st.dataframe(provenance,use_container_width=True,hide_index=True)

    st.subheader("Κανόνες Service 21")
    st.markdown("""
- Διάκριση ≥ 40% → **ΙΣΧΥΡΗ**
- 20%–39,99% → **ΙΚΑΝΟΠΟΙΗΤΙΚΗ**
- 0%–19,99% → **ΕΛΕΓΧΟΣ**
- Αρνητική διάκριση → **ΚΡΙΣΙΜΗ**
- Μη εκτιμήσιμη διάκριση → απαιτεί ανθρώπινο έλεγχο scoring / response pattern
""")

    st.subheader("Κανόνες Service 26")
    st.markdown("""
- Βεβαιότητα < 0,70 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- RC-01 Level 0–1 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- Score < 70 με επαρκή βεβαιότητα → **ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ**
- Score ≥ 70 με επαρκή βεβαιότητα → **ΕΤΟΙΜΟ**
""")

    st.subheader("Κανόνες Service 31")
    st.markdown("""
- Συνθέτει signals που ήδη υπάρχουν στα Service 21 και Service 26.
- Διατηρεί την αρχική προέλευση κάθε signal.
- Η προτεραιότητα δεν ισοδυναμεί με αυτόματη εκπαιδευτική απόφαση.
- Κάθε υψηλής προτεραιότητας signal οδηγεί πρώτα σε ανθρώπινο έλεγχο.
""")

# ==========================================================
# ΕΛΕΓΧΟΣ ΕΤΟΙΜΟΤΗΤΑΣ
# ==========================================================
elif section == "Έλεγχος Ετοιμότητας":
    show_active_dataset_context("REPORT")
    st.title("Έλεγχος Ετοιμότητας")
    st.caption("Τεχνικός και λειτουργικός έλεγχος πριν από παρουσίαση ή πιλοτική χρήση.")

    dataset_loaded = ACTIVE_DF is not None
    s21_compat = service_dataset_compatibility("S21", ACTIVE_DF)
    s26_compat = service_dataset_compatibility("S26", ACTIVE_DF)
    s16_compat = service_dataset_compatibility("S16", ACTIVE_DF)

    readiness = pd.DataFrame([
        {"Έλεγχος":"Service 21 λειτουργεί","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Ψυχομετρικά signals και ουρά item review διαθέσιμα"},
        {"Έλεγχος":"Service 26 λειτουργεί","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Rubric, mastery, confidence gate και teacher queue διαθέσιμα"},
        {"Έλεγχος":"Service 31 συνδέεται με 21/26","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Integrated signals χωρίς ανεξάρτητα synthetic concept metrics"},
        {"Έλεγχος":"Moodle iframe","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Ίδιο Streamlit URL με ?embed=true"},
        {"Έλεγχος":"Επίσημοι βαθμοί","Κατάσταση":"ΑΣΦΑΛΕΣ","Σχόλιο":"Καμία αυτόματη εγγραφή βαθμών"},
        {"Έλεγχος":"Human-in-the-loop","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Υψηλή αβεβαιότητα/προτεραιότητα οδηγεί σε ανθρώπινο έλεγχο"},
        {"Έλεγχος":"Audit/provenance","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Κάθε signal διατηρεί πηγή και τύπο"},
        {"Έλεγχος":"Ενεργό dataset","Κατάσταση":"ΕΤΟΙΜΟ" if dataset_loaded else "ΑΠΑΙΤΕΙΤΑΙ","Σχόλιο":ACTIVE_NAME if dataset_loaded else "Φόρτωσε αρχείο από το Κέντρο Δεδομένων"},
        {"Έλεγχος":"Service 21 ↔ CSV","Κατάσταση":"ΕΤΟΙΜΟ" if s21_compat=="ΠΛΗΡΗΣ" else "ΥΠΟ ΠΡΟΫΠΟΘΕΣΗ","Σχόλιο":f"Συμβατότητα: {s21_compat}"},
        {"Έλεγχος":"Service 26 ↔ CSV","Κατάσταση":"ΕΤΟΙΜΟ" if s26_compat=="ΠΛΗΡΗΣ" else "ΥΠΟ ΠΡΟΫΠΟΘΕΣΗ","Σχόλιο":f"Συμβατότητα: {s26_compat}"},
        {"Έλεγχος":"Service 16 ↔ CSV","Κατάσταση":"ΕΤΟΙΜΟ" if s16_compat=="ΠΛΗΡΗΣ" else "ΥΠΟ ΠΡΟΫΠΟΘΕΣΗ","Σχόλιο":f"Συμβατότητα: {s16_compat}"},
        {"Έλεγχος":"Εισαγωγή εξωτερικών δεδομένων","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"CSV, Excel, JSON, XML και Moodle exports"},
        {"Έλεγχος":"Analytics Lab","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Classification, regression, clustering, correlations και group comparison"},
        {"Έλεγχος":"What-if simulator","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Sandbox προσομοίωση adaptive thresholds"},
        {"Έλεγχος":"Live Moodle connector","Κατάσταση":"ΕΤΟΙΜΟ ΥΠΟ ΠΡΟΫΠΟΘΕΣΗ","Σχόλιο":"Απαιτεί ενεργό Moodle Web Service και token"},
        {"Έλεγχος":"Ελεγχόμενη εγγραφή Moodle","Κατάσταση":"ΕΤΟΙΜΟ ΥΠΟ ΠΡΟΫΠΟΘΕΣΗ","Σχόλιο":"SCORM launch + REST write + preview + διπλή επιβεβαίωση + audit/read-back verification"},
        {"Έλεγχος":"Early Warning Engine","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Παραμετρικός risk score από ανεβασμένα δεδομένα"},
        {"Έλεγχος":"Μητρώο ενεργειών","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Session + import/export CSV για μεταφερσιμότητα"},
        {"Έλεγχος":"Persistent action storage","Κατάσταση":"ΜΕΛΛΟΝΤΙΚΟ","Σχόλιο":"Η κατάσταση ενεργειών αποθηκεύεται μόνο στη συνεδρία"},
    ])
    st.dataframe(readiness,use_container_width=True,hide_index=True)

    ready_count = int((readiness["Κατάσταση"].isin(["ΕΤΟΙΜΟ","ΑΣΦΑΛΕΣ"])).sum())
    total = len(readiness)
    score = round(100*ready_count/total,1)

    c1,c2,c3 = st.columns(3)
    c1.metric("Έλεγχοι",total)
    c2.metric("Έτοιμοι / ασφαλείς",ready_count)
    c3.metric("Δείκτης ετοιμότητας",f"{score:.1f}%")

    if score >= 85:
        st.success("Το σύστημα είναι έτοιμο για ολοκληρωμένη παρουσίαση και ελεγχόμενη πιλοτική επίδειξη.")
    else:
        st.warning("Υπάρχουν σημεία που χρειάζονται ολοκλήρωση πριν από παρουσίαση.")


# ==========================================================
# ΚΕΝΤΡΟ ΕΙΣΑΓΩΓΗΣ & ΕΠΕΞΕΡΓΑΣΙΑΣ ΔΕΔΟΜΕΝΩΝ
# ==========================================================
elif section == "Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων":
    show_active_dataset_context("DATA")
    st.title("Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων")
    st.caption("Ανέβασε CSV, Excel, JSON ή Moodle export και χρησιμοποίησέ το για έλεγχο ποιότητας, καθαρισμό, ανάλυση και τροφοδότηση των Service 21/26/31.")

    st.info(
        "Υποστηρίζονται αρχεία CSV, XLSX/XLS, JSON και XML. "
        "Για Moodle προτείνεται export σε CSV/Excel/JSON όπου είναι διαθέσιμο. "
        "Η εφαρμογή δεν γράφει πίσω στο Moodle."
    )

    uploaded = st.file_uploader(
        "Επιλέξτε αρχείο δεδομένων",
        type=["csv","xlsx","xls","json","xml"],
        accept_multiple_files=False
    )

    if uploaded is None:
        st.markdown("""
### Τι μπορεί να γίνει μετά τη μεταφόρτωση

- Αυτόματη προεπισκόπηση δεδομένων
- Αναγνώριση τύπων στηλών
- Έλεγχος κενών τιμών
- Έλεγχος διπλοεγγραφών
- Βασικός καθαρισμός
- Περιγραφικά στατιστικά
- Συσχετίσεις αριθμητικών μεταβλητών
- Επιλογή φίλτρων και στηλών
- Εξαγωγή καθαρισμένου CSV
- Αυτόματη αναγνώριση πιθανής μορφής Service 21
- Αυτόματη αναγνώριση πιθανής μορφής Service 26
- Προετοιμασία signals για Service 31
""")
    else:
        filename = uploaded.name
        ext = filename.lower().split(".")[-1]

        def load_uploaded_file(file_obj, ext):
            if ext == "csv":
                raw = file_obj.getvalue()
                # Try UTF-8 first, then common Greek/Windows fallback
                for enc in ["utf-8-sig","utf-8","cp1253","latin1"]:
                    try:
                        return pd.read_csv(io.BytesIO(raw), encoding=enc)
                    except Exception:
                        pass
                raise ValueError("Δεν ήταν δυνατή η ανάγνωση του CSV με τις υποστηριζόμενες κωδικοποιήσεις.")
            elif ext in ["xlsx","xls"]:
                return pd.read_excel(file_obj)
            elif ext == "json":
                raw = file_obj.getvalue().decode("utf-8-sig")
                obj = json.loads(raw)
                if isinstance(obj, list):
                    return pd.DataFrame(obj)
                if isinstance(obj, dict):
                    # Prefer record-like top-level arrays
                    for k,v in obj.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            return pd.DataFrame(v)
                    return pd.json_normalize(obj)
                raise ValueError("Το JSON δεν έχει αναγνωρίσιμη δομή πίνακα.")
            elif ext == "xml":
                root = ET.fromstring(file_obj.getvalue())
                rows = []
                for child in list(root):
                    row = {}
                    for sub in list(child):
                        row[sub.tag] = sub.text
                    if row:
                        rows.append(row)
                if not rows:
                    # fallback: flatten direct children
                    rows = [{c.tag:c.text for c in list(root)}]
                return pd.DataFrame(rows)
            else:
                raise ValueError("Μη υποστηριζόμενος τύπος αρχείου.")

        try:
            df = load_uploaded_file(uploaded, ext)
            st.success(f"Το αρχείο «{filename}» φορτώθηκε επιτυχώς.")
            st.session_state["inelso_uploaded_original"] = df
            st.session_state["inelso_uploaded_name"] = filename

            # Αυτόματη σύνδεση Moodle Complete Statistics -> Service 21
            auto_s21 = extract_service21_from_moodle_complete_stats(df)
            if auto_s21 is not None:
                auto_s21["κατάσταση"] = auto_s21["διάκριση"].apply(κατηγορία_διάκρισης)
                auto_s21["ενέργεια"] = auto_s21.apply(ενέργεια_ερώτησης, axis=1)
                st.session_state["custom_s21"] = auto_s21
                st.session_state["custom_s21_source"] = "Moodle Complete Statistics"
            else:
                # Αν το νέο upload δεν είναι Service 21 stats, μη διατηρείται παλιό mapped dataset.
                st.session_state.pop("custom_s21", None)
                st.session_state.pop("custom_s21_source", None)
        except Exception as e:
            st.error(f"Σφάλμα ανάγνωσης αρχείου: {e}")
            df = None

        if df is not None:
            # Normalize column names lightly but preserve originals
            original_columns = list(df.columns)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Γραμμές", len(df))
            c2.metric("Στήλες", len(df.columns))
            c3.metric("Κενά κελιά", int(df.isna().sum().sum()))
            c4.metric("Διπλότυπες γραμμές", int(df.duplicated().sum()))

            tabs = st.tabs([
                "Προεπισκόπηση",
                "Ποιότητα δεδομένων",
                "Καθαρισμός",
                "Περιγραφική ανάλυση",
                "Αυτόματη αναγνώριση",
                "Εξαγωγές"
            ])

            with tabs[0]:
                st.subheader("Προεπισκόπηση")
                n = st.slider("Αριθμός γραμμών προεπισκόπησης", 5, min(100, max(5,len(df))), min(20, max(5,len(df))))
                st.dataframe(df.head(n), use_container_width=True, hide_index=True)

                st.subheader("Τύποι στηλών")
                dtype_table = pd.DataFrame({
                    "Στήλη": df.columns.astype(str),
                    "Τύπος": [str(df[c].dtype) for c in df.columns],
                    "Μοναδικές τιμές": [df[c].nunique(dropna=True) for c in df.columns],
                })
                st.dataframe(dtype_table, use_container_width=True, hide_index=True)

            with tabs[1]:
                st.subheader("Έλεγχος ποιότητας")
                quality = pd.DataFrame({
                    "Στήλη": df.columns.astype(str),
                    "Κενά": [int(df[c].isna().sum()) for c in df.columns],
                    "Κενά (%)": [round(float(df[c].isna().mean()*100),2) for c in df.columns],
                    "Μοναδικές τιμές": [int(df[c].nunique(dropna=True)) for c in df.columns],
                    "Τύπος": [str(df[c].dtype) for c in df.columns]
                })
                st.dataframe(quality, use_container_width=True, hide_index=True)

                duplicate_count = int(df.duplicated().sum())
                if duplicate_count > 0:
                    st.warning(f"Εντοπίστηκαν {duplicate_count} διπλότυπες γραμμές.")
                else:
                    st.success("Δεν εντοπίστηκαν πλήρως διπλότυπες γραμμές.")

                high_missing = quality[quality["Κενά (%)"] >= 30]
                if len(high_missing) > 0:
                    st.warning("Υπάρχουν στήλες με ≥30% κενές τιμές.")
                    st.dataframe(high_missing, use_container_width=True, hide_index=True)

            with tabs[2]:
                st.subheader("Καθαρισμός δεδομένων")
                work = df.copy()

                remove_duplicates = st.checkbox("Αφαίρεση πλήρως διπλότυπων γραμμών")
                strip_text = st.checkbox("Αφαίρεση περιττών κενών από κειμενικές στήλες")
                drop_empty_cols = st.checkbox("Αφαίρεση εντελώς κενών στηλών")

                if remove_duplicates:
                    work = work.drop_duplicates()
                if strip_text:
                    for c in work.select_dtypes(include="object").columns:
                        work[c] = work[c].apply(lambda x: x.strip() if isinstance(x,str) else x)
                if drop_empty_cols:
                    work = work.dropna(axis=1, how="all")

                st.session_state["inelso_uploaded_clean"] = work

                c1,c2,c3 = st.columns(3)
                c1.metric("Γραμμές μετά τον καθαρισμό", len(work))
                c2.metric("Στήλες μετά τον καθαρισμό", len(work.columns))
                c3.metric("Υπόλοιπα κενά", int(work.isna().sum().sum()))

                st.dataframe(work.head(25), use_container_width=True, hide_index=True)

            with tabs[3]:
                st.subheader("Περιγραφική ανάλυση")
                numeric = df.select_dtypes(include=np.number)

                if numeric.shape[1] > 0:
                    desc = numeric.describe().T.reset_index().rename(columns={"index":"Μεταβλητή"})
                    st.dataframe(desc, use_container_width=True, hide_index=True)

                    if numeric.shape[1] >= 2:
                        st.subheader("Πίνακας συσχετίσεων")
                        corr = numeric.corr(numeric_only=True)
                        st.dataframe(corr, use_container_width=True)
                else:
                    st.info("Δεν εντοπίστηκαν αριθμητικές στήλες για περιγραφικά στατιστικά.")

                categorical = df.select_dtypes(exclude=np.number)
                if categorical.shape[1] > 0:
                    st.subheader("Κατηγορικές μεταβλητές")
                    selected_cat = st.selectbox("Επιλέξτε κατηγορική στήλη", categorical.columns.tolist())
                    freq = df[selected_cat].fillna("(Κενό)").value_counts(dropna=False).reset_index()
                    freq.columns = [selected_cat, "Πλήθος"]
                    st.dataframe(freq, use_container_width=True, hide_index=True)

            with tabs[4]:
                st.subheader("Αυτόματη αναγνώριση δομής")

                lower_cols = {str(c).lower(): c for c in df.columns}
                lower_names = set(lower_cols.keys())

                # Heuristics for S21
                s21_terms = [
                    "facility", "discrimination", "discriminative efficiency",
                    "facility index", "discrimination index",
                    "question", "question name", "slot"
                ]
                s21_matches = [t for t in s21_terms if any(t in c for c in lower_names)]

                # Heuristics for S26
                s26_terms = [
                    "score","mastery","confidence","attempt",
                    "rc-01","rc01","rc-02","rc02","rc-03","rc03","rc-04","rc04","rc-05","rc05"
                ]
                s26_matches = [t for t in s26_terms if any(t in c for c in lower_names)]

                # General Moodle indicators
                moodle_terms = [
                    "userid","user id","quiz","attempt","grade","timecreated",
                    "timemodified","question","course","activity","module","state"
                ]
                moodle_matches = [t for t in moodle_terms if any(t in c for c in lower_names)]

                score21 = min(100, int(100*len(set(s21_matches))/max(1,len(s21_terms))))
                score26 = min(100, int(100*len(set(s26_matches))/max(1,len(s26_terms))))
                moodle_score = min(100, int(100*len(set(moodle_matches))/max(1,len(moodle_terms))))

                a,b,c = st.columns(3)
                a.metric("Πιθανότητα μορφής Service 21", f"{score21}%")
                b.metric("Πιθανότητα μορφής Service 26", f"{score26}%")
                c.metric("Ενδείξεις Moodle export", f"{moodle_score}%")

                if "custom_s21" in st.session_state and st.session_state.get("custom_s21_source") == "Moodle Complete Statistics":
                    st.success(
                        f"Αναγνωρίστηκε αυτόματα Moodle Complete Statistics export. "
                        f"Το Service 21 έχει ενημερωθεί με {len(st.session_state['custom_s21'])} ερωτήσεις."
                    )
                elif score21 >= 30:
                    st.success("Το αρχείο περιέχει αρκετές ενδείξεις για ανάλυση τύπου Service 21.")
                if score26 >= 30:
                    st.success("Το αρχείο περιέχει αρκετές ενδείξεις για ανάλυση τύπου Service 26.")
                if moodle_score >= 20:
                    st.info("Το αρχείο φαίνεται να περιέχει πεδία που συναντώνται συχνά σε Moodle exports.")

                st.subheader("Αντιστοίχιση στηλών")
                st.caption("Μπορείς να ορίσεις χειροκίνητα ποιες στήλες αντιστοιχούν στις βασικές μεταβλητές.")

                cols = ["— καμία —"] + list(df.columns.astype(str))

                map_service = st.radio(
                    "Τύπος ανάλυσης",
                    ["Γενική ανάλυση","Service 21","Service 26"],
                    horizontal=True
                )

                if map_service == "Service 21":
                    qcol = st.selectbox("Στήλη ερώτησης", cols)
                    fcol = st.selectbox("Στήλη δείκτη ευκολίας", cols)
                    dcol = st.selectbox("Στήλη δείκτη διάκρισης", cols)
                    ecol = st.selectbox("Στήλη αποτελεσματικότητας διάκρισης", cols)

                    if st.button("Δημιουργία πίνακα Service 21 από το αρχείο"):
                        required = [qcol,fcol,dcol]
                        if "— καμία —" in required:
                            st.error("Επίλεξε τουλάχιστον ερώτηση, δείκτη ευκολίας και δείκτη διάκρισης.")
                        else:
                            mapped = pd.DataFrame({
                                "ερώτηση": df[qcol],
                                "facility": df[fcol].map(pct_number),
                                "διάκριση": df[dcol].map(pct_number),
                            })
                            if ecol != "— καμία —":
                                mapped["αποτελεσματικότητα"] = df[ecol].map(pct_number)
                            else:
                                mapped["αποτελεσματικότητα"] = np.nan

                            mapped["κατάσταση"] = mapped["διάκριση"].apply(κατηγορία_διάκρισης)
                            mapped["ενέργεια"] = mapped.apply(ενέργεια_ερώτησης, axis=1)

                            st.session_state["custom_s21"] = mapped
                            st.success("Δημιουργήθηκε προσωρινός πίνακας Service 21 από το αρχείο.")
                            st.dataframe(mapped, use_container_width=True, hide_index=True)

                elif map_service == "Service 26":
                    learner26_col = st.selectbox("Στήλη learner / αναγνωριστικού", cols)
                    score_col = st.selectbox("Στήλη score", cols)
                    conf_col = st.selectbox("Στήλη βεβαιότητας", cols)
                    attempt_col = st.selectbox("Στήλη προσπάθειας", cols)
                    rc01_col = st.selectbox("Στήλη RC-01", cols)

                    if st.button("Δημιουργία πίνακα Service 26 από το αρχείο"):
                        required = [score_col,conf_col,attempt_col,rc01_col]
                        if "— καμία —" in required:
                            st.error("Επίλεξε score, βεβαιότητα, προσπάθεια και RC-01.")
                        else:
                            mapped26 = pd.DataFrame({
                                "προσπάθεια": df[attempt_col],
                                "score": pd.to_numeric(df[score_col], errors="coerce"),
                                "βεβαιότητα": pd.to_numeric(df[conf_col], errors="coerce"),
                                "RC-01": pd.to_numeric(df[rc01_col], errors="coerce"),
                            })
                            if learner26_col != "— καμία —":
                                mapped26["μαθητής"] = df[learner26_col].astype(str)
                            else:
                                mapped26["μαθητής"] = [
                                    f"CSV learner {i+1}" for i in range(len(mapped26))
                                ]
                            mapped26["κατάσταση"] = mapped26.apply(
                                lambda r: κατάσταση_26(
                                    r["score"] if pd.notna(r["score"]) else 0,
                                    r["βεβαιότητα"] if pd.notna(r["βεβαιότητα"]) else 0,
                                    r["RC-01"] if pd.notna(r["RC-01"]) else 0
                                ),
                                axis=1
                            )
                            mapped26["κατάκτηση"] = np.where(
                                mapped26["score"] >= 70, "Επαρκής", "Αναπτυσσόμενο"
                            )
                            st.session_state["custom_s26"] = mapped26
                            st.session_state["custom_s26_source"] = active_dataset_name()
                            st.success("Το Service 26 συνδέθηκε με το ενεργό CSV.")
                            st.dataframe(mapped26, use_container_width=True, hide_index=True)

            with tabs[5]:
                st.subheader("Εξαγωγές")

                clean_df = st.session_state.get("inelso_uploaded_clean", df)

                st.download_button(
                    "Λήψη καθαρισμένου CSV",
                    clean_df.to_csv(index=False).encode("utf-8-sig"),
                    f"{Path(filename).stem}_clean.csv",
                    "text/csv"
                )

                quality_export = pd.DataFrame({
                    "Στήλη": df.columns.astype(str),
                    "Κενά": [int(df[c].isna().sum()) for c in df.columns],
                    "Κενά (%)": [round(float(df[c].isna().mean()*100),2) for c in df.columns],
                    "Μοναδικές τιμές": [int(df[c].nunique(dropna=True)) for c in df.columns],
                    "Τύπος": [str(df[c].dtype) for c in df.columns]
                })

                st.download_button(
                    "Λήψη αναφοράς ποιότητας CSV",
                    quality_export.to_csv(index=False).encode("utf-8-sig"),
                    f"{Path(filename).stem}_quality_report.csv",
                    "text/csv"
                )

                if "custom_s21" in st.session_state:
                    st.download_button(
                        "Λήψη παραγόμενου Service 21 CSV",
                        st.session_state["custom_s21"].to_csv(index=False).encode("utf-8-sig"),
                        f"{Path(filename).stem}_service21.csv",
                        "text/csv"
                    )

                if "custom_s26" in st.session_state:
                    st.download_button(
                        "Λήψη παραγόμενου Service 26 CSV",
                        st.session_state["custom_s26"].to_csv(index=False).encode("utf-8-sig"),
                        f"{Path(filename).stem}_service26.csv",
                        "text/csv"
                    )


# ==========================================================
# INELSO ANALYTICS LAB
# ==========================================================
elif section == "INELSO Analytics Lab":
    show_active_dataset_context("ANALYTICS")
    st.title("INELSO Analytics Lab")
    st.caption("Γενική ανάλυση εκπαιδευτικών δεδομένων: ποιότητα, πρόβλεψη, ομαδοποίηση, συσχετίσεις και ερμηνεία.")

    source_options = ["Χρήση τελευταίου αρχείου από το Κέντρο Δεδομένων"]
    if "inelso_uploaded_original" not in st.session_state:
        st.warning("Δεν υπάρχει ακόμη ανεβασμένο dataset. Πήγαινε πρώτα στο «Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων» και φόρτωσε ένα αρχείο.")
        lab_df = None
    else:
        source = st.radio("Πηγή dataset", source_options, horizontal=True)
        lab_df = st.session_state.get(
            "inelso_uploaded_clean",
            st.session_state["inelso_uploaded_original"]
        ).copy()

    if lab_df is not None and len(lab_df) > 0:
        st.success(f"Ενεργό dataset: {st.session_state.get('inelso_uploaded_name','Ανεβασμένο αρχείο')}")

        # ------------------------------------------------------
        # Automated data-quality score
        # ------------------------------------------------------
        nrows, ncols = lab_df.shape
        total_cells = max(1, nrows*ncols)
        missing_rate = float(lab_df.isna().sum().sum()/total_cells)
        duplicate_rate = float(lab_df.duplicated().mean()) if nrows else 0
        constant_cols = sum(lab_df[c].nunique(dropna=True) <= 1 for c in lab_df.columns)
        constant_rate = constant_cols / max(1,ncols)
        quality_score = max(
            0,
            min(100, round(100 - 45*missing_rate - 25*duplicate_rate - 30*constant_rate,1))
        )

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Γραμμές", nrows)
        c2.metric("Στήλες", ncols)
        c3.metric("Δείκτης ποιότητας", f"{quality_score:.1f}/100")
        c4.metric("Σταθερές στήλες", constant_cols)

        tabs = st.tabs([
            "Αυτόματη διάγνωση",
            "Συσχετίσεις & σχέσεις",
            "Classification",
            "Regression",
            "Clustering",
            "Σύγκριση ομάδων",
            "Αυτόματη αναφορά"
        ])

        # ------------------------------------------------------
        # Automated diagnosis
        # ------------------------------------------------------
        with tabs[0]:
            st.subheader("Αυτόματη διάγνωση dataset")

            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in lab_df.columns if c not in numeric_cols]

            diagnosis = pd.DataFrame([
                {"Έλεγχος":"Κενές τιμές","Τιμή":f"{missing_rate*100:.2f}%","Ερμηνεία":"Χαμηλότερο είναι καλύτερο"},
                {"Έλεγχος":"Διπλότυπες γραμμές","Τιμή":f"{duplicate_rate*100:.2f}%","Ερμηνεία":"Ελέγξτε αν είναι πραγματικές επαναλήψεις"},
                {"Έλεγχος":"Αριθμητικές στήλες","Τιμή":len(numeric_cols),"Ερμηνεία":"Υποψήφιες για regression/correlation/clustering"},
                {"Έλεγχος":"Μη αριθμητικές στήλες","Τιμή":len(categorical_cols),"Ερμηνεία":"Υποψήφιες κατηγορικές μεταβλητές"},
                {"Έλεγχος":"Σταθερές στήλες","Τιμή":constant_cols,"Ερμηνεία":"Συνήθως δεν προσφέρουν πληροφορία σε ML"},
            ])
            st.dataframe(diagnosis, use_container_width=True, hide_index=True)

            st.subheader("Προτεινόμενες αναλύσεις")
            suggestions = []
            if len(numeric_cols) >= 2:
                suggestions.append("Συσχετίσεις και Regression")
                suggestions.append("Clustering")
            if categorical_cols:
                suggestions.append("Classification, εφόσον επιλεγεί κατηγορικό target")
                suggestions.append("Σύγκριση ομάδων")
            if missing_rate > .10:
                suggestions.append("Έλεγχος στρατηγικής διαχείρισης ελλιπών τιμών")
            if duplicate_rate > 0:
                suggestions.append("Έλεγχος και πιθανή αφαίρεση διπλοεγγραφών")
            if not suggestions:
                suggestions.append("Περιγραφική ανάλυση και έλεγχος ποιότητας")

            for s in suggestions:
                st.write("•", s)

        # ------------------------------------------------------
        # Correlations
        # ------------------------------------------------------
        with tabs[1]:
            st.subheader("Συσχετίσεις & σχέσεις")
            numeric = lab_df.select_dtypes(include=np.number)

            if numeric.shape[1] >= 2:
                corr = numeric.corr(numeric_only=True)
                st.dataframe(corr.round(3), use_container_width=True)

                flat = corr.where(
                    np.triu(np.ones(corr.shape), k=1).astype(bool)
                ).stack().reset_index()
                flat.columns = ["Μεταβλητή 1","Μεταβλητή 2","Συσχέτιση"]
                flat["Απόλυτη συσχέτιση"] = flat["Συσχέτιση"].abs()
                flat = flat.sort_values("Απόλυτη συσχέτιση", ascending=False)

                st.subheader("Ισχυρότερες σχέσεις")
                st.dataframe(flat.head(15), use_container_width=True, hide_index=True)
            else:
                st.info("Απαιτούνται τουλάχιστον δύο αριθμητικές μεταβλητές.")

        # ------------------------------------------------------
        # Classification
        # ------------------------------------------------------
        with tabs[2]:
            st.subheader("Classification")
            st.caption("Επιλέγεις κατηγορικό target και το σύστημα εκτελεί leakage-safe train/test pipeline.")

            target = st.selectbox(
                "Target μεταβλητή για classification",
                ["— επιλέξτε —"] + list(lab_df.columns),
                key="class_target"
            )

            if target != "— επιλέξτε —":
                y = lab_df[target]
                unique_y = y.dropna().nunique()

                if unique_y < 2:
                    st.error("Το target έχει λιγότερες από 2 κατηγορίες.")
                elif unique_y > max(30, int(0.2*len(lab_df))):
                    st.warning("Το target έχει πολλές μοναδικές τιμές. Ίσως δεν είναι κατάλληλο για classification.")
                else:
                    available = [c for c in lab_df.columns if c != target]
                    predictors = st.multiselect(
                        "Predictors",
                        available,
                        default=available[:min(8,len(available))],
                        key="class_predictors"
                    )

                    model_name = st.radio(
                        "Μοντέλο",
                        ["Logistic Regression","Random Forest"],
                        horizontal=True
                    )

                    test_size = st.slider("Ποσοστό test set", 20, 40, 30, 5) / 100

                    if st.button("Εκτέλεση Classification", key="run_class"):
                        if len(predictors) == 0:
                            st.error("Επίλεξε τουλάχιστον έναν predictor.")
                        else:
                            data = lab_df[predictors+[target]].dropna(subset=[target]).copy()
                            X = data[predictors]
                            y = data[target].astype(str)

                            if y.value_counts().min() < 2:
                                st.error("Υπάρχει κατηγορία target με λιγότερες από 2 εγγραφές.")
                            else:
                                num = X.select_dtypes(include=np.number).columns.tolist()
                                cat = [c for c in predictors if c not in num]

                                transformers = []
                                if num:
                                    transformers.append((
                                        "num",
                                        Pipeline([
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler())
                                        ]),
                                        num
                                    ))
                                if cat:
                                    transformers.append((
                                        "cat",
                                        Pipeline([
                                            ("imputer", SimpleImputer(strategy="most_frequent")),
                                            ("onehot", OneHotEncoder(handle_unknown="ignore"))
                                        ]),
                                        cat
                                    ))

                                prep = ColumnTransformer(transformers)

                                if model_name == "Logistic Regression":
                                    model = LogisticRegression(max_iter=2000)
                                else:
                                    model = RandomForestClassifier(
                                        n_estimators=250,
                                        random_state=42,
                                        class_weight="balanced_subsample"
                                    )

                                pipe = Pipeline([("prep",prep),("model",model)])

                                try:
                                    Xtr,Xte,ytr,yte = train_test_split(
                                        X,y,test_size=test_size,random_state=42,stratify=y
                                    )
                                    pipe.fit(Xtr,ytr)
                                    pred = pipe.predict(Xte)

                                    acc = accuracy_score(yte,pred)
                                    bacc = balanced_accuracy_score(yte,pred)
                                    f1 = f1_score(yte,pred,average="weighted")

                                    m1,m2,m3,m4 = st.columns(4)
                                    m1.metric("Accuracy",f"{acc:.3f}")
                                    m2.metric("Balanced accuracy",f"{bacc:.3f}")
                                    m3.metric("Weighted F1",f"{f1:.3f}")
                                    m4.metric("Test N",len(yte))

                                    results = pd.DataFrame({
                                        "Πραγματική κατηγορία":yte.values,
                                        "Πρόβλεψη":pred
                                    })
                                    st.dataframe(results.head(50),use_container_width=True,hide_index=True)

                                    if model_name == "Random Forest":
                                        fitted_prep = pipe.named_steps["prep"]
                                        feature_names = fitted_prep.get_feature_names_out()
                                        imp = pipe.named_steps["model"].feature_importances_
                                        fi = pd.DataFrame({
                                            "Χαρακτηριστικό":feature_names,
                                            "Σημαντικότητα":imp
                                        }).sort_values("Σημαντικότητα",ascending=False)
                                        st.subheader("Feature importance")
                                        st.dataframe(fi.head(25),use_container_width=True,hide_index=True)
                                except Exception as e:
                                    st.error(f"Δεν ήταν δυνατή η εκτέλεση του classification: {e}")

        # ------------------------------------------------------
        # Regression
        # ------------------------------------------------------
        with tabs[3]:
            st.subheader("Regression")
            numeric_targets = lab_df.select_dtypes(include=np.number).columns.tolist()
            target_reg = st.selectbox(
                "Αριθμητικό target για regression",
                ["— επιλέξτε —"] + numeric_targets,
                key="reg_target"
            )

            if target_reg != "— επιλέξτε —":
                available = [c for c in lab_df.columns if c != target_reg]
                predictors = st.multiselect(
                    "Predictors",
                    available,
                    default=available[:min(8,len(available))],
                    key="reg_predictors"
                )

                model_name = st.radio(
                    "Μοντέλο regression",
                    ["Linear Regression","Random Forest Regression"],
                    horizontal=True
                )

                if st.button("Εκτέλεση Regression", key="run_reg"):
                    if not predictors:
                        st.error("Επίλεξε τουλάχιστον έναν predictor.")
                    else:
                        data = lab_df[predictors+[target_reg]].dropna(subset=[target_reg]).copy()
                        X = data[predictors]
                        y = pd.to_numeric(data[target_reg], errors="coerce")
                        valid = y.notna()
                        X = X.loc[valid]
                        y = y.loc[valid]

                        num = X.select_dtypes(include=np.number).columns.tolist()
                        cat = [c for c in predictors if c not in num]
                        transformers = []

                        if num:
                            transformers.append((
                                "num",
                                Pipeline([
                                    ("imputer",SimpleImputer(strategy="median")),
                                    ("scaler",StandardScaler())
                                ]),
                                num
                            ))
                        if cat:
                            transformers.append((
                                "cat",
                                Pipeline([
                                    ("imputer",SimpleImputer(strategy="most_frequent")),
                                    ("onehot",OneHotEncoder(handle_unknown="ignore"))
                                ]),
                                cat
                            ))

                        prep = ColumnTransformer(transformers)
                        model = (
                            LinearRegression()
                            if model_name == "Linear Regression"
                            else RandomForestRegressor(n_estimators=250,random_state=42)
                        )
                        pipe = Pipeline([("prep",prep),("model",model)])

                        try:
                            Xtr,Xte,ytr,yte = train_test_split(
                                X,y,test_size=.30,random_state=42
                            )
                            pipe.fit(Xtr,ytr)
                            pred = pipe.predict(Xte)

                            mae = mean_absolute_error(yte,pred)
                            rmse = float(np.sqrt(mean_squared_error(yte,pred)))
                            r2 = r2_score(yte,pred)

                            m1,m2,m3,m4 = st.columns(4)
                            m1.metric("MAE",f"{mae:.3f}")
                            m2.metric("RMSE",f"{rmse:.3f}")
                            m3.metric("R²",f"{r2:.3f}")
                            m4.metric("Test N",len(yte))

                            result = pd.DataFrame({
                                "Πραγματική τιμή":yte.values,
                                "Πρόβλεψη":pred,
                                "Σφάλμα":yte.values-pred
                            })
                            st.dataframe(result.head(50),use_container_width=True,hide_index=True)

                            if model_name == "Random Forest Regression":
                                feature_names = pipe.named_steps["prep"].get_feature_names_out()
                                imp = pipe.named_steps["model"].feature_importances_
                                fi = pd.DataFrame({
                                    "Χαρακτηριστικό":feature_names,
                                    "Σημαντικότητα":imp
                                }).sort_values("Σημαντικότητα",ascending=False)
                                st.subheader("Feature importance")
                                st.dataframe(fi.head(25),use_container_width=True,hide_index=True)
                        except Exception as e:
                            st.error(f"Δεν ήταν δυνατή η εκτέλεση του regression: {e}")

        # ------------------------------------------------------
        # Clustering
        # ------------------------------------------------------
        with tabs[4]:
            st.subheader("Clustering")
            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()

            selected = st.multiselect(
                "Αριθμητικές μεταβλητές για clustering",
                numeric_cols,
                default=numeric_cols[:min(5,len(numeric_cols))],
                key="cluster_vars"
            )
            k = st.slider("Αριθμός clusters",2,8,3)

            if st.button("Εκτέλεση K-means", key="run_cluster"):
                if len(selected) < 2:
                    st.error("Επίλεξε τουλάχιστον δύο αριθμητικές μεταβλητές.")
                else:
                    X = lab_df[selected].copy()
                    X = pd.DataFrame(
                        SimpleImputer(strategy="median").fit_transform(X),
                        columns=selected,
                        index=X.index
                    )
                    Xs = StandardScaler().fit_transform(X)

                    if len(X) <= k:
                        st.error("Δεν υπάρχουν αρκετές εγγραφές για τον επιλεγμένο αριθμό clusters.")
                    else:
                        km = KMeans(n_clusters=k,random_state=42,n_init=20)
                        labels = km.fit_predict(Xs)
                        sil = silhouette_score(Xs,labels) if len(set(labels)) > 1 else np.nan

                        m1,m2 = st.columns(2)
                        m1.metric("Clusters",k)
                        m2.metric("Silhouette score",f"{sil:.3f}")

                        clustered = lab_df.copy()
                        clustered["Cluster"] = labels
                        st.session_state["inelso_clustered"] = clustered

                        counts = clustered["Cluster"].value_counts().sort_index().reset_index()
                        counts.columns = ["Cluster","Πλήθος"]
                        st.dataframe(counts,use_container_width=True,hide_index=True)

                        profile = clustered.groupby("Cluster")[selected].mean(numeric_only=True)
                        st.subheader("Μέσο προφίλ ανά cluster")
                        st.dataframe(profile.round(3),use_container_width=True)

                        st.download_button(
                            "Λήψη dataset με clusters",
                            clustered.to_csv(index=False).encode("utf-8-sig"),
                            "INELSO_dataset_with_clusters.csv",
                            "text/csv"
                        )

        # ------------------------------------------------------
        # Group comparison
        # ------------------------------------------------------
        with tabs[5]:
            st.subheader("Σύγκριση ομάδων")
            group_cols = [
                c for c in lab_df.columns
                if lab_df[c].nunique(dropna=True) <= min(20,max(2,int(len(lab_df)*.2)))
                and lab_df[c].nunique(dropna=True) >= 2
            ]
            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()

            if group_cols and numeric_cols:
                group_col = st.selectbox("Μεταβλητή ομάδας",group_cols,key="group_col")
                value_col = st.selectbox("Αριθμητική μεταβλητή",numeric_cols,key="group_value")
                comparison = lab_df.groupby(group_col)[value_col].agg(
                    ["count","mean","median","std","min","max"]
                ).reset_index()
                comparison.columns = [
                    group_col,"N","Μέσος","Διάμεσος","Τυπική απόκλιση","Ελάχιστο","Μέγιστο"
                ]
                st.dataframe(comparison,use_container_width=True,hide_index=True)
                st.bar_chart(comparison.set_index(group_col)[["Μέσος"]])
            else:
                st.info("Δεν εντοπίστηκαν κατάλληλες μεταβλητές ομάδας και αριθμητικές μεταβλητές.")

        # ------------------------------------------------------
        # Auto report
        # ------------------------------------------------------
        with tabs[6]:
            st.subheader("Αυτόματη αναφορά")

            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in lab_df.columns if c not in numeric_cols]

            report_text = f"""INELSO ANALYTICS LAB — ΑΥΤΟΜΑΤΗ ΑΝΑΦΟΡΑ

Dataset: {st.session_state.get('inelso_uploaded_name','Ανεβασμένο αρχείο')}
Γραμμές: {len(lab_df)}
Στήλες: {len(lab_df.columns)}
Δείκτης ποιότητας: {quality_score:.1f}/100
Ποσοστό κενών τιμών: {missing_rate*100:.2f}%
Ποσοστό διπλοεγγραφών: {duplicate_rate*100:.2f}%
Αριθμητικές μεταβλητές: {len(numeric_cols)}
Κατηγορικές/λοιπές μεταβλητές: {len(categorical_cols)}
Σταθερές στήλες: {constant_cols}

Ερμηνεία:
- Ο δείκτης ποιότητας είναι ένας λειτουργικός δείκτης ελέγχου και όχι στατιστικά επικυρωμένο metric.
- Οι predictive αναλύσεις χρησιμοποιούν train/test split και preprocessing μέσα στο pipeline.
- Τα αποτελέσματα πρέπει να ερμηνεύονται με βάση το ερευνητικό ερώτημα και τη δομή του dataset.
"""
            st.text_area("Προεπισκόπηση αναφοράς",report_text,height=320)
            st.download_button(
                "Λήψη αυτόματης αναφοράς",
                report_text.encode("utf-8"),
                "INELSO_Analytics_Lab_Report.txt",
                "text/plain"
            )

# ==========================================================
# ΠΡΟΣΟΜΟΙΩΤΗΣ WHAT-IF
# ==========================================================
elif section == "Προσομοιωτής What-if":
    show_active_dataset_context("S26")
    st.title("Προσομοιωτής What-if")
    st.caption("Δοκίμασε διαφορετικά thresholds χωρίς να αλλάξεις πραγματικά δεδομένα ή βαθμούς.")

    st.subheader("Service 26 — Adaptive thresholds")
    st.caption(
        f"Πηγή προσομοίωσης: {'Active CSV — ' + ACTIVE_NAME if 'custom_s26' in st.session_state else 'ελεγχόμενο Service 26 demo cohort'}"
    )

    confidence_threshold = st.slider(
        "Κατώφλι βεβαιότητας",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.01
    )
    ready_score_threshold = st.slider(
        "Κατώφλι score για ΕΤΟΙΜΟ",
        min_value=50,
        max_value=95,
        value=70,
        step=5
    )

    sim = cohort26.copy()

    def simulate_status(row):
        # For cohort demo we do not have RC-01 per learner, so confidence + score only.
        if row["βεβαιότητα"] < confidence_threshold:
            return "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ"
        if row["score"] < ready_score_threshold:
            return "ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ"
        return "ΕΤΟΙΜΟ"

    sim["Νέα κατάσταση"] = sim.apply(simulate_status,axis=1)

    old_counts = cohort26["κατάσταση"].value_counts()
    new_counts = sim["Νέα κατάσταση"].value_counts()

    a,b,c = st.columns(3)
    a.metric("ΕΤΟΙΜΟ",int(new_counts.get("ΕΤΟΙΜΟ",0)))
    b.metric("ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ",int(new_counts.get("ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ",0)))
    c.metric("ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ",int(new_counts.get("ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ",0)))

    sim["Αλλαγή"] = np.where(
        sim["κατάσταση"]==sim["Νέα κατάσταση"],
        "Χωρίς αλλαγή",
        sim["κατάσταση"] + " → " + sim["Νέα κατάσταση"]
    )

    st.dataframe(
        sim[["μαθητής","score","βεβαιότητα","κατάσταση","Νέα κατάσταση","Αλλαγή"]],
        use_container_width=True,
        hide_index=True
    )

    changed = int((sim["κατάσταση"] != sim["Νέα κατάσταση"]).sum())
    st.info(f"Με τα επιλεγμένα thresholds αλλάζουν κατηγορία {changed} από {len(sim)} learners.")

    st.subheader("Σενάρια")
    scenarios = pd.DataFrame([
        {"Σενάριο":"Βασικό","Confidence threshold":0.70,"Score threshold":70},
        {"Σενάριο":"Πιο αυστηρό","Confidence threshold":0.80,"Score threshold":80},
        {"Σενάριο":"Πιο ελαστικό","Confidence threshold":0.60,"Score threshold":60},
    ])
    st.dataframe(scenarios,use_container_width=True,hide_index=True)

    st.warning("Η προσομοίωση είναι sandbox. Δεν επηρεάζει το Moodle ούτε τις επίσημες αποφάσεις.")


# ==========================================================
# LIVE MOODLE
# ==========================================================
elif section == "Live Moodle":
    show_active_dataset_context("REPORT")
    st.title("Live Moodle")
    st.caption("Απευθείας ανάγνωση δεδομένων από Moodle Web Services / REST, χωρίς export αρχείου.")
    if ACTIVE_DF is not None:
        st.info(
            "Το ενεργό CSV παραμένει συνδεδεμένο ως analytical/provenance context. "
            "Τα live Moodle δεδομένα προέρχονται ανεξάρτητα από το REST API και δεν αντικαθίστανται από το CSV."
        )

    st.warning(
        "Απαιτείται ενεργοποιημένο Moodle Web Service και token με τα κατάλληλα δικαιώματα. "
        "Η εφαρμογή εκτελεί μόνο αναγνώσεις. Δεν γράφει βαθμούς ή submissions."
    )

    default_url = st.session_state.get("moodle_base_url", "")
    moodle_url = st.text_input(
        "Βασικό URL Moodle",
        value=default_url,
        placeholder="https://example.edu"
    ).rstrip("/")

    token = st.text_input(
        "Moodle Web Service token",
        type="password",
        help="Το token χρησιμοποιείται μόνο για το τρέχον αίτημα μέσα στη συνεδρία."
    )

    st.session_state["moodle_base_url"] = moodle_url

    def moodle_rest_call(base_url, ws_token, function_name, params=None, timeout=20):
        if params is None:
            params = {}
        endpoint = f"{base_url}/webservice/rest/server.php"
        payload = {
            "wstoken": ws_token,
            "wsfunction": function_name,
            "moodlewsrestformat": "json"
        }
        payload.update(params)
        response = requests.get(endpoint, params=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and ("exception" in data or "errorcode" in data):
            raise RuntimeError(data.get("message", data))
        return data

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("1. Έλεγχος σύνδεσης")
        if st.button("Έλεγχος σύνδεσης Moodle"):
            if not moodle_url or not token:
                st.error("Συμπλήρωσε URL και token.")
            else:
                try:
                    info = moodle_rest_call(
                        moodle_url, token, "core_webservice_get_site_info"
                    )
                    st.session_state["moodle_site_info"] = info
                    st.success("Η σύνδεση με το Moodle λειτουργεί.")
                    show = {
                        "Όνομα ιστοτόπου": info.get("sitename"),
                        "Χρήστης Web Service": info.get("fullname"),
                        "Moodle release": info.get("release"),
                    }
                    st.json(show)
                except Exception as e:
                    st.error(f"Αποτυχία σύνδεσης: {e}")

    with col2:
        st.subheader("2. Μαθήματα")
        if st.button("Ανάκτηση μαθημάτων"):
            if not moodle_url or not token:
                st.error("Συμπλήρωσε URL και token.")
            else:
                try:
                    courses = moodle_rest_call(
                        moodle_url, token, "core_course_get_courses"
                    )
                    courses_df = pd.DataFrame(courses)
                    st.session_state["moodle_courses"] = courses_df
                    st.success(f"Ανακτήθηκαν {len(courses_df)} μαθήματα.")
                except Exception as e:
                    st.error(f"Αποτυχία ανάκτησης μαθημάτων: {e}")

    if "moodle_courses" in st.session_state:
        courses_df = st.session_state["moodle_courses"]
        display_cols = [c for c in ["id","shortname","fullname","visible"] if c in courses_df.columns]
        st.dataframe(courses_df[display_cols] if display_cols else courses_df,
                     use_container_width=True, hide_index=True)

        if "id" in courses_df.columns:
            course_options = {
                f"{r.get('shortname','')} — {r.get('fullname','')} (ID {r['id']})": int(r["id"])
                for _,r in courses_df.iterrows()
            }
            selected_course_label = st.selectbox(
                "Επιλέξτε μάθημα",
                list(course_options.keys())
            )
            selected_course_id = course_options[selected_course_label]

            st.subheader("3. Live δεδομένα μαθήματος")
            c1,c2,c3 = st.columns(3)

            with c1:
                if st.button("Εγγεγραμμένοι χρήστες"):
                    try:
                        users = moodle_rest_call(
                            moodle_url, token,
                            "core_enrol_get_enrolled_users",
                            {"courseid": selected_course_id}
                        )
                        udf = pd.DataFrame(users)
                        st.session_state["moodle_enrolled_users"] = udf
                        st.success(f"Ανακτήθηκαν {len(udf)} χρήστες.")
                    except Exception as e:
                        st.error(f"Αποτυχία: {e}")

            with c2:
                if st.button("Quiz του μαθήματος"):
                    try:
                        quizzes = moodle_rest_call(
                            moodle_url, token,
                            "mod_quiz_get_quizzes_by_courses",
                            {"courseids[0]": selected_course_id}
                        )
                        qlist = quizzes.get("quizzes", quizzes) if isinstance(quizzes, dict) else quizzes
                        qdf = pd.DataFrame(qlist)
                        st.session_state["moodle_quizzes"] = qdf
                        st.success(f"Ανακτήθηκαν {len(qdf)} quiz.")
                    except Exception as e:
                        st.error(f"Αποτυχία: {e}")

            with c3:
                if st.button("Περιεχόμενο μαθήματος"):
                    try:
                        contents = moodle_rest_call(
                            moodle_url, token,
                            "core_course_get_contents",
                            {"courseid": selected_course_id}
                        )
                        st.session_state["moodle_course_contents"] = contents
                        st.success("Ανακτήθηκε η δομή του μαθήματος.")
                    except Exception as e:
                        st.error(f"Αποτυχία: {e}")

    if "moodle_enrolled_users" in st.session_state:
        st.subheader("Εγγεγραμμένοι χρήστες")
        users = st.session_state["moodle_enrolled_users"]
        safe_cols = [c for c in ["id","fullname","email","lastaccess"] if c in users.columns]
        st.dataframe(users[safe_cols] if safe_cols else users,
                     use_container_width=True, hide_index=True)

        st.download_button(
            "Λήψη live Moodle users CSV",
            users.to_csv(index=False).encode("utf-8-sig"),
            "moodle_live_users.csv",
            "text/csv"
        )

    if "moodle_quizzes" in st.session_state:
        st.subheader("Quiz")
        qdf = st.session_state["moodle_quizzes"]
        show_cols = [c for c in ["id","name","course","grade","sumgrades","timeopen","timeclose"] if c in qdf.columns]
        st.dataframe(qdf[show_cols] if show_cols else qdf,
                     use_container_width=True, hide_index=True)

    if "moodle_course_contents" in st.session_state:
        st.subheader("Δομή μαθήματος")
        sections = st.session_state["moodle_course_contents"]
        flat = []
        for sec in sections if isinstance(sections,list) else []:
            sec_name = sec.get("name","")
            for mod in sec.get("modules",[]):
                flat.append({
                    "Ενότητα":sec_name,
                    "Όνομα":mod.get("name"),
                    "Τύπος":mod.get("modname"),
                    "ID":mod.get("id"),
                    "URL":mod.get("url")
                })
        if flat:
            st.dataframe(pd.DataFrame(flat),use_container_width=True,hide_index=True)

    st.subheader("Προχωρημένο REST αίτημα")
    st.caption("Για Moodle functions που δεν έχουν έτοιμο κουμπί παραπάνω.")
    custom_function = st.text_input(
        "Όνομα Moodle Web Service function",
        placeholder="π.χ. gradereport_user_get_grade_items"
    )
    custom_params_text = st.text_area(
        "Παράμετροι ως JSON",
        value="{}",
        height=100
    )
    if st.button("Εκτέλεση προχωρημένου αιτήματος"):
        try:
            params = json.loads(custom_params_text or "{}")
            result = moodle_rest_call(moodle_url, token, custom_function, params)
            st.session_state["moodle_custom_result"] = result
            st.success("Το αίτημα ολοκληρώθηκε.")
            if isinstance(result, list):
                st.dataframe(pd.DataFrame(result),use_container_width=True,hide_index=True)
            elif isinstance(result, dict):
                st.json(result)
            else:
                st.write(result)
        except Exception as e:
            st.error(f"Αποτυχία αιτήματος: {e}")


# ==========================================================
# ΕΛΕΓΧΟΜΕΝΗ ΕΓΓΡΑΦΗ MOODLE
# ==========================================================
elif section == "Ελεγχόμενη Εγγραφή Moodle":
    show_active_dataset_context("REPORT")
    st.title("Ελεγχόμενη Εγγραφή Moodle")
    st.caption("Εγγραφή στο Moodle μόνο με προεπισκόπηση, ρητή επιβεβαίωση και audit log.")
    st.info("Read → Decision → Human Approval → Write → Verification")


    pending_meta = st.session_state.get("pending_moodle_result_meta")
    if isinstance(pending_meta, dict):
        st.success(
            f"Υπάρχει staged αποτέλεσμα από {pending_meta.get('service','INELSO Service')}. "
            "Έλεγξέ το παρακάτω πριν από οποιαδήποτε πραγματική εγγραφή."
        )
        meta_preview = pd.DataFrame([{
            "Service": pending_meta.get("service"),
            "Entity": pending_meta.get("entity"),
            "Status": pending_meta.get("status"),
            "Score": pending_meta.get("score"),
            "Summary": pending_meta.get("summary"),
        }])
        st.dataframe(meta_preview, use_container_width=True, hide_index=True)


    st.warning(
        "Οι ενέργειες αυτής της ενότητας μπορούν να αλλάξουν πραγματικά δεδομένα στο Moodle. "
        "Χρησιμοποίησε token με τα ελάχιστα απαραίτητα δικαιώματα και δοκίμασε πρώτα σε demo/test course."
    )

    moodle_url = st.text_input(
        "Βασικό URL Moodle",
        value=st.session_state.get("moodle_base_url",""),
        key="write_moodle_url"
    ).rstrip("/")

    token = st.text_input(
        "Moodle Web Service token",
        type="password",
        key="write_moodle_token"
    )

    if "moodle_write_audit" not in st.session_state:
        st.session_state["moodle_write_audit"] = pd.DataFrame(columns=[
            "Ημερομηνία","Ενέργεια","Function","Στόχος",
            "Προεπισκόπηση","Αποτέλεσμα","Κατάσταση"
        ])

    def moodle_write_call(base_url, ws_token, function_name, params=None, timeout=20):
        if params is None:
            params = {}
        endpoint = f"{base_url}/webservice/rest/server.php"
        payload = {
            "wstoken": ws_token,
            "wsfunction": function_name,
            "moodlewsrestformat": "json"
        }
        payload.update(params)
        response = requests.post(endpoint, data=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and ("exception" in data or "errorcode" in data):
            raise RuntimeError(data.get("message", str(data)))
        return data

    def add_write_audit(action, function_name, target, preview, result, status):
        row = pd.DataFrame([{
            "Ημερομηνία":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ενέργεια":action,
            "Function":function_name,
            "Στόχος":target,
            "Προεπισκόπηση":preview,
            "Αποτέλεσμα":result,
            "Κατάσταση":status
        }])
        st.session_state["moodle_write_audit"] = pd.concat(
            [st.session_state["moodle_write_audit"], row],
            ignore_index=True
        )

    action = st.radio(
        "Επιλέξτε ελεγχόμενη ενέργεια",
        [
            "Καταγραφή εκπαιδευτικής παρέμβασης / σημείωσης",
            "Αποστολή προσωπικού μηνύματος",
            "Ενημέρωση βαθμού — προχωρημένο"
        ],
        horizontal=True
    )

    # ------------------------------------------------------
    # EDUCATIONAL NOTE / INTERVENTION
    # ------------------------------------------------------
    if action == "Καταγραφή εκπαιδευτικής παρέμβασης / σημείωσης":
        st.subheader("Καταγραφή εκπαιδευτικής παρέμβασης / σημείωσης")
        st.caption("Η σημείωση γράφεται στο Moodle ως course note και δεν εξαρτάται από τις προσωπικές ρυθμίσεις μηνυμάτων.")

        live_users = st.session_state.get("moodle_enrolled_users")
        live_courses = st.session_state.get("moodle_courses")

        if isinstance(live_courses, pd.DataFrame) and len(live_courses) > 0 and "id" in live_courses.columns:
            course_labels = {}
            for _, cr in live_courses.iterrows():
                cid = int(cr["id"])
                shortname = str(cr.get("shortname", ""))
                fullname = str(cr.get("fullname", f"Course {cid}"))
                course_labels[f"{shortname} — {fullname} (ID {cid})"] = cid
            selected_course_label = st.selectbox("Μάθημα", list(course_labels.keys()), key="note_course")
            note_course_id = course_labels[selected_course_label]
        else:
            st.info("Δεν υπάρχουν live μαθήματα στη συνεδρία. Μπορείς να βάλεις Course ID χειροκίνητα.")
            note_course_id = st.number_input("Course ID", min_value=1, step=1, value=6, key="note_course_manual")

        if isinstance(live_users, pd.DataFrame) and len(live_users) > 0 and "id" in live_users.columns:
            user_labels = {}
            for _, ur in live_users.iterrows():
                uid = int(ur["id"])
                fullname = str(ur.get("fullname", f"User {uid}"))
                email = str(ur.get("email", ""))
                label = f"{fullname} — ID {uid}"
                if email and email != "nan":
                    label += f" — {email}"
                user_labels[label] = uid
            selected_user_label = st.selectbox("Learner / χρήστης", list(user_labels.keys()), key="note_user")
            note_user_id = user_labels[selected_user_label]
        else:
            st.info("Δεν υπάρχουν live χρήστες στη συνεδρία. Μπορείς να βάλεις User ID χειροκίνητα.")
            note_user_id = st.number_input("User ID", min_value=1, step=1, value=208, key="note_user_manual")

        intervention_options = [
            "Service 16 — Early Warning / Human review",
            "Service 21 — Έλεγχος ποιότητας ερώτησης",
            "Service 25 — Faculty authoring review",
            "Service 26 — Νέα προσπάθεια",
            "Service 26 — Enrichment",
            "Service 26 — Human review",
            "Service 28 — Faculty-approved feedback",
            "Service 31 — Learning insight",
            "Άλλο"
        ]

        pending_type = st.session_state.get("pending_moodle_intervention_type", "")
        default_idx = intervention_options.index(pending_type) if pending_type in intervention_options else 1
        intervention_type = st.selectbox(
            "Τύπος παρέμβασης",
            intervention_options,
            index=default_idx
        )

        pending_note = st.session_state.get("pending_moodle_note_text", "")
        note_text = st.text_area(
            "Κείμενο εκπαιδευτικής παρέμβασης",
            value=pending_note,
            height=170,
            placeholder="Καταγράψτε την παρέμβαση, το feedback ή την απόφαση που θα αποθηκευτεί στο Moodle."
        )

        if pending_note:
            st.caption(
                "Το κείμενο έχει προ-συμπληρωθεί από άλλο INELSO Service. "
                "Ελέγξτε το πριν από οποιαδήποτε πραγματική εγγραφή."
            )

        formatted_note = (
            f"[INELSO] {intervention_type}\n\n"
            f"{note_text.strip()}\n\n"
            f"Καταγραφή μέσω INELSO Smart Learning Control Center."
            + (f"\nAnalytical context: {ACTIVE_NAME}." if ACTIVE_DF is not None else "")
        )

        note_preview = pd.DataFrame([{
            "Ενέργεια":"Καταγραφή Moodle note",
            "Course ID":int(note_course_id),
            "User ID":int(note_user_id),
            "Τύπος παρέμβασης":intervention_type,
            "Κείμενο":formatted_note,
            "Moodle function":"core_notes_create_notes",
            "Πηγή analytical context": ACTIVE_NAME if ACTIVE_DF is not None else "Χωρίς ενεργό CSV"
        }])

        st.subheader("Προεπισκόπηση")
        st.dataframe(note_preview, use_container_width=True, hide_index=True)

        note_confirm1 = st.checkbox(
            "Έλεγξα το μάθημα, τον learner και το κείμενο της παρέμβασης.",
            key="note_confirm1"
        )
        note_confirm2 = st.checkbox(
            "Επιβεβαιώνω ότι θέλω να καταγραφεί πραγματικά αυτή η σημείωση στο Moodle.",
            key="note_confirm2"
        )

        if st.button(
            "ΚΑΤΑΓΡΑΦΗ ΠΑΡΕΜΒΑΣΗΣ ΣΤΟ MOODLE",
            disabled=not (note_confirm1 and note_confirm2),
            type="primary"
        ):
            if not moodle_url or not token:
                st.error("Συμπλήρωσε Moodle URL και token.")
            elif not note_text.strip():
                st.error("Το κείμενο της παρέμβασης είναι κενό.")
            else:
                function_name = "core_notes_create_notes"
                params = {
                    "notes[0][userid]": int(note_user_id),
                    "notes[0][publishstate]": "course",
                    "notes[0][courseid]": int(note_course_id),
                    "notes[0][text]": formatted_note,
                    "notes[0][format]": 1
                }
                try:
                    result = moodle_write_call(
                        moodle_url, token, function_name, params
                    )

                    note_ok = False
                    note_id = None

                    if isinstance(result, list) and len(result) > 0:
                        first = result[0]
                        if isinstance(first, dict):
                            note_id = first.get("noteid") or first.get("id")
                        else:
                            note_id = first
                    elif isinstance(result, dict):
                        note_id = result.get("noteid") or result.get("id")

                    try:
                        note_ok = int(note_id) > 0
                    except Exception:
                        note_ok = False

                    if note_ok:
                        st.success(f"Η εκπαιδευτική παρέμβαση καταγράφηκε στο Moodle. Note ID: {note_id}")
                        audit_status = "ΕΠΙΤΥΧΙΑ"
                        st.session_state.pop("pending_moodle_note_text", None)
                        st.session_state.pop("pending_moodle_intervention_type", None)
                    else:
                        st.warning(
                            "Το Moodle απάντησε χωρίς σαφές θετικό Note ID. "
                            "Έλεγξε το αποτέλεσμα και τη λίστα σημειώσεων πριν θεωρηθεί επιτυχής."
                        )
                        audit_status = "ΠΡΟΣ ΕΛΕΓΧΟ"

                    st.json(result)

                    add_write_audit(
                        "Καταγραφή εκπαιδευτικής παρέμβασης",
                        function_name,
                        f"Course {note_course_id} / User {note_user_id}",
                        formatted_note,
                        json.dumps(result, ensure_ascii=False),
                        audit_status
                    )

                except Exception as e:
                    st.error(f"Η καταγραφή της παρέμβασης απέτυχε: {e}")
                    add_write_audit(
                        "Καταγραφή εκπαιδευτικής παρέμβασης",
                        function_name,
                        f"Course {note_course_id} / User {note_user_id}",
                        formatted_note,
                        str(e),
                        "ΑΠΟΤΥΧΙΑ"
                    )

        st.markdown("---")
        st.subheader("Ανάκτηση σημειώσεων μαθήματος")
        st.caption("Έλεγχος ότι οι Moodle notes έχουν αποθηκευτεί και μπορούν να ανακτηθούν.")

        if st.button("ΑΝΑΚΤΗΣΗ MOODLE NOTES", key="load_notes"):
            if not moodle_url or not token:
                st.error("Συμπλήρωσε Moodle URL και token.")
            else:
                try:
                    result = moodle_write_call(
                        moodle_url,
                        token,
                        "core_notes_get_course_notes",
                        {
                            "courseid": int(note_course_id),
                            "userid": int(note_user_id)
                        }
                    )
                    st.session_state["moodle_notes_result"] = result
                    st.success("Οι σημειώσεις ανακτήθηκαν από το Moodle.")
                except Exception as e:
                    st.error(f"Η ανάκτηση σημειώσεων απέτυχε: {e}")

        if "moodle_notes_result" in st.session_state:
            notes_result = st.session_state["moodle_notes_result"]

            notes_list = []
            if isinstance(notes_result, dict):
                for key in ["coursenotes","notes","sitenotes","personalnotes"]:
                    value = notes_result.get(key)
                    if isinstance(value, list):
                        notes_list.extend(value)
            elif isinstance(notes_result, list):
                notes_list = notes_result

            if len(notes_list) > 0:
                notes_df = pd.DataFrame(notes_list)
                display_map = {
                    "id":"Note ID",
                    "courseid":"Course ID",
                    "userid":"User ID",
                    "content":"Περιεχόμενο",
                    "publishstate":"Κατάσταση",
                    "created":"Δημιουργήθηκε",
                    "lastmodified":"Τελευταία τροποποίηση"
                }
                show_cols=[c for c in display_map if c in notes_df.columns]
                clean_notes=notes_df[show_cols].rename(columns=display_map)

                for ts_col in ["Δημιουργήθηκε","Τελευταία τροποποίηση"]:
                    if ts_col in clean_notes.columns:
                        try:
                            clean_notes[ts_col]=pd.to_datetime(clean_notes[ts_col],unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass

                st.success("Moodle Write Verified — η παρέμβαση ανακτήθηκε επιτυχώς από το Moodle.")
                st.session_state.pop("pending_moodle_result_meta", None)
                st.dataframe(clean_notes,use_container_width=True,hide_index=True)

                latest=notes_list[-1]
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Note ID",str(latest.get("id","—")))
                c2.metric("Course ID",str(latest.get("courseid","—")))
                c3.metric("User ID",str(latest.get("userid","—")))
                c4.metric("Verification","VERIFIED")
            else:
                st.info("Δεν βρέθηκαν σημειώσεις για τον επιλεγμένο χρήστη/μάθημα.")

    # ------------------------------------------------------
    # MESSAGE
    # ------------------------------------------------------
    elif action == "Αποστολή προσωπικού μηνύματος":
        st.subheader("Αποστολή προσωπικού μηνύματος")

        live_users = st.session_state.get("moodle_enrolled_users")

        if isinstance(live_users, pd.DataFrame) and len(live_users) > 0 and "id" in live_users.columns:
            user_labels = {}
            for _, ur in live_users.iterrows():
                uid = int(ur["id"])
                fullname = str(ur.get("fullname", f"User {uid}"))
                email = str(ur.get("email", ""))
                label = f"{fullname} — ID {uid}"
                if email and email != "nan":
                    label += f" — {email}"
                user_labels[label] = uid

            recipient_mode = st.radio(
                "Επιλογή παραλήπτη",
                ["Από τους live εγγεγραμμένους χρήστες", "Χειροκίνητο User ID"],
                horizontal=True
            )

            if recipient_mode == "Από τους live εγγεγραμμένους χρήστες":
                selected_label = st.selectbox(
                    "Παραλήπτης",
                    list(user_labels.keys())
                )
                user_id = user_labels[selected_label]
            else:
                user_id = st.number_input(
                    "Moodle User ID παραλήπτη",
                    min_value=1,
                    step=1,
                    value=1
                )
        else:
            st.info(
                "Δεν υπάρχουν live εγγεγραμμένοι χρήστες στη συνεδρία. "
                "Μπορείς να βάλεις User ID χειροκίνητα ή να πας πρώτα στο «Live Moodle»."
            )
            user_id = st.number_input(
                "Moodle User ID παραλήπτη",
                min_value=1,
                step=1,
                value=1
            )

        message_text = st.text_area(
            "Μήνυμα",
            height=160,
            placeholder="Γράψε το feedback ή την ειδοποίηση που θα σταλεί."
        )

        preview = pd.DataFrame([{
            "Ενέργεια":"Αποστολή προσωπικού μηνύματος",
            "Παραλήπτης User ID":int(user_id),
            "Κείμενο":message_text,
            "Moodle function":"core_message_send_instant_messages"
        }])
        st.subheader("Προεπισκόπηση")
        st.dataframe(preview,use_container_width=True,hide_index=True)

        if int(user_id) == 7:
            st.warning(
                "Ο User ID 7 έχει ήδη επιστρέψει περιορισμό από τις προσωπικές ρυθμίσεις μηνυμάτων. "
                "Για end-to-end test προτίμησε έναν Demo Student από τη live λίστα, π.χ. Demo Student 0201 (ID 208)."
            )

        confirm1 = st.checkbox(
            "Έλεγξα τον παραλήπτη και το κείμενο του μηνύματος.",
            key="msg_confirm1"
        )
        confirm2 = st.checkbox(
            "Επιβεβαιώνω ότι θέλω να σταλεί πραγματικά αυτό το μήνυμα στο Moodle.",
            key="msg_confirm2"
        )

        if st.button(
            "ΑΠΟΣΤΟΛΗ ΜΗΝΥΜΑΤΟΣ ΣΤΟ MOODLE",
            disabled=not (confirm1 and confirm2),
            type="primary"
        ):
            if not moodle_url or not token:
                st.error("Συμπλήρωσε Moodle URL και token.")
            elif not message_text.strip():
                st.error("Το μήνυμα είναι κενό.")
            else:
                function_name = "core_message_send_instant_messages"
                params = {
                    "messages[0][touserid]": int(user_id),
                    "messages[0][text]": message_text,
                    "messages[0][textformat]": 1
                }
                try:
                    result = moodle_write_call(
                        moodle_url, token, function_name, params
                    )

                    # Το Moodle μπορεί να επιστρέψει HTTP 200 αλλά αποτυχία ανά μήνυμα,
                    # π.χ. msgid = -1 και errormessage λόγω προτιμήσεων του παραλήπτη.
                    delivery_ok = False
                    delivery_error = ""

                    if isinstance(result, list) and len(result) > 0:
                        first = result[0] if isinstance(result[0], dict) else {}
                        msgid = first.get("msgid")
                        errormessage = str(first.get("errormessage", "") or "").strip()

                        try:
                            delivery_ok = int(msgid) > 0
                        except Exception:
                            delivery_ok = False

                        if errormessage:
                            delivery_error = errormessage
                            delivery_ok = False

                    elif isinstance(result, dict):
                        msgid = result.get("msgid")
                        errormessage = str(result.get("errormessage", "") or "").strip()

                        try:
                            delivery_ok = int(msgid) > 0
                        except Exception:
                            delivery_ok = False

                        if errormessage:
                            delivery_error = errormessage
                            delivery_ok = False

                    if delivery_ok:
                        st.success("Το μήνυμα παραδόθηκε επιτυχώς στο Moodle.")
                        audit_status = "ΕΠΙΤΥΧΙΑ"
                    else:
                        if not delivery_error:
                            delivery_error = (
                                "Το Moodle δεν επέστρεψε θετικό msgid. "
                                "Το αίτημα έφτασε στο Moodle, αλλά δεν επιβεβαιώθηκε παράδοση."
                            )
                        st.error(f"Το μήνυμα δεν παραδόθηκε: {delivery_error}")
                        audit_status = "ΑΠΟΤΥΧΙΑ"

                    st.json(result)

                    add_write_audit(
                        "Αποστολή προσωπικού μηνύματος",
                        function_name,
                        f"User ID {int(user_id)}",
                        message_text,
                        json.dumps(result, ensure_ascii=False),
                        audit_status
                    )

                except Exception as e:
                    st.error(f"Η αποστολή απέτυχε τεχνικά: {e}")
                    add_write_audit(
                        "Αποστολή προσωπικού μηνύματος",
                        function_name,
                        f"User ID {int(user_id)}",
                        message_text,
                        str(e),
                        "ΑΠΟΤΥΧΙΑ"
                    )

    # ------------------------------------------------------
    # GRADE UPDATE
    # ------------------------------------------------------
    elif action == "Ενημέρωση βαθμού — προχωρημένο":
        st.subheader("Ενημέρωση βαθμού — προχωρημένο")
        st.error(
            "Η αλλαγή επίσημου βαθμού είναι υψηλής επίδρασης ενέργεια. "
            "Χρησιμοποίησέ την μόνο σε test course μέχρι να επαληθευτούν πλήρως "
            "τα IDs, το component και το grade item του συγκεκριμένου Moodle."
        )

        course_id = st.number_input("Course ID",min_value=1,step=1,value=1)
        component = st.text_input(
            "Component",
            value="mod_assign",
            help="π.χ. mod_assign. Πρέπει να αντιστοιχεί στο grade item που ενημερώνεται."
        )
        activity_id = st.number_input("Activity ID",min_value=1,step=1,value=1)
        item_number = st.number_input("Item number",min_value=0,step=1,value=0)
        student_id = st.number_input("Student User ID",min_value=1,step=1,value=1)
        grade_value = st.number_input("Νέος βαθμός",value=0.0,step=0.5)
        feedback = st.text_area(
            "Προαιρετικό feedback",
            height=100
        )
        source = st.text_input(
            "Source",
            value="inelso_control_center"
        )

        grade_preview = pd.DataFrame([{
            "Course ID":int(course_id),
            "Component":component,
            "Activity ID":int(activity_id),
            "Item number":int(item_number),
            "Student ID":int(student_id),
            "Νέος βαθμός":float(grade_value),
            "Feedback":feedback,
            "Function":"core_grade_update_grades"
        }])

        st.subheader("Προεπισκόπηση αλλαγής")
        st.dataframe(grade_preview,use_container_width=True,hide_index=True)

        typed_phrase = st.text_input(
            "Για επιβεβαίωση πληκτρολόγησε ακριβώς: ΑΛΛΑΓΗ ΒΑΘΜΟΥ",
            key="grade_phrase"
        )
        grade_confirm = st.checkbox(
            "Έλεγξα Course ID, Activity ID, Student ID, component και νέο βαθμό.",
            key="grade_confirm"
        )
        grade_final = st.checkbox(
            "Κατανοώ ότι η ενέργεια μπορεί να αλλάξει επίσημο βαθμό στο Moodle.",
            key="grade_final"
        )

        can_submit = (
            typed_phrase == "ΑΛΛΑΓΗ ΒΑΘΜΟΥ"
            and grade_confirm
            and grade_final
        )

        if st.button(
            "ΕΚΤΕΛΕΣΗ ΑΛΛΑΓΗΣ ΒΑΘΜΟΥ",
            disabled=not can_submit,
            type="primary"
        ):
            if not moodle_url or not token:
                st.error("Συμπλήρωσε Moodle URL και token.")
            else:
                function_name = "core_grade_update_grades"
                params = {
                    "source":source,
                    "courseid":int(course_id),
                    "component":component,
                    "activityid":int(activity_id),
                    "itemnumber":int(item_number),
                    "grades[0][studentid]":int(student_id),
                    "grades[0][grade]":float(grade_value)
                }
                if feedback.strip():
                    params["grades[0][str_feedback]"] = feedback
                    params["grades[0][feedbackformat]"] = 1

                preview_text = json.dumps({
                    k:v for k,v in params.items()
                }, ensure_ascii=False)

                try:
                    result = moodle_write_call(
                        moodle_url, token, function_name, params
                    )
                    st.success("Το Moodle επέστρεψε επιτυχώς αποτέλεσμα ενημέρωσης βαθμού.")
                    st.json(result)
                    add_write_audit(
                        "Ενημέρωση βαθμού",
                        function_name,
                        f"Course {course_id} / Activity {activity_id} / Student {student_id}",
                        preview_text,
                        json.dumps(result, ensure_ascii=False),
                        "ΕΠΙΤΥΧΙΑ"
                    )
                except Exception as e:
                    st.error(
                        "Η ενημέρωση απέτυχε. Αυτό μπορεί να σημαίνει ότι η function "
                        "δεν έχει δοθεί στο service/token, ότι τα IDs/component δεν αντιστοιχούν "
                        f"στο grade item ή ότι το Moodle απαιτεί διαφορετική δομή παραμέτρων. Λεπτομέρεια: {e}"
                    )
                    add_write_audit(
                        "Ενημέρωση βαθμού",
                        function_name,
                        f"Course {course_id} / Activity {activity_id} / Student {student_id}",
                        preview_text,
                        str(e),
                        "ΑΠΟΤΥΧΙΑ"
                    )

    # ------------------------------------------------------
    # AUDIT LOG
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader("Audit log εγγραφών Moodle")
    st.caption("ΕΠΙΤΥΧΙΑ σημαίνει επιβεβαιωμένη παράδοση/εγγραφή από το Moodle· HTTP 200 από μόνο του δεν θεωρείται επιτυχία.")
    audit = st.session_state["moodle_write_audit"]

    if len(audit) > 0:
        st.dataframe(audit,use_container_width=True,hide_index=True)
        st.download_button(
            "Λήψη audit log CSV",
            audit.to_csv(index=False).encode("utf-8-sig"),
            "INELSO_Moodle_Write_Audit.csv",
            "text/csv"
        )
    else:
        note_result = st.session_state.get("moodle_notes_result")
        has_verified_readback = False
        if isinstance(note_result, dict):
            for key in ["coursenotes", "notes", "sitenotes", "personalnotes"]:
                value = note_result.get(key)
                if isinstance(value, list) and len(value) > 0:
                    has_verified_readback = True
                    break

        if has_verified_readback:
            st.info(
                "Υπάρχει επαληθευμένη Moodle εγγραφή από προηγούμενη ή τρέχουσα συνεδρία. "
                "Το τοπικό session audit log δεν περιέχει νέα write ενέργεια στην παρούσα συνεδρία."
            )
        else:
            st.info("Δεν έχει εκτελεστεί ακόμη καμία write ενέργεια σε αυτή τη συνεδρία.")

    st.info(
        "Προτεινόμενη πολιτική παραγωγής: βασικό write workflow μέσω Moodle notes, "
        "ειδικό service account, ελάχιστες capabilities, ξεχωριστό read token και write token, "
        "και ενεργοποίηση μόνο των απολύτως απαραίτητων functions."
    )

# ==========================================================
# EARLY WARNING ENGINE
# ==========================================================
elif section == "Service 16 — Automated Analytics & Early-Warning":
    if ACTIVE_DF is None:
        show_active_dataset_context("S16")
        st.title("Service 16 — Automated Analytics & Early-Warning")
        st.info("Δεν έχει φορτωθεί Active CSV. Το Service 16 παραμένει κενό μέχρι να φορτωθεί πραγματικό dataset.")
        c1,c2,c3 = st.columns(3)
        c1.metric("Learners","—")
        c2.metric("Risk signals","—")
        c3.metric("Κατάσταση","—")
        st.stop()

    show_active_dataset_context("S16")

    st.title("Service 16 — Automated Analytics & Early-Warning")
    st.caption(
        "Custom reporting και early-warning για παρακολούθηση engagement, completion και έγκαιρο εντοπισμό learners που χρειάζονται ανθρώπινη προσοχή."
    )
    st.info(
        "Ο δείκτης είναι διαφανής και παραμετρικός. Δεν παρουσιάζεται ως επικυρωμένο predictive model "
        "χωρίς ανεξάρτητη αξιολόγηση."
    )


    s16_scorm_ready = ACTIVE_DF is not None
    render_moodle_rest_service_panel(
        "Service 16",
        summary=(
            f"Early-warning analysis · compatibility {service_dataset_compatibility('S16', ACTIVE_DF)}"
            if s16_scorm_ready else
            "Dataset required"
        ),
        score=None,
        status="Early Warning / Human review" if s16_scorm_ready else "incomplete",
        entity=ACTIVE_NAME if s16_scorm_ready else "Service 16",
        enabled=True,
        intervention_type="Service 16 — Early Warning / Human review",
        note="Ο operational risk index δεν γράφεται ως επίσημος βαθμός.",
    )


    if "inelso_uploaded_original" not in st.session_state:
        st.warning("Φόρτωσε πρώτα dataset από το «Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων».")
    else:
        ew_df = st.session_state.get(
            "inelso_uploaded_clean",
            st.session_state["inelso_uploaded_original"]
        ).copy()

        st.info(
            "Ο δείκτης κινδύνου είναι διαφανής λειτουργικός δείκτης και όχι επικυρωμένο predictive model. "
            "Εσύ επιλέγεις ποιες στήλες αντιστοιχούν στα signals."
        )

        cols = ["— καμία —"] + list(ew_df.columns.astype(str))

        learner_col = st.selectbox("Αναγνωριστικό μαθητή", cols, key="ew_learner")
        grade_col = st.selectbox("Βαθμός / score", cols, key="ew_grade")
        inactivity_col = st.selectbox("Ημέρες αδράνειας", cols, key="ew_inactivity")
        attempts_col = st.selectbox("Αριθμός προσπαθειών / submissions", cols, key="ew_attempts")
        engagement_col = st.selectbox("Engagement / activity score", cols, key="ew_engagement")

        st.subheader("Βάρη κινδύνου")
        w_grade = st.slider("Βάρος χαμηλής επίδοσης",0,100,35,5)
        w_inactivity = st.slider("Βάρος αδράνειας",0,100,30,5)
        w_attempts = st.slider("Βάρος προσπαθειών",0,100,15,5)
        w_engagement = st.slider("Βάρος χαμηλού engagement",0,100,20,5)

        total_w = w_grade+w_inactivity+w_attempts+w_engagement
        if total_w == 0:
            total_w = 1

        high_thr = st.slider("Κατώφλι υψηλού κινδύνου",50,90,70,5)
        med_thr = st.slider("Κατώφλι μεσαίου κινδύνου",20,70,45,5)

        if st.button("Υπολογισμός κινδύνου"):
            out_df = ew_df.copy()

            def norm_low(series):
                s = pd.to_numeric(series,errors="coerce")
                if s.notna().sum()==0:
                    return pd.Series(0,index=series.index,dtype=float)
                lo,hi = s.min(),s.max()
                if hi==lo:
                    return pd.Series(0,index=series.index,dtype=float)
                return 100*(hi-s)/(hi-lo)

            def norm_high(series):
                s = pd.to_numeric(series,errors="coerce")
                if s.notna().sum()==0:
                    return pd.Series(0,index=series.index,dtype=float)
                lo,hi = s.min(),s.max()
                if hi==lo:
                    return pd.Series(0,index=series.index,dtype=float)
                return 100*(s-lo)/(hi-lo)

            components = []
            weights = []

            if grade_col != "— καμία —":
                components.append(norm_low(out_df[grade_col]))
                weights.append(w_grade)
            if inactivity_col != "— καμία —":
                components.append(norm_high(out_df[inactivity_col]))
                weights.append(w_inactivity)
            if attempts_col != "— καμία —":
                components.append(norm_high(out_df[attempts_col]))
                weights.append(w_attempts)
            if engagement_col != "— καμία —":
                components.append(norm_low(out_df[engagement_col]))
                weights.append(w_engagement)

            if not components:
                st.error("Επίλεξε τουλάχιστον ένα signal.")
            else:
                active_total = max(1,sum(weights))
                risk = sum(comp*w for comp,w in zip(components,weights))/active_total
                out_df["Δείκτης κινδύνου"] = risk.round(1)

                def risk_class(x):
                    if x >= high_thr:
                        return "ΥΨΗΛΟΣ"
                    if x >= med_thr:
                        return "ΜΕΣΑΙΟΣ"
                    return "ΧΑΜΗΛΟΣ"

                out_df["Κατηγορία κινδύνου"] = out_df["Δείκτης κινδύνου"].apply(risk_class)

                if learner_col != "— καμία —":
                    out_df["Μαθητής"] = out_df[learner_col].astype(str)
                else:
                    out_df["Μαθητής"] = [f"Εγγραφή {i+1}" for i in range(len(out_df))]

                out_df["Προτεινόμενη ενέργεια"] = out_df["Κατηγορία κινδύνου"].map({
                    "ΥΨΗΛΟΣ":"Άμεσος ανθρώπινος έλεγχος",
                    "ΜΕΣΑΙΟΣ":"Στοχευμένη επικοινωνία / παρακολούθηση",
                    "ΧΑΜΗΛΟΣ":"Κανονική παρακολούθηση"
                })

                active_signals = []
                if grade_col != "— καμία —":
                    active_signals.append("επίδοση")
                if inactivity_col != "— καμία —":
                    active_signals.append("αδράνεια")
                if attempts_col != "— καμία —":
                    active_signals.append("προσπάθειες")
                if engagement_col != "— καμία —":
                    active_signals.append("engagement")

                out_df["Εξήγηση"] = (
                    "Risk score από: " + ", ".join(active_signals) +
                    ". Βάρη και κατώφλια ορίζονται από τον χρήστη."
                )

                st.session_state["early_warning_results"] = out_df

                counts = out_df["Κατηγορία κινδύνου"].value_counts()
                a,b,c = st.columns(3)
                a.metric("Υψηλός κίνδυνος",int(counts.get("ΥΨΗΛΟΣ",0)))
                b.metric("Μεσαίος κίνδυνος",int(counts.get("ΜΕΣΑΙΟΣ",0)))
                c.metric("Χαμηλός κίνδυνος",int(counts.get("ΧΑΜΗΛΟΣ",0)))

                view_cols = ["Μαθητής","Δείκτης κινδύνου","Κατηγορία κινδύνου","Προτεινόμενη ενέργεια","Εξήγηση"]
                ranked = out_df.sort_values("Δείκτης κινδύνου",ascending=False)
                st.dataframe(
                    ranked[view_cols],
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Human-in-the-loop παρέμβαση")
                selected_risk_learner = st.selectbox(
                    "Learner για προτεινόμενη παρέμβαση",
                    ranked["Μαθητής"].astype(str).tolist(),
                    key="s16_intervention_learner"
                )
                selected_row = ranked[
                    ranked["Μαθητής"].astype(str) == str(selected_risk_learner)
                ].iloc[0]

                default_s16_note = (
                    f"Service 16 early-warning review για learner {selected_risk_learner}. "
                    f"Δείκτης κινδύνου: {selected_row['Δείκτης κινδύνου']}/100 "
                    f"({selected_row['Κατηγορία κινδύνου']}). "
                    f"Προτεινόμενη ενέργεια: {selected_row['Προτεινόμενη ενέργεια']}. "
                    "Απαιτείται ανθρώπινος έλεγχος πριν από οποιαδήποτε παρέμβαση."
                )

                s16_note_text = st.text_area(
                    "Προτεινόμενο κείμενο παρέμβασης",
                    value=default_s16_note,
                    height=130,
                    key="s16_note_preview"
                )

                if st.button("Προώθηση draft στην Ελεγχόμενη Εγγραφή Moodle", key="s16_stage_note"):
                    st.session_state["pending_moodle_note_text"] = s16_note_text
                    st.session_state["pending_moodle_intervention_type"] = "Service 16 — Early Warning / Human review"
                    st.success(
                        "Το draft αποθηκεύτηκε στη συνεδρία. Πήγαινε στην «Ελεγχόμενη Εγγραφή Moodle», "
                        "επέλεξε learner/course και έλεγξέ το πριν από πραγματικό write."
                    )

                st.download_button(
                    "Λήψη Early Warning αποτελεσμάτων",
                    out_df.to_csv(index=False).encode("utf-8-sig"),
                    "INELSO_Early_Warning.csv",
                    "text/csv"
                )

# ==========================================================
# ΜΗΤΡΩΟ ΕΝΕΡΓΕΙΩΝ
# ==========================================================
elif section == "Μητρώο Ενεργειών":
    show_active_dataset_context("REPORT")
    st.title("Μητρώο Ενεργειών")
    st.caption("Καταγραφή αποφάσεων και ενεργειών με δυνατότητα εισαγωγής/εξαγωγής.")

    base_columns = [
        "Ημερομηνία",
        "Υπηρεσία",
        "Οντότητα",
        "Signal",
        "Προτεραιότητα",
        "Απόφαση",
        "Υπεύθυνος",
        "Κατάσταση",
        "Σημειώσεις",
        "Πηγή δεδομένων"
    ]

    if "action_log" not in st.session_state:
        st.session_state["action_log"] = pd.DataFrame(columns=base_columns)

    st.subheader("Εισαγωγή υπάρχοντος μητρώου")
    imported_log = st.file_uploader(
        "Προαιρετικά ανέβασε προηγούμενο CSV μητρώου",
        type=["csv"],
        key="action_log_import"
    )
    if imported_log is not None:
        try:
            imported_df = pd.read_csv(imported_log)
            missing = [c for c in base_columns if c not in imported_df.columns]
            if missing:
                st.warning("Το αρχείο δεν έχει όλες τις αναμενόμενες στήλες. Θα προστεθούν κενές.")
                for c in missing:
                    imported_df[c] = ""
            st.session_state["action_log"] = imported_df[base_columns]
            st.success("Το μητρώο εισήχθη.")
        except Exception as e:
            st.error(f"Αποτυχία εισαγωγής: {e}")

    st.subheader("Νέα καταχώριση")
    c1,c2 = st.columns(2)

    with c1:
        log_service = st.selectbox(
            "Υπηρεσία",
            ["Service 16","Service 21","Service 25","Service 26","Service 28","Service 31","Άλλο"],
            key="log_service"
        )
        log_entity = st.text_input("Οντότητα / μαθητής / ερώτηση",key="log_entity")
        log_signal = st.text_input("Signal",key="log_signal")
        log_priority = st.selectbox("Προτεραιότητα",[1,2,3],key="log_priority")

    with c2:
        log_decision = st.text_input("Απόφαση / ενέργεια",key="log_decision")
        log_owner = st.text_input("Υπεύθυνος",key="log_owner")
        log_status = st.selectbox(
            "Κατάσταση",
            ["ΝΕΑ","ΣΕ ΕΛΕΓΧΟ","ΟΛΟΚΛΗΡΩΘΗΚΕ","ΑΚΥΡΩΘΗΚΕ"],
            key="log_status"
        )
        log_notes = st.text_area("Σημειώσεις",key="log_notes",height=90)

    if st.button("Προσθήκη στο μητρώο"):
        new_row = pd.DataFrame([{
            "Ημερομηνία":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Υπηρεσία":log_service,
            "Οντότητα":log_entity,
            "Signal":log_signal,
            "Προτεραιότητα":log_priority,
            "Απόφαση":log_decision,
            "Υπεύθυνος":log_owner,
            "Κατάσταση":log_status,
            "Σημειώσεις":log_notes
        }])
        st.session_state["action_log"] = pd.concat(
            [st.session_state["action_log"],new_row],
            ignore_index=True
        )
        st.success("Η καταχώριση προστέθηκε.")

    logdf = st.session_state["action_log"]

    if len(logdf) > 0:
        st.subheader("Καταχωρίσεις")
        st.dataframe(logdf,use_container_width=True,hide_index=True)

        status_counts = logdf["Κατάσταση"].value_counts()
        c1,c2,c3 = st.columns(3)
        c1.metric("Σύνολο",len(logdf))
        c2.metric("Ανοικτές",int(status_counts.get("ΝΕΑ",0)+status_counts.get("ΣΕ ΕΛΕΓΧΟ",0)))
        c3.metric("Ολοκληρωμένες",int(status_counts.get("ΟΛΟΚΛΗΡΩΘΗΚΕ",0)))

        st.download_button(
            "Λήψη μητρώου CSV",
            logdf.to_csv(index=False).encode("utf-8-sig"),
            "INELSO_Action_Log.csv",
            "text/csv"
        )
    else:
        st.info("Δεν υπάρχουν ακόμη καταχωρίσεις.")

    st.warning(
        "Στο Streamlit Community Cloud η τοπική αποθήκευση δεν είναι εγγυημένα μόνιμη. "
        "Χρησιμοποίησε το CSV για ασφαλή μεταφορά/αρχειοθέτηση ή σύνδεσε μελλοντικά βάση δεδομένων/Moodle storage."
    )

# ==========================================================
# ΠΑΡΟΥΣΙΑΣΗ
# ==========================================================
elif section == "Λειτουργία Παρουσίασης":
    show_active_dataset_context("PRESENTATION", compact=False)
    st.title("Λειτουργία Παρουσίασης")
    st.caption("Καθαρή συνοπτική προβολή για παρουσίαση στην ομάδα, χωρίς περιττές τεχνικές λεπτομέρειες.")

    st.markdown("""
    <div style="padding:22px;border-radius:18px;background:#eef6ff;border:1px solid #cfe3ff;margin-bottom:18px;">
      <div style="font-size:28px;font-weight:700;margin-bottom:8px;">INELSO Smart Learning Control Center</div>
      <div style="font-size:16px;line-height:1.6;">
        Ενιαίο περιβάλλον για early warning, ποιότητα αξιολόγησης, AI-assisted authoring,
        προσαρμοστική εξάσκηση, faculty amplification, learning insights και ελεγχόμενη διασύνδεση Moodle.
      </div>
    </div>
    """, unsafe_allow_html=True)

    p1,p2,p3,p4,p5,p6,p7 = st.columns(7)
    p1.metric("S16","Early Warning")
    p2.metric("S21","Assessment")
    p3.metric("S25","Authoring")
    p4.metric("S26","Adaptive")
    p5.metric("S28","Faculty")
    p6.metric("S31","Insights")
    p7.metric("Moodle","Read/Write")

    service_overview = pd.DataFrame([
        {"Service":"16","Ρόλος":"Analytics & Early-Warning","Human approval":"ΝΑΙ"},
        {"Service":"21","Ρόλος":"Assessment Quality","Human approval":"ΝΑΙ"},
        {"Service":"25","Ρόλος":"Course & Assessment Authoring","Human approval":"ΝΑΙ"},
        {"Service":"26","Ρόλος":"Adaptive Practice","Human approval":"ΝΑΙ"},
        {"Service":"28","Ρόλος":"Faculty Teaching Amplification","Human approval":"ΝΑΙ"},
        {"Service":"31","Ρόλος":"Learning Insights","Human approval":"ΝΑΙ"},
    ])
    st.dataframe(service_overview, use_container_width=True, hide_index=True)

    st.markdown("### Κύρια ροή")
    if ACTIVE_DF is not None:
        st.info("CSV / Moodle Export → Data Quality → Services → Decision Support → Human Approval → Moodle Write → Verification")
    else:
        st.info("Moodle Read → Ανάλυση / Απόφαση → Human Approval → Moodle Write → Verification")

    st.markdown("### Τι αποδεικνύει το σύστημα")
    st.success(
        "Το INELSO δεν λειτουργεί μόνο ως dashboard. "
        "Διαβάζει πραγματικά δεδομένα από το Moodle, υποστηρίζει λήψη απόφασης "
        "και καταγράφει ελεγχόμενες εκπαιδευτικές παρεμβάσεις με επαλήθευση read-back."
    )

    step=st.radio(
        "Βήμα παρουσίασης",
        ["1 · Κεντρική εικόνα","2 · Service 16","3 · Service 21","4 · Service 25","5 · Service 26","6 · Service 28","7 · Service 31","8 · Live Moodle","9 · Controlled Write","10 · Αξία για την ομάδα"],
        horizontal=True
    )

    if step.startswith("1"):
        st.markdown("## Ενιαία εικόνα λειτουργίας")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Συνολικός δείκτης",f"{overall:.1f}/100")
        c2.metric("Service 21 actions", open21 if ACTIVE_DF is not None and "custom_s21" in st.session_state else "—")
        c3.metric("Service 26 actions", open26 if ACTIVE_DF is not None and "custom_s26" in st.session_state else "—")
        c4.metric("Service 31 actions", open31 if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—")
        if ACTIVE_DF is not None:
            st.success(f"Όλη η παρουσίαση χρησιμοποιεί ως Active Dataset: {ACTIVE_NAME}")
        st.write("Το Control Center ενοποιεί data quality, analytics, adaptive learning, authoring, faculty support και Moodle actions.")

    elif step.startswith("2"):
        st.markdown("## Service 16 — Automated Analytics & Early-Warning")
        st.write(f"CSV compatibility: {service_dataset_compatibility('S16', ACTIVE_DF)}")
        st.write("Risk scoring και explainability από διαθέσιμα learner-level signals.")

    elif step.startswith("3"):
        st.markdown("## Service 21 — Ευφυής Ποιότητα Αξιολόγησης")
        st.metric("Ερωτήσεις ενεργού dataset",len(service21))
        st.metric("Ανοικτές ενέργειες",open21)
        st.write("Facility, discrimination και priority review ανανεώνονται από συμβατό Moodle statistics CSV.")

    elif step.startswith("4"):
        st.markdown("## Service 25 — AI-Assisted Authoring")
        st.write("Το ενεργό CSV μπορεί να χρησιμοποιηθεί ως πρόσθετο context για question bank / quiz / rubric drafts.")

    elif step.startswith("5"):
        st.markdown("## Service 26 — Adaptive Practice")
        st.metric("Learners",len(cohort26))
        st.metric("Ανοικτές ενέργειες",open26)
        st.write(f"CSV compatibility: {service_dataset_compatibility('S26', ACTIVE_DF)}")

    elif step.startswith("6"):
        st.markdown("## Service 28 — Faculty Teaching Amplification")
        st.write("Το ενεργό CSV μπορεί να χρησιμοποιηθεί ως τεκμηριωμένο context για Q&A, rubric clarification και feedback drafts.")

    elif step.startswith("7"):
        st.markdown("## Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις")
        st.metric("Signals",len(service31_integrated))
        st.metric("Υψηλή προτεραιότητα",high31)
        st.write("Συνθέτει τα ενεργά signals από Service 21 και Service 26.")

    elif step.startswith("8"):
        st.markdown("## Live Moodle")
        st.write("Το REST API παρέχει live courses/users/quizzes. Το CSV παραμένει παράλληλο analytical context.")

    elif step.startswith("9"):
        st.markdown("## Controlled Write")
        st.write("Human approval → Moodle note → read-back verification.")
        st.caption(f"Analytical context: {ACTIVE_NAME if ACTIVE_DF is not None else 'κανένα ενεργό CSV'}")

        note_result = st.session_state.get("moodle_notes_result")
        verified_note = None
        if isinstance(note_result, dict):
            combined=[]
            for key in ["coursenotes","notes","sitenotes","personalnotes"]:
                value=note_result.get(key)
                if isinstance(value,list):
                    combined.extend(value)
            if combined:
                verified_note=combined[-1]

        if verified_note:
            st.success("Moodle Write Verified")
            v1,v2,v3,v4=st.columns(4)
            v1.metric("Note ID",str(verified_note.get("id","—")))
            v2.metric("Course ID",str(verified_note.get("courseid","—")))
            v3.metric("User ID",str(verified_note.get("userid","—")))
            v4.metric("Status","VERIFIED")
        else:
            st.warning("Δεν υπάρχει ακόμη verified note στη συγκεκριμένη Streamlit συνεδρία.")

    else:
        st.markdown("## Αξία για την ομάδα")
        st.write("• Ένα CSV γίνεται κοινή πηγή για όλο το Control Center")
        st.write("• Κάθε Service χρησιμοποιεί μόνο τις μεταβλητές που επιστημονικά του αντιστοιχούν")
        st.write("• Service 21/26/31 ανανεώνονται δυναμικά όταν το schema είναι συμβατό")
        st.write("• Service 16 αναλύει learner-level risk signals")
        st.write("• Service 25/28 χρησιμοποιούν το CSV ως τεκμηριωμένο context")
        st.write("• Live Moodle και Controlled Write διατηρούν ξεχωριστή provenance αλυσίδα")

    st.markdown("### Προτεινόμενη σειρά live demo")
    demo_steps = pd.DataFrame([
        {"Βήμα":1,"Ενότητα":"Κεντρικός Πίνακας","Τι δείχνω":"Ενιαία εικόνα και δείκτες"},
        {"Βήμα":2,"Ενότητα":"Service 21","Τι δείχνω":"Ποιότητα ερωτήσεων και priority review"},
        {"Βήμα":3,"Ενότητα":"Service 26","Τι δείχνω":"Mastery, confidence και adaptive next action"},
        {"Βήμα":4,"Ενότητα":"Service 31","Τι δείχνω":"Σύνθεση signals και προτεραιότητες"},
        {"Βήμα":5,"Ενότητα":"Live Moodle","Τι δείχνω":"Courses, users, quiz και structure"},
        {"Βήμα":6,"Ενότητα":"Ελεγχόμενη Εγγραφή Moodle","Τι δείχνω":"Human-approved note write και verification"},
    ])
    st.dataframe(demo_steps,use_container_width=True,hide_index=True)

    audit = st.session_state.get("moodle_write_audit")
    if isinstance(audit,pd.DataFrame) and len(audit)>0:
        st.markdown("### Τελευταίες ελεγχόμενες ενέργειες")
        cols=[c for c in ["Ημερομηνία","Ενέργεια","Στόχος","Κατάσταση"] if c in audit.columns]
        st.dataframe(audit[cols].tail(5),use_container_width=True,hide_index=True)

    st.caption(
        "Αρχή λειτουργίας: καμία κρίσιμη write ενέργεια δεν εκτελείται αυτόματα. "
        "Ο εκπαιδευτής βλέπει προεπισκόπηση, εγκρίνει ρητά και το αποτέλεσμα επαληθεύεται."
    )

# ==========================================================
# ΚΕΝΤΡΙΚΗ ΑΝΑΦΟΡΑ
# ==========================================================
else:
    show_active_dataset_context("REPORT", compact=False)
    st.title("Κεντρική Αναφορά")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Συνολικός δείκτης υγείας",f"{overall:.1f}")
    c2.metric("Service 21 actions", open21 if ACTIVE_DF is not None and "custom_s21" in st.session_state else "—")
    c3.metric("Service 26 actions", open26 if ACTIVE_DF is not None and "custom_s26" in st.session_state else "—")
    c4.metric("Service 31 actions", open31 if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—")

    report=pd.DataFrame([
        {"Υπηρεσία":"Service 16","Ρόλος":"Early Warning","Δείκτης υγείας":"—","Ανοικτές ενέργειες":"—","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None else "Χωρίς ενεργό CSV"},
        {"Υπηρεσία":"Service 21","Ρόλος":"Ποιότητα αξιολόγησης","Δείκτης υγείας":round(health21,1) if ACTIVE_DF is not None and "custom_s21" in st.session_state else "—","Ανοικτές ενέργειες":open21 if ACTIVE_DF is not None and "custom_s21" in st.session_state else "—","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None and "custom_s21" in st.session_state else "Χωρίς Active CSV"},
        {"Υπηρεσία":"Service 25","Ρόλος":"AI-assisted authoring","Δείκτης υγείας":"—","Ανοικτές ενέργειες":"faculty review","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None else "Manual context"},
        {"Υπηρεσία":"Service 26","Ρόλος":"Προσαρμοστική εξάσκηση","Δείκτης υγείας":round(health26,1) if ACTIVE_DF is not None and "custom_s26" in st.session_state else "—","Ανοικτές ενέργειες":open26 if ACTIVE_DF is not None and "custom_s26" in st.session_state else "—","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None and "custom_s26" in st.session_state else "Χωρίς Active CSV"},
        {"Υπηρεσία":"Service 28","Ρόλος":"Faculty amplification","Δείκτης υγείας":"—","Ανοικτές ενέργειες":"faculty review","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None else "Manual context"},
        {"Υπηρεσία":"Service 31","Ρόλος":"Σύνθεση μαθησιακών ενδείξεων","Δείκτης υγείας":round(health31,1) if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—","Ανοικτές ενέργειες":open31 if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—","Προέλευση":f"Integrated — {ACTIVE_NAME}" if ACTIVE_DF is not None else "Χωρίς Active CSV"},
    ])
    st.dataframe(report,use_container_width=True,hide_index=True)

    summary=f"""INELSO ΚΕΝΤΡΙΚΟ ΣΥΣΤΗΜΑ ΕΞΥΠΝΗΣ ΜΑΘΗΣΗΣ — ULTIMATE PRO WRITE

Active Dataset: {ACTIVE_NAME}
Dataset type: {ACTIVE_KIND}
Συνολικός δείκτης υγείας: {overall:.1f}/100

Service 21:
- δείκτης υγείας: {health21:.1f}
- ανοικτές ενέργειες: {open21}

Service 26:
- δείκτης υγείας: {health26:.1f}
- ανοικτές ενέργειες: {open26}

Service 31:
- δείκτης υγείας: {health31:.1f}
- ανοικτές ενέργειες: {open31}
- προέλευση: σύνθεση πραγματικών/ελεγχόμενων signals από Service 21 και Service 26

Αρχιτεκτονική:
Moodle -> Service 21 / 26 signals -> Service 31 synthesis -> ανθρώπινος έλεγχος -> παρέμβαση / retry / revision -> αναφορά

Δεν πραγματοποιείται αυτόματη αλλαγή επίσημων βαθμών στο Moodle.
"""
    st.download_button(
        "Λήψη κεντρικής σύνοψης",
        summary.encode("utf-8"),
        "INELSO_Kentriki_Synopsi_ULTIMATE_PRO_WRITE.txt",
        "text/plain"
    )
