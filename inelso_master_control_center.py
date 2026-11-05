
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="INELSO Κεντρικό Σύστημα Έξυπνης Μάθησης — Τελική Έκδοση",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
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
# ΚΑΘΟΛΙΚΟ ΕΡΓΑΣΤΗΡΙΟ ΔΕΔΟΜΕΝΩΝ
# ==========================================================
def _κανονικοποίηση_ονόματος_στήλης(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")

def φόρτωση_αρχείου(uploaded_file):
    """Φόρτωση συνηθισμένων tabular formats με ασφαλή μηνύματα σφάλματος."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    try:
        if name.endswith(".csv"):
            # sep=None επιτρέπει αυτόματο delimiter detection για πολλά Moodle exports.
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(raw), sep="\t")
        if name.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        if name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw), engine="xlrd")
        if name.endswith(".ods"):
            return pd.read_excel(io.BytesIO(raw), engine="odf")
        if name.endswith(".json"):
            try:
                obj = json.loads(raw.decode("utf-8-sig"))
                if isinstance(obj, list):
                    return pd.DataFrame(obj)
                if isinstance(obj, dict):
                    # Προσπαθεί να βρει το πρώτο list-of-records.
                    for v in obj.values():
                        if isinstance(v, list):
                            return pd.DataFrame(v)
                    return pd.json_normalize(obj)
            except Exception:
                return pd.read_json(io.BytesIO(raw))
        if name.endswith(".xml"):
            return pd.read_xml(io.BytesIO(raw))
    except UnicodeDecodeError:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="latin-1")
        raise

    raise ValueError("Μη υποστηριζόμενος τύπος αρχείου.")

def αναφορά_ποιότητας(df):
    rows = len(df)
    cols = len(df.columns)
    missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    total_cells = max(rows * cols, 1)
    completeness = round(100 * (1 - missing / total_cells), 2)
    numeric_cols = len(df.select_dtypes(include=np.number).columns)
    text_cols = cols - numeric_cols
    return {
        "γραμμές": rows,
        "στήλες": cols,
        "ελλείπουσες": missing,
        "διπλότυπες_γραμμές": duplicate_rows,
        "πληρότητα": completeness,
        "αριθμητικές_στήλες": numeric_cols,
        "μη_αριθμητικές_στήλες": text_cols,
    }

def μετατροπή_σε_αριθμό_όπου_γίνεται(df, threshold=0.85):
    out = df.copy()
    converted = []
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        s = out[c]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        cand = pd.to_numeric(
            non_null.astype(str)
            .str.replace("%","",regex=False)
            .str.replace(",",".",regex=False)
            .str.strip(),
            errors="coerce"
        )
        ratio = cand.notna().mean()
        if ratio >= threshold:
            full = pd.to_numeric(
                s.astype(str)
                .str.replace("%","",regex=False)
                .str.replace(",",".",regex=False)
                .str.strip(),
                errors="coerce"
            )
            out[c] = full
            converted.append(c)
    return out, converted

def εύρεση_στήλης(df, candidates):
    normalized = {_κανονικοποίηση_ονόματος_στήλης(c): c for c in df.columns}
    for cand in candidates:
        key = _κανονικοποίηση_ονόματος_στήλης(cand)
        if key in normalized:
            return normalized[key]
    # partial matching
    for cand in candidates:
        key = _κανονικοποίηση_ονόματος_στήλης(cand)
        for norm, original in normalized.items():
            if key in norm or norm in key:
                return original
    return None

def προτεινόμενη_αντιστοίχιση_service21(df):
    return {
        "ερώτηση": εύρεση_στήλης(df, [
            "question", "question_name", "questionname", "item", "name",
            "question text", "question_text", "ερώτηση"
        ]),
        "facility": εύρεση_στήλης(df, [
            "facility index", "facility", "facility_index",
            "difficulty", "difficulty_index", "δείκτης ευκολίας"
        ]),
        "διάκριση": εύρεση_στήλης(df, [
            "discrimination index", "discrimination", "discrimination_index",
            "discriminationindex", "διάκριση"
        ]),
        "αποτελεσματικότητα": εύρεση_στήλης(df, [
            "discriminative efficiency", "discrimination efficiency",
            "efficiency", "discriminative_efficiency", "αποτελεσματικότητα"
        ])
    }

def προτεινόμενη_αντιστοίχιση_service26(df):
    return {
        "μαθητής": εύρεση_στήλης(df, [
            "student", "learner", "user", "username", "fullname",
            "participant", "μαθητής"
        ]),
        "προσπάθεια": εύρεση_στήλης(df, [
            "attempt", "attempt_number", "attemptnumber", "προσπάθεια"
        ]),
        "score": εύρεση_στήλης(df, [
            "score", "grade", "finalgrade", "final_grade", "mark", "βαθμός"
        ]),
        "βεβαιότητα": εύρεση_στήλης(df, [
            "confidence", "confidence_score", "βεβαιότητα"
        ]),
        "RC-01": εύρεση_στήλης(df, ["RC-01","RC01","conceptual_accuracy"]),
        "RC-02": εύρεση_στήλης(df, ["RC-02","RC02","method_reasoning"]),
        "RC-03": εύρεση_στήλης(df, ["RC-03","RC03","evidence_support"]),
        "RC-04": εύρεση_στήλης(df, ["RC-04","RC04","completeness"]),
        "RC-05": εύρεση_στήλης(df, ["RC-05","RC05","clarity_communication"]),
    }

def standardize_service21(df, mapping):
    required = ["ερώτηση","facility","διάκριση"]
    if any(not mapping.get(k) for k in required):
        raise ValueError("Για Service 21 απαιτούνται τουλάχιστον: ερώτηση, facility και διάκριση.")
    out = pd.DataFrame({
        "ερώτηση": df[mapping["ερώτηση"]].astype(str),
        "facility": pd.to_numeric(df[mapping["facility"]], errors="coerce"),
        "διάκριση": pd.to_numeric(df[mapping["διάκριση"]], errors="coerce"),
    })
    if mapping.get("αποτελεσματικότητα"):
        out["αποτελεσματικότητα"] = pd.to_numeric(df[mapping["αποτελεσματικότητα"]], errors="coerce")
    else:
        out["αποτελεσματικότητα"] = np.nan
    out = out.dropna(subset=["ερώτηση"]).reset_index(drop=True)
    return out

def standardize_service26(df, mapping):
    if not mapping.get("score"):
        raise ValueError("Για Service 26 απαιτείται τουλάχιστον στήλη score/grade.")
    out = pd.DataFrame()
    if mapping.get("μαθητής"):
        out["μαθητής"] = df[mapping["μαθητής"]].astype(str)
    else:
        out["μαθητής"] = [f"Μαθητής {i+1}" for i in range(len(df))]
    if mapping.get("προσπάθεια"):
        out["προσπάθεια"] = pd.to_numeric(df[mapping["προσπάθεια"]], errors="coerce")
    else:
        out["προσπάθεια"] = 1
    out["score"] = pd.to_numeric(df[mapping["score"]], errors="coerce")
    if mapping.get("βεβαιότητα"):
        out["βεβαιότητα"] = pd.to_numeric(df[mapping["βεβαιότητα"]], errors="coerce")
    else:
        out["βεβαιότητα"] = np.nan

    for rc in ["RC-01","RC-02","RC-03","RC-04","RC-05"]:
        if mapping.get(rc):
            out[rc] = pd.to_numeric(df[mapping[rc]], errors="coerce")
        else:
            out[rc] = np.nan
    return out.dropna(subset=["score"]).reset_index(drop=True)

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


# ==========================================================
# ΕΝΕΡΓΗ ΠΗΓΗ ΔΕΔΟΜΕΝΩΝ
# ==========================================================
if "service21_uploaded_standardized" in st.session_state:
    service21 = st.session_state["service21_uploaded_standardized"].copy()
    service21["κατάσταση"] = service21["διάκριση"].apply(κατηγορία_διάκρισης)
    service21["ενέργεια"] = service21.apply(ενέργεια_ερώτησης, axis=1)

if "service26_uploaded_standardized" in st.session_state:
    uploaded26 = st.session_state["service26_uploaded_standardized"].copy()

    # Αν υπάρχουν rubric columns, δημιουργούμε learner-level κατάσταση.
    if uploaded26["βεβαιότητα"].notna().any():
        def _status_uploaded(r):
            rc01 = r["RC-01"] if pd.notna(r["RC-01"]) else 4
            conf = r["βεβαιότητα"] if pd.notna(r["βεβαιότητα"]) else 0.70
            return κατάσταση_26(r["score"], conf, rc01)
        uploaded26["κατάσταση"] = uploaded26.apply(_status_uploaded, axis=1)
    else:
        uploaded26["κατάσταση"] = np.where(uploaded26["score"] >= 70, "ΕΤΟΙΜΟ", "ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ")

    # Για το cohort χρησιμοποιούμε την πιο πρόσφατη προσπάθεια ανά μαθητή.
    if "προσπάθεια" in uploaded26.columns:
        cohort26 = uploaded26.sort_values(["μαθητής","προσπάθεια"]).groupby("μαθητής", as_index=False).tail(1).copy()
    else:
        cohort26 = uploaded26.copy()

    cohort26["κατάκτηση"] = pd.cut(
        cohort26["score"],
        bins=[-np.inf, 59.999, 79.999, np.inf],
        labels=["Αναπτυσσόμενο","Επαρκής","Κατάκτηση"]
    ).astype(str)

    # Για learner trajectory κρατάμε όλες τις προσπάθειες του πρώτου learner.
    first_learner = uploaded26["μαθητής"].dropna().astype(str).iloc[0] if len(uploaded26) else None
    if first_learner:
        temp = uploaded26[uploaded26["μαθητής"].astype(str)==first_learner].copy()
        if len(temp):
            service26 = temp.rename(columns={"προσπάθεια":"προσπάθεια"}).copy()
            if service26["βεβαιότητα"].isna().all():
                service26["βεβαιότητα"] = 0.70
            for rc in ["RC-01","RC-02","RC-03","RC-04","RC-05"]:
                if service26[rc].isna().all():
                    service26[rc] = 4
            service26["κατάκτηση"] = pd.cut(
                service26["score"],
                bins=[-np.inf, 59.999, 79.999, np.inf],
                labels=["Αναπτυσσόμενο","Επαρκής","Κατάκτηση"]
            ).astype(str)
            service26["κατάσταση"] = service26.apply(
                lambda r: κατάσταση_26(
                    r["score"],
                    r["βεβαιότητα"] if pd.notna(r["βεβαιότητα"]) else 0.70,
                    r["RC-01"] if pd.notna(r["RC-01"]) else 4
                ),
                axis=1
            )

# ==========================================================
# SERVICE 31: ΠΛΕΟΝ ΠΑΡΑΓΕΤΑΙ ΑΠΟ SIGNALS 21 + 26
# ==========================================================
# Μόνο το Q08 έχει τεκμηριωμένη σημασιολογική ετικέτα από το υπάρχον workflow (CTR).
# Τα Q09/Q07 διατηρούνται ως item-level signals χωρίς επινοημένη θεματική ονομασία.

# 1) Signals από Service 21
s21_rows = []
for _, r in service21.iterrows():
    q = str(r["ερώτηση"])
    d = r["διάκριση"]
    if pd.isna(d):
        direction = "ΑΒΕΒΑΙΗ"
        strength = 100.0
        signal_type = "Μη εκτιμήσιμο item-level signal"
        action = "Έλεγχος scoring / response pattern"
    elif d < 20:
        direction = "ΑΡΝΗΤΙΚΗ"
        strength = round(100 - float(d), 2)
        signal_type = "Αδύναμο item-level signal"
        action = "Ανθρώπινος έλεγχος item πριν από θεματική ερμηνεία"
    else:
        direction = "ΘΕΤΙΚΗ"
        strength = round(float(d), 2)
        signal_type = "Item-level signal"
        action = "Διατήρηση / παρακολούθηση"

    # Διατηρούμε την ήδη τεκμηριωμένη Q08 → CTR αντιστοίχιση μόνο όταν υπάρχει.
    entity = "CTR interpretation (Q08)" if "Q08" in q else q
    if "Q08" in q:
        signal_type = "Μαθησιακή έννοια με τεκμηριωμένη αντιστοίχιση"

    s21_rows.append({
        "οντότητα": entity,
        "πηγή":"Service 21",
        "τύπος":signal_type,
        "ισχύς_σήματος":strength,
        "κατεύθυνση":direction,
        "ενέργεια":action,
        "προέλευση":"Ανεβασμένο αρχείο" if "service21_uploaded_standardized" in st.session_state else "Ελεγχόμενο Moodle validation"
    })
s21_signals = pd.DataFrame(s21_rows)

# 2) Signals από Service 26
s26_rows = []
if len(service26) >= 1:
    first = service26.iloc[0]
    latest = service26.iloc[-1]
    for _, rr in rubric.iterrows():
        code = rr["κριτήριο"]
        if code in service26.columns and service26[code].notna().any():
            lv1 = int(first[code]) if pd.notna(first[code]) else 4
            lv2 = int(latest[code]) if pd.notna(latest[code]) else lv1
            gap1 = max(0, 4-lv1)
            gap2 = max(0, 4-lv2)
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
                "προέλευση":"Ανεβασμένο αρχείο" if "service26_uploaded_standardized" in st.session_state else "Ελεγχόμενη adaptive practice"
            })

# Αν δεν έχουμε rubric levels, δημιουργούμε aggregate signal από score/confidence.
if not s26_rows and len(cohort26):
    avg_score = float(pd.to_numeric(cohort26["score"], errors="coerce").mean())
    if "βεβαιότητα" in cohort26.columns:
        conf_series = pd.to_numeric(cohort26["βεβαιότητα"], errors="coerce")
        avg_conf = float(conf_series.mean()) if conf_series.notna().any() else np.nan
    else:
        avg_conf = np.nan

    direction = "ΘΕΤΙΚΗ" if avg_score >= 70 else "ΑΡΝΗΤΙΚΗ"
    strength = round(abs(avg_score - 70), 2)
    s26_rows.append({
        "οντότητα":"Συνολική επίδοση cohort",
        "πηγή":"Service 26",
        "τύπος":"Aggregate performance signal",
        "ισχύς_σήματος":strength,
        "κατεύθυνση":direction,
        "ενέργεια":"Παρακολούθηση" if avg_score >= 70 else "Στοχευμένη επανεξέταση cohort",
        "προέλευση":"Ανεβασμένο αρχείο" if "service26_uploaded_standardized" in st.session_state else "Ελεγχόμενη adaptive practice"
    })
    if not np.isnan(avg_conf):
        s26_rows.append({
            "οντότητα":"Μέση βεβαιότητα cohort",
            "πηγή":"Service 26",
            "τύπος":"Aggregate confidence signal",
            "ισχύς_σήματος":round(abs(avg_conf-.70)*100,2),
            "κατεύθυνση":"ΘΕΤΙΚΗ" if avg_conf >= .70 else "ΑΡΝΗΤΙΚΗ",
            "ενέργεια":"Παρακολούθηση" if avg_conf >= .70 else "Ανθρώπινος έλεγχος",
            "προέλευση":"Ανεβασμένο αρχείο" if "service26_uploaded_standardized" in st.session_state else "Ελεγχόμενη adaptive practice"
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
    - 8*int((service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ").sum())
)

ready_rate = (cohort26["κατάσταση"]=="ΕΤΟΙΜΟ").mean()*100
_conf_series = pd.to_numeric(cohort26["βεβαιότητα"], errors="coerce") if "βεβαιότητα" in cohort26.columns else pd.Series(dtype=float)
avg_conf_for_health = float(_conf_series.mean()) if len(_conf_series) and _conf_series.notna().any() else 0.70
health26 = περιορισμός(
    .55*pd.to_numeric(cohort26["score"], errors="coerce").mean()
    + .25*(avg_conf_for_health*100)
    + .20*ready_rate
)

high31 = int((service31_integrated["προτεραιότητα"]=="ΥΨΗΛΗ").sum())
medium31 = int((service31_integrated["προτεραιότητα"]=="ΜΕΣΑΙΑ").sum())
health31 = περιορισμός(100 - high31*18 - medium31*8)

overall = round((health21+health26+health31)/3,1)

open21 = int(
    ((service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
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
        "Εργαστήριο Δεδομένων",
        "Service 21 — Ευφυής Ποιότητα Αξιολόγησης",
        "Service 26 — Ευφυής Προσαρμοστική Εξάσκηση",
        "Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις",
        "Διασύνδεση Υπηρεσιών",
        "Κύκλος Ζωής Ενεργειών",
        "Μεθοδολογία & Προέλευση",
        "Έλεγχος Ετοιμότητας",
        "Λειτουργία Παρουσίασης",
        "Κεντρική Αναφορά"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Service 31: παράγεται από πραγματικά/ελεγχόμενα signals των Service 21 και 26.")
st.sidebar.caption("Δεν πραγματοποιείται αυτόματη εγγραφή επίσημων βαθμών.")

# ==========================================================
# ΚΕΝΤΡΙΚΟΣ ΠΙΝΑΚΑΣ
# ==========================================================
if section == "Κεντρικός Πίνακας Ελέγχου":
    st.markdown("""
    <div class="hero">
      <h1>INELSO Κεντρικό Σύστημα Έξυπνης Μάθησης</h1>
      <p>Ποιότητα αξιολόγησης + προσαρμοστική εξάσκηση + μαθησιακές ενδείξεις, σε μία ενιαία αλυσίδα τεκμηρίωσης.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Συνολικός δείκτης υγείας", f"{overall:.1f}/100")
    c2.metric("Service 21", f"{health21:.1f}")
    c3.metric("Service 26", f"{health26:.1f}")
    c4.metric("Service 31", f"{health31:.1f}")
    c5.metric("Ανοικτές ενέργειες", open21+open26+open31)

    st.markdown(
        '<div class="flow">Moodle → Service 21 / 26 signals → Service 31 synthesis → '
        'διαφανείς κανόνες → ανθρώπινος έλεγχος → παρέμβαση / retry / revision → αναφορά</div>',
        unsafe_allow_html=True
    )

    st.subheader("Σύγκριση υπηρεσιών")
    comparison = pd.DataFrame([
        {"Υπηρεσία":"Service 21","Ρόλος":"Ποιότητα αξιολόγησης","Δείκτης υγείας":round(health21,1),"Ανοικτές ενέργειες":open21,"Πηγή":"Ελεγχόμενο Moodle validation"},
        {"Υπηρεσία":"Service 26","Ρόλος":"Προσαρμοστική εξάσκηση","Δείκτης υγείας":round(health26,1),"Ανοικτές ενέργειες":open26,"Πηγή":"Ελεγχόμενη adaptive practice"},
        {"Υπηρεσία":"Service 31","Ρόλος":"Σύνθεση μαθησιακών ενδείξεων","Δείκτης υγείας":round(health31,1),"Ανοικτές ενέργειες":open31,"Πηγή":"Παράγωγο των signals Service 21 + 26"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ==========================================================
# ΕΡΓΑΣΤΗΡΙΟ ΔΕΔΟΜΕΝΩΝ
# ==========================================================
elif section == "Εργαστήριο Δεδομένων":
    st.title("Εργαστήριο Δεδομένων")
    st.caption("Ανέβασμα, έλεγχος, καθαρισμός, ανάλυση και σύνδεση αρχείων με τα Service 21 / 26 / 31.")

    uploaded = st.file_uploader(
        "Επιλέξτε αρχείο",
        type=["csv","tsv","xlsx","xls","ods","json","xml"],
        help="Υποστηρίζονται CSV/TSV, Excel, ODS, JSON και XML. Τα Moodle exports σε CSV/Excel μπορούν να αναλυθούν εδώ."
    )

    if uploaded is None:
        st.info("Ανεβάστε ένα αρχείο για να ενεργοποιηθεί το εργαστήριο.")
        if "ενεργό_dataset" in st.session_state:
            st.success("Υπάρχει ήδη ενεργό dataset από προηγούμενο ανέβασμα στη συγκεκριμένη συνεδρία.")
            st.dataframe(st.session_state["ενεργό_dataset"].head(20), use_container_width=True)
    else:
        try:
            df_raw = φόρτωση_αρχείου(uploaded)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            st.session_state["ενεργό_dataset_raw"] = df_raw.copy()

            st.success(f"Το αρχείο «{uploaded.name}» φορτώθηκε επιτυχώς.")

            quality = αναφορά_ποιότητας(df_raw)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Γραμμές", quality["γραμμές"])
            c2.metric("Στήλες", quality["στήλες"])
            c3.metric("Πληρότητα", f"{quality['πληρότητα']:.2f}%")
            c4.metric("Ελλείπουσες τιμές", quality["ελλείπουσες"])
            c5.metric("Διπλότυπες γραμμές", quality["διπλότυπες_γραμμές"])

            tabs = st.tabs([
                "Προεπισκόπηση",
                "Ποιότητα δεδομένων",
                "Καθαρισμός",
                "Περιγραφική ανάλυση",
                "Φίλτρα & ομαδοποίηση",
                "Συσχετίσεις",
                "Σύνδεση με Service 21",
                "Σύνδεση με Service 26",
                "Εξαγωγές"
            ])

            # ------------------- Προεπισκόπηση -------------------
            with tabs[0]:
                st.subheader("Προεπισκόπηση")
                n = st.slider("Πλήθος γραμμών προεπισκόπησης", 5, min(200, max(5,len(df_raw))), min(30,max(5,len(df_raw))))
                st.dataframe(df_raw.head(n), use_container_width=True)
                dtype_table = pd.DataFrame({
                    "Στήλη": df_raw.columns,
                    "Τύπος": [str(df_raw[c].dtype) for c in df_raw.columns],
                    "Μη κενές": [int(df_raw[c].notna().sum()) for c in df_raw.columns],
                    "Μοναδικές": [int(df_raw[c].nunique(dropna=True)) for c in df_raw.columns]
                })
                st.dataframe(dtype_table, use_container_width=True, hide_index=True)

            # ------------------- Ποιότητα -------------------
            with tabs[1]:
                st.subheader("Αναφορά ποιότητας")
                miss = pd.DataFrame({
                    "Στήλη": df_raw.columns,
                    "Κενές": [int(df_raw[c].isna().sum()) for c in df_raw.columns],
                    "Κενές (%)": [round(100*df_raw[c].isna().mean(),2) for c in df_raw.columns],
                    "Μοναδικές": [int(df_raw[c].nunique(dropna=True)) for c in df_raw.columns]
                }).sort_values("Κενές (%)", ascending=False)
                st.dataframe(miss, use_container_width=True, hide_index=True)

                constant_cols = [c for c in df_raw.columns if df_raw[c].nunique(dropna=True) <= 1]
                if constant_cols:
                    st.warning("Στήλες με σταθερή ή σχεδόν κενή πληροφορία: " + ", ".join(map(str, constant_cols)))
                else:
                    st.success("Δεν εντοπίστηκαν στήλες με μία μόνο μοναδική τιμή.")

            # ------------------- Καθαρισμός -------------------
            with tabs[2]:
                st.subheader("Καθαρισμός και προετοιμασία")
                working = df_raw.copy()

                if st.checkbox("Αφαίρεση πλήρως κενών γραμμών", value=True):
                    working = working.dropna(how="all")
                if st.checkbox("Αφαίρεση πλήρως κενών στηλών", value=True):
                    working = working.dropna(axis=1, how="all")
                if st.checkbox("Αφαίρεση διπλότυπων γραμμών", value=False):
                    working = working.drop_duplicates()

                if st.checkbox("Αυτόματη μετατροπή αριθμητικών στηλών", value=True):
                    working, converted = μετατροπή_σε_αριθμό_όπου_γίνεται(working)
                    if converted:
                        st.info("Μετατράπηκαν σε αριθμητικές: " + ", ".join(map(str, converted)))

                selected_drop = st.multiselect(
                    "Στήλες προς αφαίρεση",
                    list(working.columns)
                )
                if selected_drop:
                    working = working.drop(columns=selected_drop)

                fill_mode = st.selectbox(
                    "Διαχείριση ελλειπουσών τιμών",
                    ["Καμία αλλαγή","Αφαίρεση γραμμών με κενές τιμές","Αριθμητικές: διάμεσος / κείμενο: συχνότερη τιμή"]
                )
                if fill_mode == "Αφαίρεση γραμμών με κενές τιμές":
                    working = working.dropna()
                elif fill_mode.startswith("Αριθμητικές"):
                    for c in working.columns:
                        if pd.api.types.is_numeric_dtype(working[c]):
                            if working[c].isna().any():
                                working[c] = working[c].fillna(working[c].median())
                        else:
                            if working[c].isna().any():
                                mode = working[c].mode(dropna=True)
                                if len(mode):
                                    working[c] = working[c].fillna(mode.iloc[0])

                st.dataframe(working.head(50), use_container_width=True)
                if st.button("Ορισμός ως ενεργό καθαρισμένο dataset", type="primary"):
                    st.session_state["ενεργό_dataset"] = working.copy()
                    st.success("Το καθαρισμένο dataset ορίστηκε ως ενεργό για την τρέχουσα συνεδρία.")

            # ------------------- Περιγραφική -------------------
            with tabs[3]:
                st.subheader("Περιγραφική ανάλυση")
                numeric = df_raw.select_dtypes(include=np.number)
                if len(numeric.columns):
                    desc = numeric.describe().T.reset_index().rename(columns={"index":"Μεταβλητή"})
                    st.dataframe(desc, use_container_width=True, hide_index=True)

                    col = st.selectbox("Μεταβλητή για κατανομή", list(numeric.columns))
                    st.bar_chart(numeric[col].value_counts(bins=min(20,max(5,int(np.sqrt(len(numeric)))))).sort_index())
                else:
                    st.info("Δεν υπάρχουν αναγνωρισμένες αριθμητικές στήλες.")

                cat_cols = [c for c in df_raw.columns if not pd.api.types.is_numeric_dtype(df_raw[c])]
                if cat_cols:
                    cat = st.selectbox("Κατηγορική μεταβλητή", cat_cols)
                    freq = df_raw[cat].astype(str).value_counts(dropna=False).head(50).rename_axis(cat).reset_index(name="Πλήθος")
                    st.dataframe(freq, use_container_width=True, hide_index=True)
                    st.bar_chart(freq.set_index(cat))

            # ------------------- Φίλτρα -------------------
            with tabs[4]:
                st.subheader("Φίλτρα και ομαδοποίηση")
                col_filter = st.selectbox("Στήλη φίλτρου", ["—"] + list(df_raw.columns))
                filtered = df_raw.copy()
                if col_filter != "—":
                    if pd.api.types.is_numeric_dtype(df_raw[col_filter]):
                        vals = pd.to_numeric(df_raw[col_filter], errors="coerce").dropna()
                        if len(vals):
                            lo, hi = float(vals.min()), float(vals.max())
                            if lo < hi:
                                chosen = st.slider("Εύρος", lo, hi, (lo,hi))
                                filtered = filtered[
                                    pd.to_numeric(filtered[col_filter], errors="coerce").between(chosen[0],chosen[1])
                                ]
                    else:
                        options = sorted(df_raw[col_filter].dropna().astype(str).unique().tolist())
                        selected = st.multiselect("Τιμές", options, default=options[:min(10,len(options))])
                        if selected:
                            filtered = filtered[filtered[col_filter].astype(str).isin(selected)]
                st.metric("Γραμμές μετά το φίλτρο", len(filtered))
                st.dataframe(filtered.head(100), use_container_width=True)

                group_col = st.selectbox("Ομαδοποίηση κατά", ["—"] + list(df_raw.columns))
                num_cols = list(df_raw.select_dtypes(include=np.number).columns)
                if group_col != "—" and num_cols:
                    measure = st.selectbox("Αριθμητική μεταβλητή", num_cols)
                    agg = st.selectbox("Συνάρτηση", ["mean","median","sum","count","min","max"])
                    grouped = filtered.groupby(group_col, dropna=False)[measure].agg(agg).reset_index()
                    st.dataframe(grouped, use_container_width=True, hide_index=True)
                    st.bar_chart(grouped.set_index(group_col))

            # ------------------- Συσχετίσεις -------------------
            with tabs[5]:
                st.subheader("Συσχετίσεις αριθμητικών μεταβλητών")
                numeric = df_raw.select_dtypes(include=np.number)
                if len(numeric.columns) >= 2:
                    method = st.radio("Μέθοδος", ["pearson","spearman"], horizontal=True)
                    corr = numeric.corr(method=method)
                    st.dataframe(corr.round(3), use_container_width=True)

                    pairs = []
                    cols = corr.columns
                    for i in range(len(cols)):
                        for j in range(i+1,len(cols)):
                            val = corr.iloc[i,j]
                            if pd.notna(val):
                                pairs.append({
                                    "Μεταβλητή 1":cols[i],
                                    "Μεταβλητή 2":cols[j],
                                    "Συσχέτιση":round(float(val),3),
                                    "|r|":round(abs(float(val)),3)
                                })
                    if pairs:
                        pair_df = pd.DataFrame(pairs).sort_values("|r|",ascending=False)
                        st.dataframe(pair_df.head(30), use_container_width=True, hide_index=True)
                else:
                    st.info("Απαιτούνται τουλάχιστον δύο αριθμητικές μεταβλητές.")

            # ------------------- Mapping S21 -------------------
            with tabs[6]:
                st.subheader("Σύνδεση αρχείου με Service 21")
                suggested = προτεινόμενη_αντιστοίχιση_service21(df_raw)
                cols = ["—"] + list(df_raw.columns)

                def idx_for(v):
                    return cols.index(v) if v in cols else 0

                m_q = st.selectbox("Στήλη ερώτησης / item", cols, index=idx_for(suggested["ερώτηση"]), key="s21q")
                m_f = st.selectbox("Στήλη facility / δείκτη ευκολίας", cols, index=idx_for(suggested["facility"]), key="s21f")
                m_d = st.selectbox("Στήλη discrimination / δείκτη διάκρισης", cols, index=idx_for(suggested["διάκριση"]), key="s21d")
                m_e = st.selectbox("Στήλη discriminative efficiency (προαιρετική)", cols, index=idx_for(suggested["αποτελεσματικότητα"]), key="s21e")

                if st.button("Χρήση αυτού του αρχείου στο Service 21", type="primary"):
                    mapping = {
                        "ερώτηση":None if m_q=="—" else m_q,
                        "facility":None if m_f=="—" else m_f,
                        "διάκριση":None if m_d=="—" else m_d,
                        "αποτελεσματικότητα":None if m_e=="—" else m_e,
                    }
                    try:
                        standardized = standardize_service21(df_raw, mapping)
                        st.session_state["service21_uploaded_standardized"] = standardized
                        st.session_state["service21_source_name"] = uploaded.name
                        st.success("Το Service 21 συνδέθηκε με το ανεβασμένο αρχείο. Η εφαρμογή θα ανανεωθεί.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            # ------------------- Mapping S26 -------------------
            with tabs[7]:
                st.subheader("Σύνδεση αρχείου με Service 26")
                suggested = προτεινόμενη_αντιστοίχιση_service26(df_raw)
                cols = ["—"] + list(df_raw.columns)

                def idx2(v):
                    return cols.index(v) if v in cols else 0

                mapping_ui = {}
                fields = [
                    ("μαθητής","Στήλη μαθητή / learner"),
                    ("προσπάθεια","Στήλη προσπάθειας"),
                    ("score","Στήλη score / grade"),
                    ("βεβαιότητα","Στήλη βεβαιότητας (προαιρετική)"),
                    ("RC-01","RC-01 (προαιρετική)"),
                    ("RC-02","RC-02 (προαιρετική)"),
                    ("RC-03","RC-03 (προαιρετική)"),
                    ("RC-04","RC-04 (προαιρετική)"),
                    ("RC-05","RC-05 (προαιρετική)")
                ]
                for key,label in fields:
                    mapping_ui[key] = st.selectbox(
                        label, cols, index=idx2(suggested.get(key)),
                        key=f"s26map_{key}"
                    )

                if st.button("Χρήση αυτού του αρχείου στο Service 26", type="primary"):
                    mapping = {k:(None if v=="—" else v) for k,v in mapping_ui.items()}
                    try:
                        standardized = standardize_service26(df_raw, mapping)
                        st.session_state["service26_uploaded_standardized"] = standardized
                        st.session_state["service26_source_name"] = uploaded.name
                        st.success("Το Service 26 συνδέθηκε με το ανεβασμένο αρχείο. Το Service 31 θα ανανεωθεί επίσης.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            # ------------------- Export -------------------
            with tabs[8]:
                st.subheader("Εξαγωγές")
                source_for_export = st.session_state.get("ενεργό_dataset", df_raw)
                st.download_button(
                    "Λήψη ενεργού dataset ως CSV",
                    source_for_export.to_csv(index=False).encode("utf-8-sig"),
                    "INELSO_energo_dataset.csv",
                    "text/csv"
                )
                quality_df = pd.DataFrame([quality])
                st.download_button(
                    "Λήψη αναφοράς ποιότητας",
                    quality_df.to_csv(index=False).encode("utf-8-sig"),
                    "INELSO_anfora_poiotitas.csv",
                    "text/csv"
                )

                if "service21_uploaded_standardized" in st.session_state:
                    st.download_button(
                        "Λήψη standardized Service 21 dataset",
                        st.session_state["service21_uploaded_standardized"].to_csv(index=False).encode("utf-8-sig"),
                        "INELSO_Service21_standardized.csv",
                        "text/csv"
                    )
                if "service26_uploaded_standardized" in st.session_state:
                    st.download_button(
                        "Λήψη standardized Service 26 dataset",
                        st.session_state["service26_uploaded_standardized"].to_csv(index=False).encode("utf-8-sig"),
                        "INELSO_Service26_standardized.csv",
                        "text/csv"
                    )

            st.markdown("---")
            st.subheader("Διαχείριση συνδεδεμένων δεδομένων")
            a,b = st.columns(2)
            with a:
                if "service21_uploaded_standardized" in st.session_state:
                    st.success("Service 21: χρησιμοποιεί ανεβασμένο αρχείο.")
                    if st.button("Επαναφορά Service 21 στα δεδομένα επίδειξης"):
                        st.session_state.pop("service21_uploaded_standardized",None)
                        st.session_state.pop("service21_source_name",None)
                        st.rerun()
            with b:
                if "service26_uploaded_standardized" in st.session_state:
                    st.success("Service 26: χρησιμοποιεί ανεβασμένο αρχείο.")
                    if st.button("Επαναφορά Service 26 στα δεδομένα επίδειξης"):
                        st.session_state.pop("service26_uploaded_standardized",None)
                        st.session_state.pop("service26_source_name",None)
                        st.rerun()

        except Exception as e:
            st.error(f"Δεν ήταν δυνατή η φόρτωση ή επεξεργασία του αρχείου: {e}")


# ==========================================================
# SERVICE 21
# ==========================================================
elif section == "Service 21 — Ευφυής Ποιότητα Αξιολόγησης":
    st.title("Service 21 — Ευφυής Ποιότητα Αξιολόγησης")
    st.caption("Από ψυχομετρικούς δείκτες σε τεκμηριωμένη ουρά ελέγχου ερωτήσεων.")
    if "service21_uploaded_standardized" in st.session_state:
        st.success("Ενεργή πηγή: ανεβασμένο αρχείο — " + st.session_state.get("service21_source_name",""))
    else:
        st.info("Ενεργή πηγή: ελεγχόμενα δεδομένα επίδειξης / Moodle validation.")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ερωτήσεις",len(service21))
    c2.metric("Μέσος δείκτης ευκολίας",f"{service21['facility'].mean():.1f}%")
    c3.metric("Μέσος δείκτης διάκρισης",f"{valid_disc.mean():.1f}%")
    c4.metric("Δείκτης υγείας",f"{health21:.1f}/100")
    c5.metric("Ανοικτές ενέργειες",open21)

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
            (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
            (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
        ].copy()
        q["προτεραιότητα"] = q["κατάσταση"].map({"ΜΗ ΕΚΤΙΜΗΣΙΜΗ":1,"ΕΛΕΓΧΟΣ":2})
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
    st.title("Service 26 — Ευφυής Προσαρμοστική Εξάσκηση")
    st.caption("Rubric-level gaps, κατάκτηση, βεβαιότητα, retry και ανθρώπινη κλιμάκωση.")
    if "service26_uploaded_standardized" in st.session_state:
        st.success("Ενεργή πηγή: ανεβασμένο αρχείο — " + st.session_state.get("service26_source_name",""))
    else:
        st.info("Ενεργή πηγή: ελεγχόμενα δεδομένα adaptive practice.")

    latest = service26.iloc[-1]
    prev = service26.iloc[0]
    gain = latest["score"]-prev["score"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Τρέχουσα επίδοση",f"{latest['score']:.0f}/100")
    c2.metric("Κατάκτηση",latest["κατάκτηση"])
    c3.metric("Βεβαιότητα",f"{latest['βεβαιότητα']:.2f}")
    c4.metric("Πρόοδος",f"+{gain:.2f}")
    c5.metric("Δείκτης υγείας",f"{health26:.1f}/100")

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
elif section == "Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις":
    st.title("Service 31 — Διασυνδεδεμένες Μαθησιακές Ενδείξεις")
    st.caption("Σύνθεση πραγματικών/ελεγχόμενων signals από Service 21 και Service 26 — χωρίς επινοημένες θεματικές αντιστοιχίσεις.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Συνολικά signals",len(service31_integrated))
    c2.metric("Υψηλή προτεραιότητα",high31)
    c3.metric("Μεσαία προτεραιότητα",medium31)
    c4.metric("Δείκτης υγείας",f"{health31:.1f}/100")

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
    st.title("Διασύνδεση Υπηρεσιών")
    st.caption("Μία ενιαία ουρά ενεργειών από Service 21, Service 26 και το διασυνδεδεμένο Service 31.")

    actions=[]
    for _,r in service21[
        (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
        (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
    ].iterrows():
        actions.append({
            "Προτεραιότητα":1 if r["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ" else 2,
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
    st.title("Μεθοδολογία & Προέλευση")
    st.caption("Διαφάνεια για το τι υπολογίζεται, από πού προέρχεται και τι επιτρέπεται να συμπεράνουμε.")

    st.subheader("Προέλευση δεδομένων")
    provenance = pd.DataFrame([
        {
            "Υπηρεσία":"Service 21",
            "Πηγή":"Ελεγχόμενο Moodle validation",
            "Τύπος":"Item-level psychometric signals",
            "Χρήση":"Ποιότητα ερωτήσεων και ουρά ελέγχου",
            "Περιορισμός":"Δεν επιτρέπεται αυτόματη διαγραφή/αλλαγή item"
        },
        {
            "Υπηρεσία":"Service 26",
            "Πηγή":"Ελεγχόμενη adaptive practice",
            "Τύπος":"Rubric levels, score, mastery, confidence",
            "Χρήση":"Retry / enrichment / ανθρώπινος έλεγχος",
            "Περιορισμός":"Δεν αλλάζει επίσημο βαθμό Moodle"
        },
        {
            "Υπηρεσία":"Service 31",
            "Πηγή":"Παράγωγο signals από Service 21 + 26",
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
    st.title("Έλεγχος Ετοιμότητας")
    st.caption("Τεχνικός και λειτουργικός έλεγχος πριν από παρουσίαση ή πιλοτική χρήση.")

    readiness = pd.DataFrame([
        {"Έλεγχος":"Service 21 λειτουργεί","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Ψυχομετρικά signals και ουρά item review διαθέσιμα"},
        {"Έλεγχος":"Service 26 λειτουργεί","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Rubric, mastery, confidence gate και teacher queue διαθέσιμα"},
        {"Έλεγχος":"Service 31 συνδέεται με 21/26","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Integrated signals χωρίς ανεξάρτητα synthetic concept metrics"},
        {"Έλεγχος":"Moodle iframe","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Ίδιο Streamlit URL με ?embed=true"},
        {"Έλεγχος":"Επίσημοι βαθμοί","Κατάσταση":"ΑΣΦΑΛΕΣ","Σχόλιο":"Καμία αυτόματη εγγραφή βαθμών"},
        {"Έλεγχος":"Human-in-the-loop","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Υψηλή αβεβαιότητα/προτεραιότητα οδηγεί σε ανθρώπινο έλεγχο"},
        {"Έλεγχος":"Audit/provenance","Κατάσταση":"ΕΤΟΙΜΟ","Σχόλιο":"Κάθε signal διατηρεί πηγή και τύπο"},
        {"Έλεγχος":"Persistent action storage","Κατάσταση":"ΜΕΛΛΟΝΤΙΚΟ","Σχόλιο":"Η κατάσταση ενεργειών στη vFinal είναι session-only"},
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
# ΠΑΡΟΥΣΙΑΣΗ
# ==========================================================
elif section == "Λειτουργία Παρουσίασης":

    st.title("Λειτουργία Παρουσίασης")
    step=st.radio(
        "Βήμα παρουσίασης",
        ["1 · Πρόβλημα","2 · Service 21","3 · Service 26","4 · Service 31","5 · Διασύνδεση","6 · Διακυβέρνηση","7 · Αξία για τον πελάτη"],
        horizontal=True
    )
    if step.startswith("1"):
        st.markdown("## Τα LMS έχουν δεδομένα. Το δύσκολο είναι να μετατραπούν σε καθαρές αποφάσεις.")
        st.write("Ποια ερώτηση θέλει έλεγχο; Ποιος μαθητής χρειάζεται νέα προσπάθεια; Ποιο signal απαιτεί ανθρώπινη παρέμβαση;")
    elif step.startswith("2"):
        st.markdown("## Service 21: από ψυχομετρικά στατιστικά σε έλεγχο ποιότητας")
        st.metric("Ανοικτές ενέργειες",open21)
    elif step.startswith("3"):
        st.markdown("## Service 26: από rubric σε προσαρμοστική εξάσκηση")
        st.metric("Πρόοδος","+38.75")
        st.metric("Κατάσταση","ΕΤΟΙΜΟ")
    elif step.startswith("4"):
        st.markdown("## Service 31: σύνθεση πραγματικών signals από 21 και 26")
        st.metric("Signals",len(service31_integrated))
        st.metric("Υψηλή προτεραιότητα",high31)
        st.info("Δεν χρησιμοποιούνται ανεξάρτητα synthetic concept metrics στη v6.")
    elif step.startswith("5"):
        st.markdown("## Διασύνδεση")
        st.markdown('<div class="flow">Service 21 + Service 26 → Service 31 synthesis → ενιαία ουρά ενεργειών → ανθρώπινη απόφαση</div>',unsafe_allow_html=True)
    elif step.startswith("6"):
        st.markdown("## Διακυβέρνηση και ασφάλεια")
        st.write("• Human-in-the-loop")
        st.write("• Καμία αυτόματη εγγραφή επίσημων βαθμών")
        st.write("• Διατήρηση προέλευσης κάθε signal")
        st.write("• Διαχωρισμός recommendation από τελική ανθρώπινη απόφαση")
    else:
        st.markdown("## Αξία για τον πελάτη")
        st.write("• Ενιαία εικόνα")
        st.write("• Διαφανείς κανόνες")
        st.write("• Ανθρώπινος έλεγχος όπου απαιτείται")
        st.write("• Καμία αυτόματη αλλαγή επίσημων βαθμών")
        st.write("• Σαφής προέλευση κάθε signal")

# ==========================================================
# ΚΕΝΤΡΙΚΗ ΑΝΑΦΟΡΑ
# ==========================================================
else:
    st.title("Κεντρική Αναφορά")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Συνολικός δείκτης υγείας",f"{overall:.1f}")
    c2.metric("Service 21 actions",open21)
    c3.metric("Service 26 actions",open26)
    c4.metric("Service 31 actions",open31)

    report=pd.DataFrame([
        {"Υπηρεσία":"Service 21","Ρόλος":"Ποιότητα αξιολόγησης","Δείκτης υγείας":round(health21,1),"Ανοικτές ενέργειες":open21,"Προέλευση":"Ελεγχόμενο Moodle validation"},
        {"Υπηρεσία":"Service 26","Ρόλος":"Προσαρμοστική εξάσκηση","Δείκτης υγείας":round(health26,1),"Ανοικτές ενέργειες":open26,"Προέλευση":"Ελεγχόμενη adaptive practice"},
        {"Υπηρεσία":"Service 31","Ρόλος":"Σύνθεση μαθησιακών ενδείξεων","Δείκτης υγείας":round(health31,1),"Ανοικτές ενέργειες":open31,"Προέλευση":"Παράγωγο των Service 21 + 26"},
    ])
    st.dataframe(report,use_container_width=True,hide_index=True)

    summary=f"""INELSO ΚΕΝΤΡΙΚΟ ΣΥΣΤΗΜΑ ΕΞΥΠΝΗΣ ΜΑΘΗΣΗΣ — ΤΕΛΙΚΗ ΕΚΔΟΣΗ

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
        "INELSO_Kentriki_Synopsi_FINAL.txt",
        "text/plain"
    )
