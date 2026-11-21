
from pathlib import Path
import re
import ast
import io
import json
import hashlib
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
    page_title="INELSO Academia — Πλατφόρμα Έξυπνης Μάθησης",
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
    st.markdown("### Σύνδεση με Moodle")
    if note:
        st.caption(note)

    st.caption(
        "Η προτεινόμενη ενέργεια μπορεί να προωθηθεί για ελεγχόμενη καταγραφή στο Moodle, "
        "μόνο μετά από ανθρώπινη επιβεβαίωση."
    )

    if not enabled:
        st.info("Δεν υπάρχει ακόμη κατάλληλο αποτέλεσμα για καταγραφή στο Moodle.")
        return

    button_key = "moodle_rest_" + re.sub(
        r"[^a-zA-Z0-9_]+", "_", service_name
    ).strip("_").lower()

    if st.button(
        f"Προετοιμασία καταγραφής στο Moodle",
        key=button_key,
    ):
        lines = [
            f"Λειτουργία: {service_name}",
        ]
        if entity:
            lines.append(f"Οντότητα: {entity}")
        if status:
            lines.append(f"Κατάσταση: {status}")
        if score is not None:
            lines.append(f"Βαθμολογία: {float(score):.2f}/100")
        lines.append(f"Σύνοψη: {summary}")

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
            "Το αποτέλεσμα προετοιμάστηκε για ελεγχόμενη καταγραφή στο Moodle. "
            "Μεταβείτε στην «Ελεγχόμενη Εγγραφή Moodle» για προεπισκόπηση, "
            "διπλή επιβεβαίωση, καταγραφή και επαλήθευση."
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
.block-container{padding-top:.35rem!important;padding-bottom:3rem;max-width:1550px}
[data-testid="stHeader"]{height:0!important;min-height:0!important;background:transparent!important}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu{display:none!important}
header{visibility:hidden!important;height:0!important}

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

/* Εμπορική έκδοση: καθαρότερη, φιλική εμφάνιση */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f8fbfd 0%,#eef5f9 100%);border-right:1px solid #d8e4ec}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.22rem .15rem;border-radius:9px}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{border-radius:10px;font-weight:650;min-height:2.55rem}
[data-testid="stExpander"]{border:1px solid #dfe8ef;border-radius:14px!important;background:#fff}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e3ebf1;border-radius:14px;padding:.65rem .75rem;box-shadow:0 3px 14px rgba(16,42,67,.045)}
.inelso-card{background:white;border:1px solid #e1e9ef;border-radius:16px;padding:1rem 1.15rem;box-shadow:0 5px 18px rgba(16,42,67,.055);margin:.55rem 0}
.inelso-card h3{margin:.05rem 0 .35rem;color:#173b5e!important}.inelso-card p{margin:.18rem 0;color:#475467}
.inelso-step{display:inline-block;background:#e8f4f6;color:#17616c;border-radius:999px;padding:.2rem .58rem;font-size:.78rem;font-weight:700;margin-right:.25rem}
.inelso-note{background:#fffbea;border:1px solid #f3e3a3;border-radius:12px;padding:.8rem 1rem;color:#594a16}
.inelso-module-head{margin:.05rem 0 .55rem 0!important;padding:1.05rem 1.25rem!important;border-left:6px solid #2a8c99}
.inelso-module-head h2{margin:0!important;font-size:1.55rem!important;line-height:1.2!important;color:#173b5e!important}
.inelso-module-head p{margin:.45rem 0 0!important;font-size:1rem!important;line-height:1.45!important;color:#475467!important}

/* Product-grade polish */
[data-testid="stSidebar"]{min-width:305px!important;max-width:305px!important}
[data-testid="stSidebar"] .stRadio>div{gap:.08rem!important}
[data-testid="stSidebar"] [role="radiogroup"] label{
  border:1px solid transparent!important;
  padding:.42rem .55rem!important;
  margin:.02rem 0!important;
  transition:all .16s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  background:#ffffff!important;border-color:#dce7ee!important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:#ffffff!important;border-color:#cfe0e9!important;
  box-shadow:0 4px 12px rgba(16,42,67,.05)!important;
}
.inelso-status-row{display:flex;gap:.5rem;flex-wrap:wrap;margin:.45rem 0 .9rem}
.inelso-pill{
  display:inline-flex;align-items:center;gap:.35rem;padding:.35rem .65rem;border-radius:999px;
  border:1px solid #dce7ee;background:#fff;color:#344054;font-size:.82rem;font-weight:650
}
.inelso-pill.ok{background:#eefaf5;border-color:#ccebdd;color:#176b4d}
.inelso-pill.warn{background:#fff8e8;border-color:#f1dfac;color:#7a5a0b}
.inelso-pill.neutral{background:#f7f9fb;border-color:#e2e8f0;color:#475467}
.inelso-empty{
  background:linear-gradient(180deg,#ffffff 0%,#f8fbfd 100%);
  border:1px solid #dfe8ef;border-radius:18px;padding:1.2rem 1.3rem;margin:.65rem 0 1rem;
  box-shadow:0 5px 18px rgba(16,42,67,.04)
}
.inelso-empty h3{margin:.05rem 0 .35rem;color:#173b5e!important}
.inelso-source-card{
  border:1px solid #dfe8ef;background:#fff;border-radius:16px;padding:1rem 1.1rem;height:100%;
  box-shadow:0 4px 14px rgba(16,42,67,.035)
}
.inelso-source-card b{font-size:1rem;color:#173b5e}
.inelso-source-card p{margin:.35rem 0 0;color:#667085;font-size:.92rem}
.inelso-kpi-title{color:#667085;font-size:.82rem;font-weight:650;margin-bottom:.18rem}
.inelso-section-label{
  color:#667085;font-size:.75rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
  margin:.7rem 0 .15rem
}
[data-testid="stPopover"] button{white-space:nowrap!important}
div[data-baseweb="popover"]{max-width:560px!important}
.inelso-process{
  display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:.7rem 0 1rem
}
.inelso-process .step{
  border:1px solid #dfe8ef;border-radius:14px;padding:.8rem;background:#fff;
  min-height:86px;box-shadow:0 3px 12px rgba(16,42,67,.03)
}
.inelso-process .step strong{display:block;color:#173b5e;margin-bottom:.2rem}
.inelso-process .step span{color:#667085;font-size:.86rem;line-height:1.35}
.inelso-process .step.active{border-color:#9fd7db;background:#f3fbfb}
.inelso-process .step.safe{border-color:#cae8dc;background:#f4fbf7}
.inelso-exec-card{
  border:1px solid #dfe8ef;border-radius:16px;padding:1rem 1.05rem;background:#fff;
  box-shadow:0 4px 15px rgba(16,42,67,.035);height:100%
}
.inelso-exec-card h4{margin:.05rem 0 .28rem;color:#173b5e}
.inelso-exec-card p{margin:0;color:#667085;font-size:.9rem;line-height:1.45}
.inelso-danger-lite{
  border:1px solid #f0d9ad;background:#fff9ec;border-radius:14px;padding:.8rem 1rem;
  color:#7b5b12;margin:.65rem 0
}
@media(max-width:900px){.inelso-process{grid-template-columns:1fr 1fr}}

/* Premium dashboard / iframe presentation layer */
.inelso-home-hero{
  background:linear-gradient(135deg,#163b5c 0%,#245f86 58%,#2b7894 100%);
  color:white;border-radius:22px;padding:1.55rem 1.7rem;margin:.2rem 0 1rem;
  box-shadow:0 10px 28px rgba(16,42,67,.13);
}
.inelso-home-hero h1{color:#fff!important;margin:0!important;font-size:2rem!important;letter-spacing:-.02em}
.inelso-home-hero p{margin:.55rem 0 0!important;color:rgba(255,255,255,.9)!important;font-size:1rem!important}
.inelso-home-eyebrow{font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;opacity:.78;margin-bottom:.45rem}
.inelso-dashboard-tabs [data-baseweb="tab-list"]{
  gap:.35rem;background:#f5f8fb;padding:.35rem;border:1px solid #e1e9ef;border-radius:14px
}
.inelso-dashboard-tabs [data-baseweb="tab"]{
  height:2.75rem;border-radius:10px;padding:0 1rem;font-weight:700;color:#475467
}
.inelso-dashboard-tabs [aria-selected="true"]{
  background:#fff!important;color:#173b5e!important;box-shadow:0 3px 10px rgba(16,42,67,.07)
}
.inelso-feature-card{
  background:#fff;border:1px solid #dfe8ef;border-radius:18px;padding:1.05rem 1.1rem;
  box-shadow:0 5px 18px rgba(16,42,67,.045);min-height:160px;height:100%
}
.inelso-feature-card .kicker{font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#2a8c99;margin-bottom:.35rem}
.inelso-feature-card h3{margin:.05rem 0 .35rem!important;color:#173b5e!important;font-size:1.08rem!important}
.inelso-feature-card p{margin:0!important;color:#667085!important;font-size:.9rem!important;line-height:1.45!important}
.inelso-feature-value{font-size:1.7rem;font-weight:760;color:#102a43;line-height:1.15;margin:.35rem 0}
.inelso-mini-note{font-size:.78rem;color:#667085;margin-top:.3rem}
.inelso-action-banner{
  background:linear-gradient(180deg,#f8fbff 0%,#f2f7fb 100%);
  border:1px solid #d9e6f0;border-radius:16px;padding:1rem 1.1rem;margin:.65rem 0
}
.inelso-action-banner strong{color:#173b5e}
.inelso-data-strip{
  display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:.65rem 0 1rem
}
.inelso-data-chip{
  border:1px solid #e0e8ef;background:#fff;border-radius:14px;padding:.8rem .9rem
}
.inelso-data-chip .label{font-size:.76rem;color:#667085;font-weight:650}
.inelso-data-chip .value{font-size:1.35rem;color:#173b5e;font-weight:760;margin-top:.15rem}
@media(max-width:900px){
  .inelso-data-strip{grid-template-columns:1fr 1fr}
  .inelso-home-hero{padding:1.2rem}
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ / ΕΠΕΞΗΓΗΣΕΙΣ ΓΙΑ ΜΗ ΕΙΔΙΚΟΥΣ
# ==========================================================
MODULE_HELP = {
    "dashboard": {
        "title": "Κεντρική Εικόνα Ελέγχου",
        "purpose": "Συνοψίζει την κατάσταση των δεδομένων και των επιμέρους λειτουργιών σε μία οθόνη.",
        "steps": [
            "Φορτώστε CSV ή Excel από την «Εισαγωγή CSV ή Excel» ή ενεργοποιήστε πραγματικό κύριο σύνολο δεδομένων SCORM.",
            "Ελέγξτε ποια λειτουργία είναι συμβατή με τις διαθέσιμες στήλες.",
            "Ανοίξτε την αντίστοιχη ενότητα για λεπτομερή αποτελέσματα και προτεινόμενες ενέργειες.",
            "Οι ενέργειες προς Moodle περνούν πάντα από ανθρώπινη επιβεβαίωση."
        ],
        "data": ""
    },
    "s16": {
        "title": "Μαθησιακή Ανάλυση & Έγκαιρη Υποστήριξη",
        "purpose": "Εντοπίζει μοτίβα χαμηλής συμμετοχής ή επίδοσης ώστε ο διδάσκων να μπορεί να παρέμβει έγκαιρα.",
        "steps": [
            "Χρησιμοποιήστε ανώνυμο αναγνωριστικό φοιτητή.",
            "Συμπεριλάβετε, όπου υπάρχουν, επίδοση, ημέρες αδράνειας, συμμετοχή, υποβολές και ποσοστό ολοκλήρωσης.",
            "Ελέγξτε τους φοιτητές που εμφανίζονται με αυξημένη προτεραιότητα.",
            "Η ένδειξη είναι υποστήριξη απόφασης και όχι αυτόματη παιδαγωγική κρίση."
        ],
        "data": ""
    },
    "s21": {
        "title": "Ποιότητα Αξιολόγησης",
        "purpose": "Βοηθά τον διδάσκοντα να εντοπίσει ερωτήσεις κουίζ που είναι υπερβολικά εύκολες/δύσκολες ή δεν διακρίνουν καλά μεταξύ επιπέδων επίδοσης.",
        "steps": [
            "Δημιουργήστε το κουίζ στο Moodle με μία σαφή ερώτηση ανά στοιχείο και ορισμένη σωστή απάντηση/βαθμολόγηση.",
            "Αφήστε να συγκεντρωθεί επαρκής αριθμός πραγματικών προσπαθειών πριν ερμηνεύσετε τους ψυχομετρικούς δείκτες.",
            "Εξαγάγετε από το Moodle τα Πλήρη στατιστικά ή αντίστοιχο αρχείο δεδομένων ανά ερώτηση.",
            "Η πλατφόρμα διαβάζει δείκτη ευκολίας, δείκτη διάκρισης και, όπου υπάρχει, αποτελεσματικότητα διάκρισης.",
            "Οι προτάσεις αναθεώρησης είναι συμβουλευτικές· ο διδάσκων εγκρίνει κάθε αλλαγή."
        ],
        "data": ""
    },
    "s25": {
        "title": "Δημιουργία Μαθήματος & Αξιολόγησης",
        "purpose": "Υποστηρίζει τη δημιουργία προσχεδίων ερωτήσεων, κουίζ, κλιμάκων αξιολόγησης και ασκήσεων από το εκπαιδευτικό υλικό.",
        "steps": [
            "Δώστε το υλικό/περιεχόμενο και έναν σαφή μαθησιακό στόχο.",
            "Επιλέξτε αν θέλετε τράπεζα ερωτήσεων, κουίζ, κλίμακα αξιολόγησης ή ασκήσεις εξάσκησης.",
            "Για την κλίμακα αξιολόγησης ορίστε κριτήρια που αντιστοιχούν σε παρατηρήσιμα στοιχεία της απάντησης και βάρη που αθροίζουν σε 100%.",
            "Ελέγξτε και επεξεργαστείτε το προσχέδιο πριν το χαρακτηρίσετε εγκεκριμένο.",
            "Τίποτα δεν δημοσιεύεται αυτόματα στο μάθημα."
        ],
        "data": ""
    },
    "s26": {
        "title": "Προσαρμοστική Εξάσκηση",
        "purpose": "Χρησιμοποιεί επίδοση, βεβαιότητα και κλίμακα αξιολόγησης για να προτείνει νέα προσπάθεια, ανθρώπινο έλεγχο ή επόμενη δραστηριότητα.",
        "steps": [
            "Δημιουργήστε κλίμακα αξιολόγησης πριν από την αξιολόγηση. Κάθε κριτήριο πρέπει να έχει σαφή περιγραφή και κλίμακα, π.χ. 0–4.",
            "Η προτεινόμενη κλίμακα αξιολόγησης της πλατφόρμας έχει: RC-01 Εννοιολογική ακρίβεια 30%, RC-02 Μεθοδολογία 25%, RC-03 Τεκμηρίωση 15%, RC-04 Πληρότητα 15%, RC-05 Σαφήνεια 15%.",
            "Στο αρχείο κάθε γραμμή αντιστοιχεί σε έναν φοιτητή και μία προσπάθεια. Κρατήστε σταθερό το αναγνωριστικό εκπαιδευόμενου και αυξήστε τον αριθμό προσπάθειας σε κάθε νέα προσπάθεια.",
            "Η βαθμολογία είναι 0–100 και η βεβαιότητα 0–1. Τα RC-01…RC-05 περιέχουν το επίπεδο επίδοσης κάθε κριτηρίου.",
            "Βεβαιότητα < 0,70 ή RC-01 στο 0–1 οδηγεί σε ανθρώπινο έλεγχο. Επίδοση < 70 με επαρκή βεβαιότητα οδηγεί σε νέα προσπάθεια."
        ],
        "data": ""
    },
    "s28": {
        "title": "Υποστήριξη Διδάσκοντα",
        "purpose": "Παράγει ελεγχόμενα προσχέδια απαντήσεων, διευκρινίσεων κλίμακας αξιολόγησης και ανατροφοδότησης για να μειώνει επαναλαμβανόμενη εργασία.",
        "steps": [
            "Δώστε την ερώτηση του φοιτητή και το εγκεκριμένο πλαίσιο/πολιτική του μαθήματος.",
            "Επιλέξτε τύπο υποστήριξης: απάντηση ρουτίνας, διευκρίνιση κλίμακας αξιολόγησης ή ανατροφοδότηση.",
            "Ελέγξτε το προσχέδιο και προσαρμόστε το στον τόνο και στους κανόνες του μαθήματος.",
            "Η αποστολή/καταγραφή γίνεται μόνο μετά από έγκριση του διδάσκοντα."
        ],
        "data": "Χρήσιμα πεδία: ερώτηση φοιτητή, πλαίσιο διδάσκοντα, θεματική μαθήματος και, όταν αφορά αξιολόγηση, η εγκεκριμένη κλίμακα αξιολόγησης."
    },
    "s31": {
        "title": "Ενοποιημένες Μαθησιακές Ενδείξεις",
        "purpose": "Συγκεντρώνει πραγματικές ενδείξεις από τις προηγούμενες αναλύσεις και τις οργανώνει σε κοινή προτεραιότητα δράσης.",
        "steps": [
            "Φορτώστε πρώτα δεδομένα που υποστηρίζουν Ποιότητα Αξιολόγησης ή/και Προσαρμοστική Εξάσκηση.",
            "Η ενότητα δεν δημιουργεί τεχνητά ενδείξεις όταν δεν υπάρχουν δεδομένα.",
            "Ελέγξτε την πηγή κάθε ένδειξης πριν αποφασίσετε ενέργεια.",
            "Χρησιμοποιήστε την Ενιαία Ροή Ενεργειών για συγκεντρωτική διαχείριση."
        ],
        "data": "Οι ενδείξεις διατηρούν την προέλευσή τους ώστε να είναι ελέγξιμες και εξηγήσιμες."
    },
    "csv_import": {
        "title": "Εισαγωγή CSV ή Excel",
        "purpose": "Φορτώνει εξωτερικά ή χειροκίνητα αρχεία δεδομένων ως εναλλακτική/συμπληρωματική πηγή για την πλατφόρμα.",
        "steps": [
            "Ανεβάστε CSV, Excel, JSON ή XML με σαφείς επικεφαλίδες στηλών.",
            "Ελέγξτε προεπισκόπηση, κενά, διπλοεγγραφές και τύπους μεταβλητών.",
            "Χρησιμοποιήστε την αυτόματη αναγνώριση για να δείτε ποιες λειτουργίες υποστηρίζονται από το αρχείο.",
            "Το καθαρισμένο αρχείο γίνεται το ενεργό σύνολο δεδομένων για όλη την πλατφόρμα."
        ],
        "data": "Η εισαγωγή CSV ή Excel παραμένει ανεξάρτητη από τη ζωντανή ανάκτηση SCORM και είναι χρήσιμη για εξωτερικά σύνολα δεδομένων ή πρόσθετες μεταβλητές."
    },
    "scorm_analysis": {
        "title": "Ανάλυση SCORM Αξιολόγησης",
        "purpose": "Ανακτά πραγματικές προσπάθειες από SCORM δραστηριότητες του Moodle και μετατρέπει τις απαντήσεις ανά ερώτηση σε δείκτες ποιότητας αξιολόγησης.",
        "steps": [
            "Επιλέξτε μάθημα και SCORM δραστηριότητα.",
            "Η πλατφόρμα ανακτά τις υπάρχουσες προσπάθειες των εγγεγραμμένων χρηστών μέσω της διασύνδεσης REST του Moodle.",
            "Οι απαντήσεις ανά ερώτηση μετατρέπονται σε ποσοστό επιτυχίας και, όταν υπάρχει επαρκές δείγμα, δείκτη διάκρισης.",
            "Πατήστε «Ενημέρωση Ποιότητας Αξιολόγησης» ώστε τα πραγματικά SCORM αποτελέσματα να γίνουν η ενεργή πηγή της Ποιότητας Αξιολόγησης.",
            "Δεν αλλάζει κανένας βαθμός ή προσπάθεια SCORM· η λειτουργία είναι μόνο για ανάγνωση και ανάλυση."
        ],
        "data": ""
    },
    "flow": {
        "title": "Ενιαία Ροή Ενεργειών",
        "purpose": "Συγκεντρώνει σε ένα σημείο τις ενδείξεις και τις προτεινόμενες ενέργειες ώστε ο διδάσκων να αποφασίζει τι θα γίνει πρώτα.",
        "steps": [
            "Ελέγξτε ποιες ενδείξεις έχουν δημιουργηθεί από τις επιμέρους λειτουργίες.",
            "Δείτε προτεραιότητα, πηγή και προτεινόμενη ενέργεια για κάθε περίπτωση.",
            "Μην εκτελείτε ενέργεια μόνο επειδή εμφανίζεται ως υψηλής προτεραιότητας· επιβεβαιώστε το εκπαιδευτικό πλαίσιο.",
            "Όταν αποφασίσετε να προχωρήσετε, μεταφέρετε την ενέργεια στον ελεγχόμενο κύκλο έγκρισης/καταγραφής."
        ],
        "data": "Δεν απαιτεί ξεχωριστό αρχείο. Χρησιμοποιεί τις ενδείξεις που έχουν ήδη δημιουργηθεί από το ενεργό σύνολο δεδομένων."
    },
    "lifecycle": {
        "title": "Κύκλος Ζωής Ενεργειών",
        "purpose": "Παρακολουθεί μια ενέργεια από την αρχική ένδειξη μέχρι τον ανθρώπινο έλεγχο, την έγκριση, την εκτέλεση και την επαλήθευση.",
        "steps": [
            "Επιλέξτε την ενέργεια που θέλετε να παρακολουθήσετε.",
            "Ελέγξτε σε ποια κατάσταση βρίσκεται: νέα, υπό εξέταση, εγκεκριμένη, εκτελεσμένη ή επαληθευμένη.",
            "Καταγράψτε υπεύθυνο και σχόλια όπου χρειάζεται.",
            "Θεωρήστε την ενέργεια ολοκληρωμένη μόνο όταν υπάρχει επιβεβαίωση του αποτελέσματος."
        ],
        "data": "Η ενότητα χρησιμοποιεί το μητρώο ενεργειών και τις αποφάσεις που δημιουργούνται μέσα στην πλατφόρμα."
    },
    "methodology": {
        "title": "Μεθοδολογία & Προέλευση",
        "purpose": "Εξηγεί από πού προέρχεται κάθε ένδειξη, ποιος κανόνας ή υπολογισμός χρησιμοποιήθηκε και ποιοι περιορισμοί ισχύουν.",
        "steps": [
            "Χρησιμοποιήστε την ενότητα όταν θέλετε να ελέγξετε πώς προέκυψε ένα αποτέλεσμα.",
            "Ελέγξτε την πηγή δεδομένων και αν το αρχείο είναι πραγματικό, δοκιμαστικό ή εξαγωγή Moodle.",
            "Διαβάστε τους κανόνες/όρια πριν ερμηνεύσετε μια ένδειξη.",
            "Σε παρουσίαση ή έλεγχο, χρησιμοποιήστε την προέλευση για τεκμηρίωση και διαφάνεια."
        ],
        "data": "Η προέλευση συνδέει το αποτέλεσμα με το ενεργό αρχείο, τις σχετικές στήλες και τη λειτουργία που το παρήγαγε."
    },
    "data_center": {
        "title": "Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων",
        "purpose": "Είναι το σημείο από όπου φορτώνεται το αρχείο που χρησιμοποιεί ολόκληρη η πλατφόρμα ως ενεργό σύνολο δεδομένων.",
        "steps": [
            "Ανεβάστε CSV ή το υποστηριζόμενο αρχείο εξαγωγής με σαφείς επικεφαλίδες στηλών.",
            "Ελέγξτε Προεπισκόπηση και Ποιότητα δεδομένων για κενά, διπλότυπα και ασυνήθιστες τιμές.",
            "Χρησιμοποιήστε την Αυτόματη αναγνώριση για να δείτε ποιες λειτουργίες μπορούν να αξιοποιήσουν το αρχείο.",
            "Μην αλλάζετε αυθαίρετα ονόματα/τιμές εάν πρόκειται για επίσημη εξαγωγή· κρατήστε αντίγραφο του αρχικού αρχείου."
        ],
        "data": "Το ενεργό αρχείο εμφανίζεται στην αριστερή στήλη και είναι κοινό για όλες τις λειτουργίες. Κάθε λειτουργία χρησιμοποιεί μόνο τα πεδία που είναι κατάλληλα για αυτή."
    },
    "analytics_lab": {
        "title": "Εργαστήριο Ανάλυσης Δεδομένων",
        "purpose": "Επιτρέπει διερευνητική ανάλυση του ενεργού αρχείου χωρίς να απαιτείται εξωτερικό στατιστικό πρόγραμμα.",
        "steps": [
            "Επιλέξτε τις μεταβλητές που θέλετε να εξετάσετε.",
            "Χρησιμοποιήστε περιγραφικά στατιστικά και γραφήματα για αρχική κατανόηση.",
            "Ελέγξτε ποιότητα και τύπους δεδομένων πριν εφαρμόσετε πιο σύνθετη ανάλυση.",
            "Μην ερμηνεύετε συσχετίσεις ως αιτιώδεις σχέσεις χωρίς κατάλληλο ερευνητικό σχεδιασμό."
        ],
        "data": "Χρησιμοποιεί το ενεργό σύνολο δεδομένων. Οι διαθέσιμες αναλύσεις εξαρτώνται από τον τύπο και την ποιότητα των στηλών."
    },
    "whatif": {
        "title": "Προσομοίωση Σεναρίων",
        "purpose": "Δείχνει πώς μπορεί να αλλάξει μια ένδειξη όταν μεταβάλλονται συγκεκριμένες τιμές, χωρίς να αλλάζει τα πραγματικά δεδομένα ή τους επίσημους βαθμούς.",
        "steps": [
            "Επιλέξτε την περίπτωση ή τις μεταβλητές που θέλετε να εξετάσετε.",
            "Αλλάξτε μία ή περισσότερες τιμές για να δημιουργήσετε υποθετικό σενάριο.",
            "Συγκρίνετε το υποθετικό αποτέλεσμα με την αρχική κατάσταση.",
            "Χρησιμοποιήστε το αποτέλεσμα για συζήτηση/σχεδιασμό και όχι ως πραγματική καταγραφή."
        ],
        "data": "Οι προσομοιώσεις είναι υποθετικές και δεν γράφουν αλλαγές στο αρχικό CSV ή στο Moodle."
    },
    "moodle_live": {
        "title": "Ζωντανή Σύνδεση Moodle",
        "purpose": "Ελέγχει τη σύνδεση με το Moodle και διαβάζει επιτρεπόμενες πληροφορίες μέσω της επίσημης διασύνδεσης REST.",
        "steps": [
            "Επιβεβαιώστε τη διεύθυνση Moodle και ότι το υπηρεσία ιστού είναι ενεργό.",
            "Χρησιμοποιήστε λογαριασμό υπηρεσίας με μόνο τα απολύτως απαραίτητα δικαιώματα.",
            "Ελέγξτε πρώτα ανάγνωση μαθημάτων/χρηστών/δραστηριοτήτων.",
            "Η επιτυχής ανάγνωση δεν σημαίνει ότι επιτρέπεται αυτόματη αλλαγή βαθμών ή περιεχομένου."
        ],
        "data": "Η σύνδεση χρησιμοποιεί διασύνδεσης REST του Moodle υπηρεσία ιστούs. Τα δικαιώματα καθορίζονται από τον λογαριασμό και τις ενεργοποιημένες λειτουργίες του Moodle."
    },
    "moodle_write": {
        "title": "Ελεγχόμενη Εγγραφή Moodle",
        "purpose": "Καταγράφει εγκεκριμένες ενέργειες στο Moodle μόνο μετά από ρητή ανθρώπινη επιβεβαίωση και έλεγχο ότι η εγγραφή αποθηκεύτηκε σωστά.",
        "steps": [
            "Ελέγξτε την προεπισκόπηση: μάθημα, χρήστη, τύπο ενέργειας και κείμενο.",
            "Επιβεβαιώστε ότι δεν περιέχει προσωπικά ή λανθασμένα στοιχεία.",
            "Δώστε ρητή έγκριση και εκτελέστε την εγγραφή.",
            "Περιμένετε το επαλήθευση μετά την εγγραφή: επιτυχία θεωρείται μόνο όταν η πληροφορία ανακτηθεί ξανά από το Moodle."
        ],
        "data": "Η πλατφόρμα δεν αλλάζει αυτόματα επίσημους βαθμούς. Η εγγραφή γίνεται μέσω ελεγχόμενων διασύνδεσης REST του Moodle λειτουργιών."
    },
    "registry": {
        "title": "Μητρώο Ενεργειών",
        "purpose": "Κρατά συγκεντρωτικό ιστορικό αποφάσεων, ενεργειών, υπευθύνων και καταστάσεων για ιχνηλασιμότητα.",
        "steps": [
            "Χρησιμοποιήστε φίλτρα ή την εισαγωγή υπάρχοντος μητρώου για να βρείτε προηγούμενες ενέργειες.",
            "Σε νέα καταχώρηση επιλέξτε σωστή ενότητα, στόχο/οντότητα, υπεύθυνο, κατάσταση και προτεραιότητα.",
            "Το πεδίο «Ένδειξη» περιγράφει την ένδειξη που οδήγησε στην απόφαση· οι Σημειώσεις εξηγούν το σκεπτικό.",
            "Ενημερώνετε την κατάσταση όταν η ενέργεια προχωρά, ώστε το μητρώο να παραμένει αξιόπιστο."
        ],
        "data": "Το μητρώο είναι αρχείο ιχνηλασιμότητας. Δεν αποτελεί από μόνο του εντολή προς Moodle και δεν αντικαθιστά την ελεγχόμενη εγγραφή."
    },
    "report": {
        "title": "Κεντρική Αναφορά",
        "purpose": "Συνοψίζει την κατάσταση των βασικών λειτουργιών, τις ανοικτές ενέργειες και την προέλευση των διαθέσιμων αποτελεσμάτων.",
        "steps": [
            "Ελέγξτε πρώτα ποιο ενεργό αρχείο χρησιμοποιείται.",
            "Διαβάστε τους δείκτες μόνο για λειτουργίες που εμφανίζονται ως συμβατές με το αρχείο.",
            "Χρησιμοποιήστε την αναφορά για συνοπτική ενημέρωση και όχι ως υποκατάστατο της λεπτομερούς ανάλυσης.",
            "Για κάθε κρίσιμη απόφαση επιστρέψτε στην αντίστοιχη λειτουργία και ελέγξτε τα πρωτογενή στοιχεία."
        ],
        "data": "Η αναφορά δεν δημιουργεί νέα αποτελέσματα· συνοψίζει όσα έχουν ήδη υπολογιστεί ή καταγραφεί."
    },

}

def render_info_button(key):
    info = MODULE_HELP.get(key)
    if not info:
        return
    with st.popover("ℹ️ Βοήθεια", use_container_width=False):
        st.markdown(f"### {info['title']}")
        st.write(info['purpose'])
        st.markdown("**Τι κάνω εδώ**")
        for i, step in enumerate(info['steps'][:3], start=1):
            st.markdown(f"**{i}.** {step}")
        if len(info["steps"]) > 3:
            with st.expander("Περισσότερες οδηγίες"):
                for i, step in enumerate(info['steps'][3:], start=4):
                    st.markdown(f"**{i}.** {step}")

def render_module_header(title, subtitle, help_key=None, badge=None):
    st.markdown("<div id='inelso-page-top'></div>", unsafe_allow_html=True)
    left, right = st.columns([8.6,1.4], vertical_alignment="center")
    with left:
        st.markdown(
            f"<div class='inelso-card inelso-module-head'><h2>{title}</h2><p>{subtitle}</p></div>",
            unsafe_allow_html=True
        )
    with right:
        if help_key:
            render_info_button(help_key)
    if badge:
        st.caption(badge)


def render_empty_state(title, message, next_step=None):
    next_html = f"<p><b>Επόμενο βήμα:</b> {next_step}</p>" if next_step else ""
    st.markdown(
        f"<div class='inelso-empty'><h3>{title}</h3><p>{message}</p>{next_html}</div>",
        unsafe_allow_html=True,
    )

def dataset_quality_metrics(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    completeness = max(0.0, 100.0 * (1.0 - missing / cells))
    uniqueness = max(0.0, 100.0 * (1.0 - duplicates / max(len(df), 1)))
    quality = round(0.7 * completeness + 0.3 * uniqueness, 1)
    return {
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "quality": quality,
        "missing": missing,
        "duplicates": duplicates,
    }

def render_source_quality_strip(df, service_key=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    q = dataset_quality_metrics(df)
    compat = service_dataset_compatibility(service_key, df) if service_key else "ΕΝΕΡΓΟ"
    compat_class = "ok" if compat in {"ΠΛΗΡΗΣ","ΔΥΝΑΜΙΚΗ","ΕΝΕΡΓΟ"} else ("warn" if compat in {"ΜΕΡΙΚΗ","CONTEXT"} else "neutral")
    st.markdown(
        "<div class='inelso-status-row'>"
        f"<span class='inelso-pill ok'>● Πηγή ενεργή</span>"
        f"<span class='inelso-pill {compat_class}'>Συμβατότητα: {compat}</span>"
        f"<span class='inelso-pill neutral'>Πληρότητα: {q['completeness']:.1f}%</span>"
        f"<span class='inelso-pill neutral'>Ποιότητα δεδομένων: {q['quality']:.1f}/100</span>"
        "</div>",
        unsafe_allow_html=True,
    )

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
    Αναγνωρίζει Moodle Πλήρη στατιστικά export, όπου:
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
# SCORM / STORYLINE LIVE ANALYTICS
# ==========================================================
def scorm_rest_call(base_url, ws_token, function_name, params=None, timeout=30):
    params = params or {}
    endpoint = base_url.rstrip("/") + "/webservice/rest/server.php"
    payload = {
        "wstoken": ws_token,
        "wsfunction": function_name,
        "moodlewsrestformat": "json",
        **params,
    }
    response = requests.get(endpoint, params=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and (data.get("exception") or data.get("errorcode")):
        raise RuntimeError(data.get("message", data.get("exception", str(data))))
    return data


def _scorm_parse_possible_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return None


def _scorm_flatten_track_items(items, out):
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        element = item.get("element") or item.get("name") or item.get("track")
        value = item.get("value")
        if str(element) == "tracks":
            nested = _scorm_parse_possible_list(value)
            if nested is not None:
                _scorm_flatten_track_items(nested, out)
            else:
                out["tracks"] = value
        elif element is not None:
            out[str(element)] = value


def flatten_scorm_track_response(track_response):
    """Flatten Moodle track responses including the nested Storyline format observed live."""
    out = {}
    if not isinstance(track_response, dict):
        return out

    data = track_response.get("data")
    if isinstance(data, dict):
        if "attempt" in data:
            out["attempt"] = data.get("attempt")
        nested = _scorm_parse_possible_list(data.get("tracks"))
        if nested is not None:
            _scorm_flatten_track_items(nested, out)
        if data.get("element") is not None:
            _scorm_flatten_track_items([data], out)
    elif isinstance(data, list):
        _scorm_flatten_track_items(data, out)

    top_tracks = _scorm_parse_possible_list(track_response.get("tracks"))
    if top_tracks is not None:
        _scorm_flatten_track_items(top_tracks, out)

    if track_response.get("element") is not None:
        _scorm_flatten_track_items([track_response], out)

    return out


def _scorm_interaction_indexes(tracks):
    indexes = set()
    for key in tracks:
        for pattern in (r"cmi\.interactions\.(\d+)\.", r"cmi\.interactions_(\d+)\."):
            m = re.match(pattern, str(key))
            if m:
                indexes.add(int(m.group(1)))
                break
    return sorted(indexes)


def _scorm_interaction_value(tracks, idx, *suffixes):
    for suffix in suffixes:
        for key in (f"cmi.interactions.{idx}.{suffix}", f"cmi.interactions_{idx}.{suffix}"):
            if key in tracks:
                return tracks[key]
    return None


def extract_scorm_interactions(tracks, *, user_id=None, learner=None, attempt=None, sco_id=None, score=None, status=None):
    rows = []
    for idx in _scorm_interaction_indexes(tracks):
        result = _scorm_interaction_value(tracks, idx, "result")
        rows.append({
            "user_id": user_id,
            "learner": learner,
            "attempt": attempt,
            "sco_id": sco_id,
            "interaction_index": idx,
            "question_id": _scorm_interaction_value(tracks, idx, "id"),
            "type": _scorm_interaction_value(tracks, idx, "type"),
            "student_response": _scorm_interaction_value(tracks, idx, "student_response", "learner_response"),
            "correct_response": _scorm_interaction_value(tracks, idx, "correct_responses_0.pattern", "correct_responses.0.pattern"),
            "result": result,
            "is_correct": str(result).strip().lower() == "correct",
            "weighting": _scorm_interaction_value(tracks, idx, "weighting"),
            "latency": _scorm_interaction_value(tracks, idx, "latency"),
            "time": _scorm_interaction_value(tracks, idx, "time"),
            "attempt_score": pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0],
            "scorm_status": status,
        })
    return rows


def scorm_question_statistics(interactions_df, min_discrimination_n=8):
    """
    Build S21-compatible statistics from real SCORM interactions.
    Uses only the latest attempt per learner so repeat attempts do not overweight one learner.
    Discrimination = upper 27% correctness - lower 27% correctness, only when n >= threshold.
    """
    if not isinstance(interactions_df, pd.DataFrame) or interactions_df.empty:
        return pd.DataFrame()

    d = interactions_df.copy()
    d["attempt"] = pd.to_numeric(d["attempt"], errors="coerce")
    d["attempt_score"] = pd.to_numeric(d["attempt_score"], errors="coerce")
    d["is_correct"] = d["is_correct"].astype(bool)

    if "user_id" in d.columns and d["user_id"].notna().any():
        latest = d.groupby("user_id", dropna=True)["attempt"].transform("max")
        d = d[d["attempt"] == latest].copy()

    rows = []
    for qid, g in d.groupby("question_id", dropna=False):
        g = g[g["question_id"].notna()].copy()
        n = int(len(g))
        if n == 0:
            continue
        facility = float(g["is_correct"].mean() * 100.0)
        discrimination = np.nan
        eligible = g.dropna(subset=["attempt_score"]).copy()
        if len(eligible) >= int(min_discrimination_n) and eligible["attempt_score"].nunique() >= 2:
            eligible = eligible.sort_values("attempt_score")
            k = max(1, int(np.ceil(len(eligible) * 0.27)))
            low = float(eligible.head(k)["is_correct"].mean())
            high = float(eligible.tail(k)["is_correct"].mean())
            discrimination = (high - low) * 100.0

        rows.append({
            "ερώτηση": str(qid),
            "facility": round(facility, 2),
            "διάκριση": round(discrimination, 2) if pd.notna(discrimination) else np.nan,
            "αποτελεσματικότητα": np.nan,
            "απαντήσεις": n,
            "σωστές": int(g["is_correct"].sum()),
        })
    return pd.DataFrame(rows)


def build_scorm_unified_dataset(interactions_df, attempts_df, psych_df=None, scorm_name=None, scorm_id=None):
    """
    Δημιουργεί ένα ενιαίο long-format dataset από πραγματικές SCORM προσπάθειες.
    Δεν επινοεί confidence/rubric. Περιέχει μόνο πεδία που προκύπτουν από Moodle/SCORM.
    """
    if not isinstance(interactions_df, pd.DataFrame) or interactions_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = interactions_df.copy()
    x["attempt"] = pd.to_numeric(x.get("attempt"), errors="coerce")
    x["attempt_score"] = pd.to_numeric(x.get("attempt_score"), errors="coerce")
    x["is_correct"] = x.get("is_correct", False).astype(bool)

    # learner/attempt summary from the actual interactions
    group_cols = [c for c in ["user_id","learner","attempt"] if c in x.columns]
    learner_summary = (
        x.groupby(group_cols, dropna=False)
         .agg(
             score=("attempt_score", "max"),
             status=("scorm_status", "last"),
             questions_answered=("question_id", "count"),
             correct_count=("is_correct", "sum"),
         )
         .reset_index()
    )
    learner_summary["incorrect_count"] = learner_summary["questions_answered"] - learner_summary["correct_count"]
    learner_summary["accuracy_pct"] = np.where(
        learner_summary["questions_answered"] > 0,
        learner_summary["correct_count"] / learner_summary["questions_answered"] * 100.0,
        np.nan
    )
    learner_summary["learner_id"] = learner_summary.get("user_id")
    learner_summary["completion"] = learner_summary["status"]
    learner_summary["scorm_name"] = scorm_name
    learner_summary["scorm_id"] = scorm_id

    # Add timing/status collected at attempt level when available.
    if isinstance(attempts_df, pd.DataFrame) and not attempts_df.empty:
        extra_cols = [c for c in ["user_id","attempt","total_time","session_time"] if c in attempts_df.columns]
        if len(extra_cols) > 2:
            extras = attempts_df[extra_cols].drop_duplicates(["user_id","attempt"])
            learner_summary = learner_summary.merge(extras, on=["user_id","attempt"], how="left")

    # Merge learner summary back onto each question interaction.
    merge_cols = ["user_id","attempt"]
    summary_for_merge = learner_summary.drop(columns=[c for c in ["learner"] if c in learner_summary.columns], errors="ignore")
    unified = x.merge(summary_for_merge, on=merge_cols, how="left", suffixes=("", "_summary"))

    # Add item-level psychometrics so one CSV can support item and learner analyses.
    if isinstance(psych_df, pd.DataFrame) and not psych_df.empty:
        pmap = psych_df.rename(columns={
            "ερώτηση": "question_id",
            "facility": "facility_pct",
            "διάκριση": "discrimination_pct",
            "αποτελεσματικότητα": "discriminative_efficiency_pct",
            "απαντήσεις": "question_responses_n",
            "σωστές": "question_correct_n",
        }).copy()
        keep = [c for c in [
            "question_id","facility_pct","discrimination_pct",
            "discriminative_efficiency_pct","question_responses_n","question_correct_n"
        ] if c in pmap.columns]
        unified = unified.merge(pmap[keep], on="question_id", how="left")

    unified["review_flag"] = False
    if "facility_pct" in unified.columns:
        unified["review_flag"] = unified["review_flag"] | (
            (pd.to_numeric(unified["facility_pct"], errors="coerce") <= 10) |
            (pd.to_numeric(unified["facility_pct"], errors="coerce") >= 90)
        )
    if "discrimination_pct" in unified.columns:
        disc = pd.to_numeric(unified["discrimination_pct"], errors="coerce")
        unified["review_flag"] = unified["review_flag"] | (disc.notna() & (disc < 20))

    unified["source_type"] = "MOODLE_SCORM"
    unified["scorm_name"] = scorm_name
    unified["scorm_id"] = scorm_id

    preferred = [
        "learner_id","user_id","learner","attempt","score","status","completion",
        "total_time","session_time","questions_answered","correct_count","incorrect_count",
        "accuracy_pct","sco_id","interaction_index","question_id","type",
        "student_response","correct_response","result","is_correct","weighting","latency","time",
        "facility_pct","discrimination_pct","discriminative_efficiency_pct",
        "question_responses_n","question_correct_n","review_flag","scorm_id","scorm_name","source_type"
    ]
    ordered = [c for c in preferred if c in unified.columns] + [c for c in unified.columns if c not in preferred]
    unified = unified[ordered]
    return unified, learner_summary


# ==========================================================
# ΚΟΙΝΟ ACTIVE DATASET CONTEXT ΓΙΑ ΟΛΗ ΤΗΝ ΕΦΑΡΜΟΓΗ
# ==========================================================
def get_active_dataset():
    df = st.session_state.get(
        "inelso_uploaded_clean",
        st.session_state.get("inelso_uploaded_original")
    )
    if isinstance(df, pd.DataFrame):
        return df.copy()
    scorm_df = st.session_state.get("inelso_scorm_active_df")
    return scorm_df.copy() if isinstance(scorm_df, pd.DataFrame) else None


def active_dataset_name():
    if st.session_state.get("inelso_uploaded_name"):
        return st.session_state.get("inelso_uploaded_name")
    if st.session_state.get("inelso_scorm_active_name"):
        return st.session_state.get("inelso_scorm_active_name")
    return "Δεν έχει φορτωθεί πηγή δεδομένων"


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


def auto_map_unified_services(df, source_name="Ανεβασμένο αρχείο"):
    """
    Αυτόματη αντιστοίχιση ενός ενιαίου CSV/Excel στα βασικά services.
    Δεν επινοεί δεδομένα: χρησιμοποιεί μόνο στήλες που υπάρχουν πραγματικά.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"s21": False, "s26": False}

    mapped = {"s21": False, "s26": False}

    # -----------------------------
    # Ποιότητα Αξιολόγησης (S21)
    # -----------------------------
    q_col = detect_column(df, ["ερώτηση", "question_id", "question", "question name"])
    f_col = detect_column(df, ["facility", "facility_pct", "facility index"])
    d_col = detect_column(df, ["διάκριση", "discrimination_pct", "discrimination", "discrimination index"])
    e_col = detect_column(df, ["αποτελεσματικότητα", "discriminative_efficiency_pct", "efficiency"])

    if q_col is not None and f_col is not None and d_col is not None:
        cols = [q_col, f_col, d_col] + ([e_col] if e_col is not None else [])
        s21 = df[cols].copy()
        rename_map = {q_col: "ερώτηση", f_col: "facility", d_col: "διάκριση"}
        if e_col is not None:
            rename_map[e_col] = "αποτελεσματικότητα"
        s21 = s21.rename(columns=rename_map)

        if "αποτελεσματικότητα" not in s21.columns:
            s21["αποτελεσματικότητα"] = np.nan

        s21["facility"] = pd.to_numeric(s21["facility"], errors="coerce")
        s21["διάκριση"] = pd.to_numeric(s21["διάκριση"], errors="coerce")
        s21["αποτελεσματικότητα"] = pd.to_numeric(s21["αποτελεσματικότητα"], errors="coerce")
        s21 = (
            s21.dropna(subset=["ερώτηση"])
               .drop_duplicates(subset=["ερώτηση"], keep="first")
               .reset_index(drop=True)
        )

        if not s21.empty:
            st.session_state["custom_s21"] = s21
            st.session_state["custom_s21_source"] = f"Αυτόματη αντιστοίχιση — {source_name}"
            mapped["s21"] = True
        else:
            st.session_state.pop("custom_s21", None)
            st.session_state.pop("custom_s21_source", None)
    else:
        st.session_state.pop("custom_s21", None)
        st.session_state.pop("custom_s21_source", None)

    # --------------------------------
    # Προσαρμοστική Εξάσκηση (S26)
    # --------------------------------
    learner_col = detect_column(df, ["μαθητής", "learner", "learner_id", "user_id"])
    score_col = detect_column(df, ["score", "final_score", "grade", "final score"])
    conf_col = detect_column(df, ["βεβαιότητα", "confidence"])
    attempt_col = detect_column(df, ["προσπάθεια", "attempt"])

    rc_cols = {}
    for rc in ["RC-01","RC-02","RC-03","RC-04","RC-05"]:
        found = detect_column(df, [rc])
        if found is not None:
            rc_cols[rc] = found

    if learner_col is not None and score_col is not None and conf_col is not None:
        use_cols = [learner_col, score_col, conf_col]
        if attempt_col is not None:
            use_cols.append(attempt_col)
        use_cols += list(rc_cols.values())

        s26 = df[use_cols].copy()
        rename_map = {
            learner_col: "μαθητής",
            score_col: "score",
            conf_col: "βεβαιότητα",
        }
        if attempt_col is not None:
            rename_map[attempt_col] = "προσπάθεια"
        for rc, original in rc_cols.items():
            rename_map[original] = rc

        s26 = s26.rename(columns=rename_map)
        s26["score"] = pd.to_numeric(s26["score"], errors="coerce")
        s26["βεβαιότητα"] = pd.to_numeric(s26["βεβαιότητα"], errors="coerce")

        if "προσπάθεια" in s26.columns:
            s26["προσπάθεια"] = pd.to_numeric(s26["προσπάθεια"], errors="coerce")
        else:
            s26["προσπάθεια"] = 1

        for rc in ["RC-01","RC-02","RC-03","RC-04","RC-05"]:
            if rc not in s26.columns:
                s26[rc] = 2
            s26[rc] = pd.to_numeric(s26[rc], errors="coerce").fillna(2)

        # Το unified CSV είναι long-format (μία γραμμή ανά learner × question).
        # Για S26 κρατάμε μία γραμμή ανά learner/attempt.
        s26 = (
            s26.dropna(subset=["μαθητής", "score", "βεβαιότητα"])
               .sort_values(["μαθητής", "προσπάθεια"])
               .drop_duplicates(subset=["μαθητής", "προσπάθεια"], keep="last")
               .reset_index(drop=True)
        )

        if not s26.empty:
            st.session_state["custom_s26"] = s26
            st.session_state["custom_s26_source"] = f"Αυτόματη αντιστοίχιση — {source_name}"
            mapped["s26"] = True
        else:
            st.session_state.pop("custom_s26", None)
            st.session_state.pop("custom_s26_source", None)
    else:
        st.session_state.pop("custom_s26", None)
        st.session_state.pop("custom_s26_source", None)

    return mapped


def active_dataset_kind(df):
    if not isinstance(df, pd.DataFrame):
        return "ΚΑΝΕΝΑ"
    cols = [str(c).strip().lower() for c in df.columns]
    if {"question_id", "result", "attempt"}.issubset(set(cols)):
        return "SCORM_INTERACTIONS"
    if (
        st.session_state.get("custom_s21_source") == "Moodle Πλήρη στατιστικά"
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
        return "Δεν υπάρχει ενεργό σύνολο δεδομένων."
    p = dataset_profile(df)
    if max_cols is None:
        cols = [str(c) for c in df.columns]
        suffix = ""
    else:
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
        return "ΠΛΑΙΣΙΟ"
    if service_key == "S31":
        if st.session_state.get("custom_s21") is not None or st.session_state.get("custom_s26") is not None:
            return "ΔΥΝΑΜΙΚΗ"
        return "ΠΛΑΙΣΙΟ"
    if service_key in {"ANALYTICS","DATA","REPORT","DASHBOARD","PRESENTATION"}:
        return "ΠΛΗΡΗΣ"
    return "ΠΛΑΙΣΙΟ"


def show_active_dataset_context(service_key=None, compact=True):
    df = get_active_dataset()
    if df is None:
        st.caption("Δεν υπάρχει ενεργή πηγή δεδομένων.")
        return

    p = dataset_profile(df)
    st.caption(
        f"Πηγή δεδομένων: {active_dataset_name()}  •  "
        f"{p['rows']} εγγραφές × {p['cols']} πεδία"
    )
    render_source_quality_strip(df, service_key)
    if not compact:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Εγγραφές", p["rows"])
        c2.metric("Πεδία", p["cols"])
        c3.metric("Κενές τιμές", p["missing"])
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

if isinstance(ACTIVE_DF, pd.DataFrame) and (
    "custom_s21" not in st.session_state or "custom_s26" not in st.session_state
):
    auto_map_unified_services(ACTIVE_DF, ACTIVE_NAME)

# ==========================================================
# ΠΡΑΓΜΑΤΙΚΑ / ΕΛΕΓΧΟΜΕΝΑ SIGNALS SERVICE 21
# ==========================================================
service21 = pd.DataFrame([
    {"ερώτηση":"Marketing Q01","facility":2.00,"διάκριση":34.92,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"Marketing Q02","facility":2.00,"διάκριση":34.92,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"Marketing Q04","facility":0.67,"διάκριση":39.68,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"Marketing Q07","facility":0.00,"διάκριση":np.nan,"αποτελεσματικότητα":np.nan},
    {"ερώτηση":"Marketing Q10","facility":2.00,"διάκριση":34.92,"αποτελεσματικότητα":100.00},
    {"ερώτηση":"Marketing Q05","facility":20.00,"διάκριση":44.11,"αποτελεσματικότητα":61.54},
    {"ερώτηση":"Marketing Q09","facility":72.00,"διάκριση":12.88,"αποτελεσματικότητα":21.49},
    {"ερώτηση":"Marketing Q03","facility":44.00,"διάκριση":73.22,"αποτελεσματικότητα":92.35},
    {"ερώτηση":"Marketing Q06","facility":32.00,"διάκριση":44.38,"αποτελεσματικότητα":52.03},
    {"ερώτηση":"Marketing Q08","facility":52.00,"διάκριση":60.89,"αποτελεσματικότητα":80.71},
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
        return "Έλεγχος διατύπωσης, παραπλανητικών επιλογών και ευθυγράμμισης με μαθησιακό στόχο"
    if d < 40:
        return "Διατήρηση με παρακολούθηση"
    return "Διατήρηση"

service21["κατάσταση"] = service21["διάκριση"].apply(κατηγορία_διάκρισης)
service21["ενέργεια"] = service21.apply(ενέργεια_ερώτησης, axis=1)

# Αν έχει δημιουργηθεί Ποιότητα Αξιολόγησης dataset από upload, αυτό γίνεται η ενεργή πηγή.
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

# Αν υπάρχει mapped Προσαρμοστική Εξάσκηση dataset από το ενεργή πηγή, γίνεται η ενεργή cohort πηγή.
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
        # Teacher cohort = τελευταία διαθέσιμη προσπάθεια ανά learner.
        if "προσπάθεια" in mapped26_active.columns:
            _latest_idx = (
                mapped26_active.sort_values(["μαθητής","προσπάθεια"])
                .groupby("μαθητής", dropna=False)
                .tail(1).index
            )
            cohort26 = mapped26_active.loc[_latest_idx, [
                "μαθητής","score","βεβαιότητα","κατάκτηση","κατάσταση"
            ]].copy().reset_index(drop=True)
        else:
            cohort26 = mapped26_active[
                ["μαθητής","score","βεβαιότητα","κατάκτηση","κατάσταση"]
            ].copy()

# ==========================================================
# SERVICE 31: ΠΛΕΟΝ ΠΑΡΑΓΕΤΑΙ ΑΠΟ SIGNALS 21 + 26
# ==========================================================
# Μόνο το Q08 έχει τεκμηριωμένη σημασιολογική ετικέτα από το υπάρχον workflow (CTR).
# Τα Q09/Q07 διατηρούνται ως item-level ενδείξεις χωρίς επινοημένη θεματική ονομασία.

# 1) Signals από Ποιότητα Αξιολόγησης
# Παράγονται δυναμικά από το ενεργό Ποιότητα Αξιολόγησης dataset.
s21_ένδειξη_rows = []

estimable_s21 = service21.dropna(subset=["διάκριση"]).copy()
if not estimable_s21.empty:
    strongest = estimable_s21.sort_values("διάκριση", ascending=False).iloc[0]
    weakest = estimable_s21.sort_values("διάκριση", ascending=True).iloc[0]

    s21_ένδειξη_rows.append({
        "οντότητα": str(strongest["ερώτηση"]),
        "επίπεδο_ένδειξης":"Ερώτηση",
        "πηγή":"Ποιότητα Quiz",
        "τύπος":"Ισχυρή ένδειξη ανά ερώτηση",
        "ισχύς_σήματος":round(float(strongest["διάκριση"]),2),
        "κατεύθυνση":"ΘΕΤΙΚΗ",
        "ενέργεια":"Διατήρηση / χρήση ως ισχυρό σημείο αναφοράς",
        "προέλευση":f"Ενεργό αρχείο: {ACTIVE_NAME}"
    })

    weak_strength = min(100.0, max(0.0, 100.0 - float(weakest["διάκριση"])))
    s21_ένδειξη_rows.append({
        "οντότητα":str(weakest["ερώτηση"]),
        "επίπεδο_ένδειξης":"Ερώτηση",
        "πηγή":"Ποιότητα Quiz",
        "τύπος":"Αδύναμη ένδειξη ανά ερώτηση",
        "ισχύς_σήματος":round(weak_strength,2),
        "κατεύθυνση":"ΑΡΝΗΤΙΚΗ" if float(weakest["διάκριση"]) < 20 else "ΜΕΣΑΙΑ",
        "ενέργεια":str(weakest["ενέργεια"]),
        "προέλευση":f"Ενεργό αρχείο: {ACTIVE_NAME}"
    })

nonestimable_s21 = service21[service21["διάκριση"].isna()]
if not nonestimable_s21.empty:
    rr = nonestimable_s21.iloc[0]
    s21_ένδειξη_rows.append({
        "οντότητα":str(rr["ερώτηση"]),
        "επίπεδο_ένδειξης":"Ερώτηση",
        "πηγή":"Ποιότητα Quiz",
        "τύπος":"Μη εκτιμήσιμη ένδειξη ανά ερώτηση",
        "ισχύς_σήματος":100.0,
        "κατεύθυνση":"ΑΒΕΒΑΙΗ",
        "ενέργεια":str(rr["ενέργεια"]),
        "προέλευση":f"Ενεργό αρχείο: {ACTIVE_NAME}"
    })

s21_ενδείξεις = pd.DataFrame(s21_ένδειξη_rows)
if "custom_s21" not in st.session_state:
    s21_ενδείξεις = pd.DataFrame(columns=[
        "οντότητα","επίπεδο_ένδειξης","πηγή","τύπος","ισχύς_σήματος","κατεύθυνση","ενέργεια","προέλευση"
    ])

# 2) Signals από Προσαρμοστική Εξάσκηση
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
            "οντότητα":str(rr.get("μαθητής","Εκπαιδευόμενος CSV")),
            "επίπεδο_ένδειξης":"Εκπαιδευόμενος",
            "πηγή":"Προσαρμοστική Εξάσκηση",
            "τύπος":"Προσαρμοστική ένδειξη ανά εκπαιδευόμενο",
            "ισχύς_σήματος":strength,
            "κατεύθυνση":direction,
            "ενέργεια":action,
            "προέλευση":f"Ενεργό αρχείο: {ACTIVE_NAME}"
        })
    s26_ενδείξεις = pd.DataFrame(s26_rows)
else:
    # Χωρίς mapped Προσαρμοστική Εξάσκηση dataset δεν δημιουργούνται demo/τεχνητές ενδείξεις.
    s26_ενδείξεις = pd.DataFrame(columns=[
        "οντότητα","επίπεδο_ένδειξης","πηγή","τύπος","ισχύς_σήματος","κατεύθυνση","ενέργεια","προέλευση"
    ])

service31_integrated = pd.concat([s21_ενδείξεις, s26_ενδείξεις], ignore_index=True)

# Ενιαία προτεραιότητα Ενοποιημένες Μαθησιακές Ενδείξεις από πραγματικά ενδείξεις.
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
st.sidebar.markdown("## 🎓 INELSO Academia")
st.sidebar.caption("Πλατφόρμα Έξυπνης Μάθησης")
st.sidebar.markdown("<div class='inelso-section-label'>Κύριες λειτουργίες</div>", unsafe_allow_html=True)

def navigate_main(target_label):
    st.session_state["inelso_main_navigation"] = target_label

NAV_DISPLAY_TO_INTERNAL = {
    "Αρχική": "Κεντρική Εικόνα Ελέγχου",
    "Έγκαιρη Υποστήριξη": "Μαθησιακή Ανάλυση & Έγκαιρη Υποστήριξη",
    "Ποιότητα Quiz": "Ποιότητα Αξιολόγησης",
    "Ανάλυση SCORM": "Ανάλυση SCORM Αξιολόγησης",
    "Δημιουργία Αξιολόγησης": "Δημιουργία Μαθήματος & Αξιολόγησης",
    "Προσαρμοστική Εξάσκηση": "Προσαρμοστική Εξάσκηση",
    "Υποστήριξη Διδάσκοντα": "Υποστήριξη Διδάσκοντα",
    "Μαθησιακές Ενδείξεις": "Ενοποιημένες Μαθησιακές Ενδείξεις",
    "Εισαγωγή Δεδομένων": "Εισαγωγή CSV / Excel",
    "Εργαλεία & Διαχείριση": "__ADVANCED__",
}
_nav_choice = st.sidebar.radio(
    "Πλοήγηση",
    list(NAV_DISPLAY_TO_INTERNAL.keys()),
    label_visibility="collapsed",
    key="inelso_main_navigation",
)
section = NAV_DISPLAY_TO_INTERNAL[_nav_choice]

if section == "__ADVANCED__":
    st.sidebar.markdown("<div class='inelso-section-label'>Εργαλεία & διαχείριση</div>", unsafe_allow_html=True)
    section = st.sidebar.selectbox(
        "Επιλέξτε εργαλείο",
        [
            "Ενιαία Ροή Ενεργειών",
            "Κύκλος Ζωής Ενεργειών",
            "Μεθοδολογία & Προέλευση",
            "Εργαστήριο Ανάλυσης Δεδομένων",
            "Προσομοίωση Σεναρίων",
            "Ζωντανή Σύνδεση Moodle",
            "Ελεγχόμενη Εγγραφή Moodle",
            "Μητρώο Ενεργειών",
            "Κεντρική Αναφορά",
        ],
        key="inelso_advanced_navigation",
    )

st.sidebar.markdown("---")
if ACTIVE_DF is not None:
    _p = dataset_profile(ACTIVE_DF)
    _q = dataset_quality_metrics(ACTIVE_DF)
    st.sidebar.success(f"Πηγή ενεργή: {ACTIVE_NAME}")
    st.sidebar.caption(f"{_p['rows']} εγγραφές × {_p['cols']} πεδία")
    if _q:
        st.sidebar.caption(f"Ποιότητα δεδομένων: {_q['quality']:.1f}/100")
else:
    st.sidebar.info("Δεν υπάρχει ενεργή πηγή δεδομένων.")

st.sidebar.caption("🔒 Καμία επίσημη βαθμολογία ή κρίσιμη ενέργεια δεν αλλάζει αυτόματα.")
# Βοήθεια για κάθε οθόνη. Οι έξι βασικές μαθησιακές λειτουργίες έχουν το
# κουμπί ενσωματωμένο στην κεφαλίδα τους· οι υπόλοιπες το εμφανίζουν εδώ.
SECTION_HELP = {
    "Ανάλυση SCORM Αξιολόγησης": "scorm_analysis",
    "Ενιαία Ροή Ενεργειών": "flow",
    "Κύκλος Ζωής Ενεργειών": "lifecycle",
    "Μεθοδολογία & Προέλευση": "methodology",
    "Εισαγωγή CSV / Excel": "csv_import",
    "Εργαστήριο Ανάλυσης Δεδομένων": "analytics_lab",
    "Προσομοίωση Σεναρίων": "whatif",
    "Ζωντανή Σύνδεση Moodle": "moodle_live",
    "Ελεγχόμενη Εγγραφή Moodle": "moodle_write",
    "Μητρώο Ενεργειών": "registry",
    "Κεντρική Αναφορά": "report",
}
# Οι βοηθητικές ενότητες εμφανίζουν τις οδηγίες μέσα στην ίδια την οθόνη,
# ώστε η κεφαλίδα της σελίδας να παραμένει πάντα το πρώτο καθαρό στοιχείο.

# ==========================================================
# ΚΕΝΤΡΙΚΟΣ ΠΙΝΑΚΑΣ
# ==========================================================
if section == "Κεντρική Εικόνα Ελέγχου":

    def _home_nav(target_label):
        # Button callback: ενημερώνει το βασικό radio στην επόμενη εκτέλεση.
        st.session_state["inelso_main_navigation"] = target_label

    st.markdown("""
    <div class="inelso-home-hero">
      <h1>Πλατφόρμα Έξυπνης Μάθησης</h1>
    </div>
    """, unsafe_allow_html=True)

    hleft, hright = st.columns([8.6, 1.4], vertical_alignment="center")
    with hright:
        render_info_button("dashboard")

    show_active_dataset_context("DASHBOARD", compact=False)

    if ACTIVE_DF is None:
        a,b = st.columns(2)
        with a:
            st.markdown(
                "<div class='inelso-feature-card'><div class='kicker'>Ζωντανή πηγή</div>"
                "<h3>Moodle / SCORM</h3><p>Ανάκτηση πραγματικών προσπαθειών, βαθμών, χρόνων και απαντήσεων ανά ερώτηση.</p></div>",
                unsafe_allow_html=True
            )
            st.button("Άνοιγμα Ανάλυσης SCORM", use_container_width=True, on_click=_home_nav, args=("Ανάλυση SCORM",), key="home_open_scorm_empty")
        with b:
            st.markdown(
                "<div class='inelso-feature-card'><div class='kicker'>Αρχείο</div>"
                "<h3>CSV / Excel</h3><p>Εισαγωγή εξωτερικών ή συμπληρωματικών δεδομένων με αυτόματο έλεγχο ποιότητας και συμβατότητας.</p></div>",
                unsafe_allow_html=True
            )
            st.button("Εισαγωγή δεδομένων", use_container_width=True, on_click=_home_nav, args=("Εισαγωγή Δεδομένων",), key="home_open_data_empty")

    else:
        q = dataset_quality_metrics(ACTIVE_DF)
        valid_healths = []
        if "custom_s21" in st.session_state:
            valid_healths.append(float(health21))
        if "custom_s26" in st.session_state:
            valid_healths.append(float(health26))
        if ("custom_s21" in st.session_state) or ("custom_s26" in st.session_state):
            valid_healths.append(float(health31))

        dashboard_overall = round(sum(valid_healths)/len(valid_healths), 1) if valid_healths else None
        dashboard_open = (
            (open21 if "custom_s21" in st.session_state else 0)
            + (open26 if "custom_s26" in st.session_state else 0)
            + (open31 if (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else 0)
        )

        st.markdown("<div class='inelso-dashboard-tabs'>", unsafe_allow_html=True)
        home_tabs = st.tabs(["Επισκόπηση", "Αναλύσεις", "Ενέργειες", "Δεδομένα"])
        st.markdown("</div>", unsafe_allow_html=True)

        with home_tabs[0]:
            st.markdown("### Επιχειρησιακή εικόνα")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Συνολικός δείκτης", f"{dashboard_overall:.1f}/100" if dashboard_overall is not None else "—")
            c2.metric("Ποιότητα Quiz", f"{health21:.1f}/100" if "custom_s21" in st.session_state else "—")
            c3.metric("Προσαρμοστική Εξάσκηση", f"{health26:.1f}/100" if "custom_s26" in st.session_state else "—")
            c4.metric("Μαθησιακές Ενδείξεις", f"{health31:.1f}/100" if (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—")
            c5.metric("Ανοικτές ενέργειες", dashboard_open if valid_healths else "—")

            if dashboard_open:
                st.warning(f"Υπάρχουν {dashboard_open} σημεία που χρειάζονται προσοχή. Δείτε το tab «Ενέργειες».")
            else:
                st.success("Δεν υπάρχουν αυτή τη στιγμή καταγεγραμμένες ενέργειες υψηλής προτεραιότητας.")

        with home_tabs[1]:
            st.markdown("### Κύριες αναλύσεις")
            st.caption("Οι κάρτες δείχνουν μόνο αποτελέσματα που υποστηρίζονται από την ενεργή πηγή.")

            a1,a2,a3 = st.columns(3)
            with a1:
                s21_value = f"{health21:.1f}/100" if "custom_s21" in st.session_state else "—"
                s21_note = f"{open21} ανοικτές ενέργειες" if "custom_s21" in st.session_state else "Χρειάζονται δεδομένα ανά ερώτηση"
                st.markdown(
                    f"<div class='inelso-feature-card'><h3>Ποιότητα Κουίζ</h3>"
                    f"<div class='inelso-feature-value'>{s21_value}</div><p>{s21_note}</p></div>",
                    unsafe_allow_html=True
                )
                st.button("Άνοιγμα Ποιότητας Κουίζ", use_container_width=True, on_click=_home_nav, args=("Ποιότητα Quiz",), key="home_open_s21")
            with a2:
                s26_value = f"{health26:.1f}/100" if "custom_s26" in st.session_state else "—"
                s26_note = f"{open26} περιπτώσεις για ενέργεια" if "custom_s26" in st.session_state else "Χρειάζονται δεδομένα ανά εκπαιδευόμενο"
                st.markdown(
                    f"<div class='inelso-feature-card'><h3>Προσαρμοστική Εξάσκηση</h3>"
                    f"<div class='inelso-feature-value'>{s26_value}</div><p>{s26_note}</p></div>",
                    unsafe_allow_html=True
                )
                st.button("Άνοιγμα Προσαρμοστικής Εξάσκησης", use_container_width=True, on_click=_home_nav, args=("Προσαρμοστική Εξάσκηση",), key="home_open_s26")
            with a3:
                s31_value = f"{health31:.1f}/100" if (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—"
                s31_note = f"{len(service31_integrated)} πραγματικές ενδείξεις" if (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "Αναμονή πραγματικών ενδείξεων"
                st.markdown(
                    f"<div class='inelso-feature-card'><h3>Μαθησιακές Ενδείξεις</h3>"
                    f"<div class='inelso-feature-value'>{s31_value}</div><p>{s31_note}</p></div>",
                    unsafe_allow_html=True
                )
                st.button("Άνοιγμα Μαθησιακών Ενδείξεων", use_container_width=True, on_click=_home_nav, args=("Μαθησιακές Ενδείξεις",), key="home_open_s31")

            b1,b2 = st.columns(2)
            with b1:
                st.markdown(
                    "<div class='inelso-feature-card'><h3>Έγκαιρη Υποστήριξη</h3>"
                    "<p>Κίνδυνος ανά εκπαιδευόμενο και προτεραιοποίηση υποστήριξης.</p></div>",
                    unsafe_allow_html=True
                )
                st.button("Άνοιγμα Έγκαιρης Υποστήριξης", use_container_width=True, on_click=_home_nav, args=("Έγκαιρη Υποστήριξη",), key="home_open_s16")
            with b2:
                st.markdown(
                    "<div class='inelso-feature-card'><h3>Δημιουργία Αξιολόγησης</h3>"
                    "<p>Τράπεζα ερωτήσεων, κουίζ, κλίμακα αξιολόγησης και ασκήσεις με έλεγχο διδάσκοντα.</p></div>",
                    unsafe_allow_html=True
                )
                st.button("Άνοιγμα Δημιουργίας Αξιολόγησης", use_container_width=True, on_click=_home_nav, args=("Δημιουργία Αξιολόγησης",), key="home_open_s25")

            with st.expander("Τεχνική κατάσταση όλων των λειτουργιών", expanded=False):
                comparison = pd.DataFrame([
                    {"Ενότητα":"Έγκαιρη Υποστήριξη","Συμβατότητα":service_dataset_compatibility("S16", ACTIVE_DF),"Αποτέλεσμα":"Έτοιμο για ανάλυση","Πηγή":ACTIVE_NAME},
                    {"Ενότητα":"Ποιότητα Quiz","Συμβατότητα":service_dataset_compatibility("S21", ACTIVE_DF),"Αποτέλεσμα":f"{health21:.1f}/100" if "custom_s21" in st.session_state else "Αναμονή δεδομένων ανά ερώτηση","Πηγή":ACTIVE_NAME},
                    {"Ενότητα":"Δημιουργία Αξιολόγησης","Συμβατότητα":service_dataset_compatibility("S25", ACTIVE_DF),"Αποτέλεσμα":"Διαθέσιμο ως πλαίσιο","Πηγή":ACTIVE_NAME},
                    {"Ενότητα":"Προσαρμοστική Εξάσκηση","Συμβατότητα":service_dataset_compatibility("S26", ACTIVE_DF),"Αποτέλεσμα":f"{health26:.1f}/100" if "custom_s26" in st.session_state else "Αναμονή αντιστοίχισης ανά εκπαιδευόμενο","Πηγή":ACTIVE_NAME},
                    {"Ενότητα":"Υποστήριξη Διδάσκοντα","Συμβατότητα":service_dataset_compatibility("S28", ACTIVE_DF),"Αποτέλεσμα":"Διαθέσιμο ως πλαίσιο","Πηγή":ACTIVE_NAME},
                    {"Ενότητα":"Μαθησιακές Ενδείξεις","Συμβατότητα":service_dataset_compatibility("S31", ACTIVE_DF),"Αποτέλεσμα":f"{len(service31_integrated)} ενδείξεις" if (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "Αναμονή πραγματικών ενδείξεων","Πηγή":ACTIVE_NAME},
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)

        with home_tabs[2]:
            st.markdown("### Τι χρειάζεται προσοχή")
            e1,e2,e3 = st.columns(3)
            with e1:
                st.metric("Έλεγχος κουίζ", open21 if "custom_s21" in st.session_state else "—")
                st.caption("Ερωτήσεις που χρειάζονται έλεγχο ή αναθεώρηση.")
            with e2:
                st.metric("Ενέργειες προσαρμοστικής εξάσκησης", open26 if "custom_s26" in st.session_state else "—")
                st.caption("Εκπαιδευόμενοι/προσπάθειες που χρειάζονται νέα προσπάθεια ή ανθρώπινο έλεγχο.")
            with e3:
                st.metric("Ενδείξεις υψηλής προτεραιότητας", open31 if (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—")
                st.caption("Ενοποιημένες ενδείξεις υψηλής προτεραιότητας.")

            if dashboard_open:
                st.markdown(
                    f"<div class='inelso-danger-lite'><b>{dashboard_open} ανοικτές ενέργειες συνολικά.</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.success("Δεν υπάρχουν ανοικτές ενέργειες από τις ενεργές αναλύσεις.")

            c1,c2 = st.columns(2)
            with c1:
                st.button("Άνοιγμα Ποιότητας Κουίζ", use_container_width=True, on_click=_home_nav, args=("Ποιότητα Quiz",), key="home_actions_s21")
            with c2:
                st.button("Άνοιγμα Προσαρμοστικής Εξάσκησης", use_container_width=True, on_click=_home_nav, args=("Προσαρμοστική Εξάσκηση",), key="home_actions_s26")

        with home_tabs[3]:
            st.markdown("### Ενεργή πηγή δεδομένων")
            st.markdown(
                f"<div class='inelso-feature-card'><h3>{ACTIVE_NAME}</h3>"
                f"<p>{len(ACTIVE_DF)} εγγραφές · {len(ACTIVE_DF.columns)} πεδία · "
                f"πληρότητα {q['completeness']:.1f}% · ποιότητα {q['quality']:.1f}/100.</p></div>",
                unsafe_allow_html=True
            )
            x1,x2 = st.columns(2)
            with x1:
                st.button("Ανάλυση SCORM", use_container_width=True, on_click=_home_nav, args=("Ανάλυση SCORM",), key="home_data_scorm")
            with x2:
                st.button("Εισαγωγή νέου αρχείου", use_container_width=True, on_click=_home_nav, args=("Εισαγωγή Δεδομένων",), key="home_data_import")

            with st.expander("Περισσότερες πληροφορίες για την πηγή", expanded=False):
                st.write(f"**Όνομα πηγής:** {ACTIVE_NAME}")
                st.write(f"**Τύπος:** {ACTIVE_KIND}")
                st.write(f"**Εγγραφές:** {len(ACTIVE_DF)}")
                st.write(f"**Πεδία:** {len(ACTIVE_DF.columns)}")
                st.write(f"**Κενές τιμές:** {q['missing'] if q else '—'}")
                st.write(f"**Διπλότυπα:** {q['duplicates'] if q else '—'}")

    
# ==========================================================
# SERVICE 21
# ==========================================================
elif section == "Ποιότητα Αξιολόγησης":
    if ACTIVE_DF is None:
        render_module_header("Ποιότητα Αξιολόγησης", "", "s21")
        show_active_dataset_context("S21")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Ερωτήσεις","—")
        c2.metric("Μέσος δείκτης ευκολίας","—")
        c3.metric("Μέσος δείκτης διάκρισης","—")
        c4.metric("Δείκτης υγείας","—")
        c5.metric("Ανοικτές ενέργειες","—")
        st.stop()
    if "custom_s21" not in st.session_state:
        render_module_header("Ποιότητα Αξιολόγησης", "", "s21")
        show_active_dataset_context("S21")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Ερωτήσεις","—")
        c2.metric("Μέσος δείκτης ευκολίας","—")
        c3.metric("Μέσος δείκτης διάκρισης","—")
        c4.metric("Δείκτης υγείας","—")
        c5.metric("Ανοικτές ενέργειες","—")
        st.stop()

    render_module_header("Ποιότητα Αξιολόγησης", "", "s21")
    show_active_dataset_context("S21")
    

    with st.expander("📘 Πώς πρέπει να είναι δομημένο το Quiz;", expanded=False):
        st.markdown("""
Για αξιόπιστη ανάλυση, το quiz πρέπει να έχει **ξεχωριστές, σαφώς αναγνωρίσιμες ερωτήσεις** και να έχει χρησιμοποιηθεί από πραγματικούς φοιτητές.

- Κάθε ερώτηση χρειάζεται μοναδικό όνομα ή ID.
- Η βαθμολόγηση πρέπει να είναι ορισμένη πριν τη συλλογή δεδομένων.
- Χρειάζονται αρκετές ολοκληρωμένες προσπάθειες ώστε οι δείκτες να έχουν νόημα.
- Από το Moodle προτιμάται η εξαγωγή **Πλήρη στατιστικά ερωτήσεων**.
- Ο **δείκτης ευκολίας** δείχνει πόσο συχνά απαντήθηκε σωστά μια ερώτηση.
- Ο **δείκτης διάκρισης** δείχνει κατά πόσο η ερώτηση ξεχωρίζει καλύτερες από ασθενέστερες επιδόσεις.

Η πλατφόρμα δεν διαγράφει ούτε αλλάζει αυτόματα ερώτηση. Προτείνει έλεγχο από τον διδάσκοντα.
        """)

    if "custom_s21" in st.session_state:
        st.success(
            f"Ενεργή πηγή: {st.session_state.get('custom_s21_source', ACTIVE_NAME)} — "
            f"{len(service21)} ερωτήσεις με αναλυτικά στατιστικά. Όλοι οι δείκτες και τα γραφήματα παρακάτω "
            "υπολογίζονται από αυτό το αρχείο."
        )

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ερωτήσεις",len(service21))
    c2.metric("Μέσος δείκτης ευκολίας",f"{service21['facility'].mean():.1f}%")
    c3.metric("Μέσος δείκτης διάκρισης",f"{valid_disc.mean():.1f}%")
    c4.metric("Δείκτης υγείας",f"{health21:.1f}/100")
    c5.metric("Ανοικτές ενέργειες",open21)
    render_moodle_rest_service_panel(
        "Ποιότητα Αξιολόγησης",
        summary=f"Assessment health {health21:.1f}/100 · open actions {open21}",
        score=None,
        status="Assessment quality review",
        entity=ACTIVE_NAME if ACTIVE_DF is not None else "σύνολο δεδομένων Ποιότητας Αξιολόγησης",
        enabled=True,
        intervention_type="Ποιότητα Αξιολόγησης — Έλεγχος ποιότητας ερώτησης",
        note="",
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
        

# ==========================================================
# ΑΝΑΛΥΣΗ SCORM ΑΞΙΟΛΟΓΗΣΗΣ
# ==========================================================
elif section == "Ανάλυση SCORM Αξιολόγησης":
    render_module_header(
        "Ανάλυση SCORM Αξιολόγησης",
        "",
        "scorm_analysis"
    )
    with st.expander("🔎 Ρυθμίσεις σύνδεσης & εντοπισμός SCORM", expanded=False):
        default_scorm_url = st.session_state.get("moodle_base_url", "https://contentsolutions.eu") or "https://contentsolutions.eu"

        if st.session_state.get("scorm_discovery_ready", False):
            st.success("Η σύνδεση είναι ενεργή και τα στοιχεία πρόσβασης έχουν αποθηκευτεί μόνο για την τρέχουσα συνεδρία.")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.caption(
                    f"Moodle: {st.session_state.get('moodle_base_url', default_scorm_url)} · "
                    f"Μάθημα ID {st.session_state.get('scorm_course_id', 6)}"
                )
            with c2:
                if st.button("Αλλαγή στοιχείων", use_container_width=True, key="scorm_change_connection"):
                    st.session_state["scorm_discovery_ready"] = False
                    st.rerun()
            scorm_moodle_url = st.session_state.get("moodle_base_url", default_scorm_url)
            scorm_course_id = int(st.session_state.get("scorm_course_id", 6))
            scorm_token = st.session_state.get("moodle_shared_token", "")
        else:
            col_a, col_b = st.columns([2,1])
            with col_a:
                scorm_moodle_url = st.text_input(
                    "Moodle URL",
                    value=default_scorm_url,
                    key="scorm_moodle_url"
                ).rstrip("/")
            with col_b:
                scorm_course_id = st.number_input(
                    "ID Μαθήματος",
                    min_value=1,
                    value=int(st.session_state.get("scorm_course_id", 6)),
                    step=1,
                    key="scorm_course_id_input"
                )
            st.session_state["moodle_base_url"] = scorm_moodle_url
            st.session_state["scorm_course_id"] = int(scorm_course_id)

            secret_token = ""
            for secret_key in ("MOODLE_TOKEN", "moodle_token", "WSTOKEN", "wstoken"):
                try:
                    if secret_key in st.secrets and st.secrets[secret_key]:
                        secret_token = str(st.secrets[secret_key])
                        break
                except Exception:
                    pass

            scorm_token = st.text_input(
                "Κλειδί πρόσβασης Moodle",
                value=st.session_state.get("moodle_shared_token", secret_token),
                type="password",
                key="scorm_rest_token",
                help="Δίνεται μία φορά ανά συνεδρία και δεν εμφανίζεται μετά την επιτυχή σύνδεση."
            )
            if scorm_token:
                st.session_state["moodle_shared_token"] = scorm_token

            if st.button("🔎 Εντοπισμός SCORM και χρηστών", type="primary", key="scorm_discover"):
                if not scorm_moodle_url or not scorm_token:
                    st.error("Συμπληρώστε το Moodle URL και το κλειδί πρόσβασης.")
                else:
                    try:
                        contents = scorm_rest_call(
                            scorm_moodle_url, scorm_token, "core_course_get_contents",
                            {"courseid": int(scorm_course_id)}
                        )
                        found_scorms = []
                        for sec in contents if isinstance(contents, list) else []:
                            for mod in sec.get("modules", []):
                                if str(mod.get("modname", "")).lower() == "scorm":
                                    found_scorms.append({
                                        "section": sec.get("name"),
                                        "name": mod.get("name"),
                                        "cmid": mod.get("id"),
                                        "scorm_id": mod.get("instance"),
                                        "url": mod.get("url"),
                                    })
                        users = scorm_rest_call(
                            scorm_moodle_url, scorm_token, "core_enrol_get_enrolled_users",
                            {"courseid": int(scorm_course_id)}
                        )
                        user_rows = [
                            {"id": u.get("id"), "fullname": u.get("fullname"), "username": u.get("username")}
                            for u in users if isinstance(u, dict) and u.get("id")
                        ]
                        st.session_state["scorm_discovered_activities"] = found_scorms
                        st.session_state["scorm_discovered_users"] = user_rows
                        st.session_state["scorm_discovery_ready"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Αποτυχία εντοπισμού: {e}")

    found_scorms = st.session_state.get("scorm_discovered_activities", [])
    discovered_users = st.session_state.get("scorm_discovered_users", [])

    if found_scorms:
        st.success(
            f"Έτοιμα αποτελέσματα: {len(found_scorms)} δραστηριότητες SCORM και "
            f"{len(discovered_users)} εγγεγραμμένοι χρήστες."
        )
        with st.expander("📊 Προβολή αποτελεσμάτων SCORM", expanded=False):
            st.subheader("Επιλογή SCORM")
            st.dataframe(pd.DataFrame(found_scorms), use_container_width=True, hide_index=True)
            scorm_opts = {
                f"{x['name']} — SCORM ID {x['scorm_id']}": x for x in found_scorms
            }
            selected_scorm_label = st.selectbox("SCORM δραστηριότητα", list(scorm_opts.keys()), key="scorm_selected_label")
            selected_scorm = scorm_opts[selected_scorm_label]
            selected_scorm_id = int(selected_scorm["scorm_id"])

            st.markdown("### Ανάκτηση πραγματικών προσπαθειών")
            st.caption(
                "Η ομαδική ανάλυση διαβάζει τις υπάρχουσες προσπάθειες όλων των εγγεγραμμένων χρηστών. "
                "Χρήστες χωρίς προσπάθεια αγνοούνται."
            )

            min_n = st.number_input(
                "Ελάχιστες απαντήσεις ανά ερώτηση για υπολογισμό δείκτη διάκρισης",
                min_value=6, max_value=50, value=8, step=1,
                key="scorm_min_discrimination_n"
            )

            if st.button("📊 Ανάλυση όλων των υπαρχουσών προσπαθειών", key="scorm_fetch_all"):
                if not discovered_users:
                    st.error("Δεν έχει ανακτηθεί λίστα χρηστών. Πάτησε πρώτα «Εντοπισμός SCORM και χρηστών».")
                else:
                    progress = st.progress(0)
                    status_box = st.empty()
                    interaction_rows = []
                    attempt_summary_rows = []
                    errors = []
                    users_with_attempts = 0

                    try:
                        scoes = scorm_rest_call(
                            scorm_moodle_url, scorm_token, "mod_scorm_get_scorm_scoes",
                            {"scormid": selected_scorm_id}
                        )
                        sco_list = scoes.get("scoes", []) if isinstance(scoes, dict) else []

                        total_users = max(1, len(discovered_users))
                        for pos, u in enumerate(discovered_users, start=1):
                            uid = int(u["id"])
                            learner_name = str(u.get("fullname") or f"User {uid}")
                            status_box.caption(f"Έλεγχος {pos}/{total_users}: {learner_name}")
                            try:
                                count = scorm_rest_call(
                                    scorm_moodle_url, scorm_token, "mod_scorm_get_scorm_attempt_count",
                                    {"scormid": selected_scorm_id, "userid": uid}
                                )
                                attempts = int(count.get("attemptscount", count.get("attempts", 0)) or 0)
                                if attempts <= 0:
                                    progress.progress(pos/total_users)
                                    continue
                                users_with_attempts += 1

                                for attempt_no in range(1, attempts + 1):
                                    attempt_score = np.nan
                                    attempt_status = None
                                    attempt_total_time = None
                                    attempt_session_time = None
                                    attempt_interactions = []
                                    for sco in sco_list:
                                        sco_id = sco.get("id")
                                        if not sco_id:
                                            continue
                                        response = scorm_rest_call(
                                            scorm_moodle_url, scorm_token, "mod_scorm_get_scorm_sco_tracks",
                                            {"scoid": int(sco_id), "userid": uid, "attempt": attempt_no}
                                        )
                                        tracks = flatten_scorm_track_response(response)
                                        score_candidate = tracks.get("cmi.core.score.raw", tracks.get("score_raw"))
                                        if pd.notna(pd.to_numeric(pd.Series([score_candidate]), errors="coerce").iloc[0]):
                                            attempt_score = pd.to_numeric(pd.Series([score_candidate]), errors="coerce").iloc[0]
                                        attempt_status = tracks.get("cmi.core.lesson_status", tracks.get("status", attempt_status))
                                        attempt_total_time = tracks.get("cmi.core.total_time", tracks.get("total_time", attempt_total_time))
                                        attempt_session_time = tracks.get("session_time", tracks.get("cmi.core.session_time", attempt_session_time))

                                        attempt_interactions.extend(
                                            extract_scorm_interactions(
                                                tracks,
                                                user_id=uid,
                                                learner=learner_name,
                                                attempt=attempt_no,
                                                sco_id=sco_id,
                                                score=attempt_score,
                                                status=attempt_status,
                                            )
                                        )

                                    # ensure all rows receive final score/status found across SCOs
                                    for rr in attempt_interactions:
                                        rr["attempt_score"] = attempt_score
                                        rr["scorm_status"] = attempt_status
                                    interaction_rows.extend(attempt_interactions)
                                    attempt_summary_rows.append({
                                        "user_id": uid,
                                        "learner": learner_name,
                                        "attempt": attempt_no,
                                        "score": attempt_score,
                                        "status": attempt_status,
                                        "total_time": attempt_total_time,
                                        "session_time": attempt_session_time,
                                        "interactions": len(attempt_interactions),
                                    })
                            except Exception as user_exc:
                                errors.append(f"{learner_name} (ID {uid}): {user_exc}")
                            progress.progress(pos/total_users)

                        progress.empty()
                        status_box.empty()

                        interactions_df = pd.DataFrame(interaction_rows)
                        attempts_df = pd.DataFrame(attempt_summary_rows)
                        st.session_state["scorm_live_interactions"] = interactions_df
                        st.session_state["scorm_live_attempts"] = attempts_df
                        st.session_state["scorm_live_name"] = selected_scorm.get("name", f"SCORM {selected_scorm_id}")
                        st.session_state["scorm_live_id"] = selected_scorm_id
                        st.session_state["scorm_live_errors"] = errors

                        if interactions_df.empty:
                            st.warning("Βρέθηκαν προσπάθειες, αλλά δεν ανακτήθηκαν interactions για ανάλυση.")
                        else:
                            psych = scorm_question_statistics(interactions_df, min_discrimination_n=int(min_n))
                            st.session_state["scorm_live_psychometrics"] = psych
                            unified_scorm, learner_scorm = build_scorm_unified_dataset(
                                interactions_df, attempts_df, psych,
                                scorm_name=selected_scorm.get("name", f"SCORM {selected_scorm_id}"),
                                scorm_id=selected_scorm_id,
                            )
                            st.session_state["scorm_unified_dataset"] = unified_scorm
                            st.session_state["scorm_learner_summary"] = learner_scorm
                            st.success(
                                f"Ανακτήθηκαν {len(interactions_df)} πραγματικές απαντήσεις από "
                                f"{users_with_attempts} χρήστες με καταγεγραμμένη προσπάθεια."
                            )
                    except Exception as e:
                        progress.empty()
                        status_box.empty()
                        st.error(f"Αποτυχία SCORM ανάλυσης: {e}")

            interactions_df = st.session_state.get("scorm_live_interactions")
            attempts_df = st.session_state.get("scorm_live_attempts")
            psych = st.session_state.get("scorm_live_psychometrics")

            if isinstance(attempts_df, pd.DataFrame) and not attempts_df.empty:
                st.subheader("Προσπάθειες")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Χρήστες με προσπάθεια", attempts_df["user_id"].nunique())
                c2.metric("Συνολικές προσπάθειες", len(attempts_df))
                c3.metric("Μέσο score", f"{attempts_df['score'].dropna().mean():.1f}" if attempts_df["score"].notna().any() else "—")
                c4.metric("SCORM", st.session_state.get("scorm_live_name", "—"))
                st.dataframe(attempts_df, use_container_width=True, hide_index=True)

            if isinstance(interactions_df, pd.DataFrame) and not interactions_df.empty:
                st.subheader("Πραγματικά SCORM interactions")
                st.dataframe(interactions_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Λήψη interactions CSV",
                    interactions_df.to_csv(index=False).encode("utf-8-sig"),
                    "INELSO Academia_SCORM_interactions.csv",
                    "text/csv",
                    key="scorm_download_interactions"
                )

            if isinstance(psych, pd.DataFrame) and not psych.empty:
                st.subheader("Δείκτες ποιότητας ανά ερώτηση")
                st.caption(
                    "Facility = ποσοστό σωστών απαντήσεων. Discrimination = διαφορά επιτυχίας μεταξύ άνω και κάτω 27% "
                    "βάσει συνολικού score. Όταν το δείγμα δεν επαρκεί, ο δείκτης διάκρισης παραμένει μη εκτιμήσιμος."
                )
                psych_view = psych.copy()
                psych_view["κατάσταση"] = psych_view["διάκριση"].apply(κατηγορία_διάκρισης)
                psych_view["ενέργεια"] = psych_view.apply(ενέργεια_ερώτησης, axis=1)
                st.dataframe(psych_view, use_container_width=True, hide_index=True)

                n_learners = interactions_df["user_id"].nunique() if isinstance(interactions_df, pd.DataFrame) else 0
                if n_learners < int(min_n):
                    st.warning(
                        f"Υπάρχουν μόνο {n_learners} εκπαιδευόμενοι με interactions. "
                        f"Ο δείκτης διάκρισης απαιτεί τουλάχιστον {int(min_n)} και γι' αυτό δεν ερμηνεύεται ακόμη."
                    )

                unified_scorm = st.session_state.get("scorm_unified_dataset")
                learner_scorm = st.session_state.get("scorm_learner_summary")
                if isinstance(unified_scorm, pd.DataFrame) and not unified_scorm.empty:
                    st.subheader("Ενιαίο dataset από SCORM")
                    st.caption(
                        "Το κύριο σύνολο δεδομένων συνδυάζει πραγματικά δεδομένα ανά εκπαιδευόμενο και ανά ερώτηση. "
                        "Μπορεί να γίνει άμεσα ενεργή πηγή για όλη την πλατφόρμα ή να εξαχθεί σε CSV."
                    )
                    a,b,c = st.columns(3)
                    a.metric("Γραμμές Master Dataset", len(unified_scorm))
                    b.metric("Εκπαιδευόμενοι", unified_scorm["user_id"].nunique() if "user_id" in unified_scorm.columns else "—")
                    c.metric("Ερωτήσεις", unified_scorm["question_id"].nunique() if "question_id" in unified_scorm.columns else "—")

                    d1,d2,d3 = st.columns(3)
                    with d1:
                        st.download_button(
                            "⬇️ SCORM Master Dataset",
                            unified_scorm.to_csv(index=False).encode("utf-8-sig"),
                            "INELSO Academia_SCORM_MASTER_DATASET.csv",
                            "text/csv",
                            key="scorm_download_master",
                            use_container_width=True,
                        )
                    with d2:
                        if isinstance(learner_scorm, pd.DataFrame) and not learner_scorm.empty:
                            st.download_button(
                                "⬇️ Σύνοψη φοιτητών",
                                learner_scorm.to_csv(index=False).encode("utf-8-sig"),
                                "INELSO Academia_SCORM_LEARNER_SUMMARY.csv",
                                "text/csv",
                                key="scorm_download_learners",
                                use_container_width=True,
                            )
                    with d3:
                        st.download_button(
                            "⬇️ Item Statistics",
                            psych_view.to_csv(index=False).encode("utf-8-sig"),
                            "INELSO Academia_SCORM_ITEM_STATISTICS.csv",
                            "text/csv",
                            key="scorm_download_items",
                            use_container_width=True,
                        )

                    st.info(
                        "Συμβατότητα: Μαθησιακή Ανάλυση = άμεση, Ποιότητα Αξιολόγησης = άμεση, "
                        "Ενοποιημένες Ενδείξεις = μέσω των πραγματικών signals, Δημιουργία/Υποστήριξη Διδάσκοντα = context. "
                        "Η Προσαρμοστική Εξάσκηση είναι μερική, επειδή το SCORM δεν παρέχει από μόνο του βεβαιότητα ή κλίμακα αξιολόγησης RC-01–RC-05."
                    )

                    if st.button(
                        "✅ Χρήση SCORM Master Dataset σε όλη την πλατφόρμα",
                        type="primary",
                        key="scorm_activate_global",
                        use_container_width=True,
                    ):
                        mapped = psych[["ερώτηση","facility","διάκριση","αποτελεσματικότητα"]].copy()
                        st.session_state["custom_s21"] = mapped
                        st.session_state["custom_s21_source"] = f"Live SCORM — {st.session_state.get('scorm_live_name','SCORM')}"
                        st.session_state["inelso_scorm_active_df"] = unified_scorm.copy()
                        st.session_state["inelso_scorm_active_name"] = f"SCORM Master Dataset — {st.session_state.get('scorm_live_name','δραστηριότητα')}"
                        # SCORM becomes the chosen active source; external uploads remain a separate workflow and can replace it later.
                        st.session_state.pop("inelso_uploaded_clean", None)
                        st.session_state.pop("inelso_uploaded_original", None)
                        st.session_state.pop("inelso_uploaded_name", None)
                        st.success("Το SCORM Master Dataset έγινε η ενεργή πηγή για όλη την πλατφόρμα.")
                        st.rerun()

                if st.button("✅ Ενημέρωση μόνο της Ποιότητας Αξιολόγησης από το SCORM", key="scorm_activate_s21"):
                    mapped = psych[["ερώτηση","facility","διάκριση","αποτελεσματικότητα"]].copy()
                    st.session_state["custom_s21"] = mapped
                    st.session_state["custom_s21_source"] = f"Live SCORM — {st.session_state.get('scorm_live_name','SCORM')}"
                    st.success("Η Ποιότητα Αξιολόγησης ενημερώθηκε από τα πραγματικά SCORM δεδομένα. Η ενεργή πηγή των άλλων λειτουργιών δεν άλλαξε.")
                    st.rerun()

            errors = st.session_state.get("scorm_live_errors", [])
            if errors:
                with st.expander(f"⚠️ Παραλείψεις / σφάλματα χρηστών ({len(errors)})"):
                    for err in errors[:50]:
                        st.write("•", err)


# ==========================================================
# SERVICE 26
# ==========================================================
elif section == "Προσαρμοστική Εξάσκηση":
    if ACTIVE_DF is None:
        render_module_header("Προσαρμοστική Εξάσκηση", "", "s26")
        show_active_dataset_context("S26")
        c1,c2,c3,c4,c5 = st.columns([1, 1.35, 1, 1, 1])
        c1.metric("Τρέχουσα επίδοση","—")
        c2.metric("Κατάκτηση","—")
        c3.metric("Βεβαιότητα","—")
        c4.metric("Πρόοδος","—")
        c5.metric("Δείκτης υγείας","—")
        st.stop()
    if "custom_s26" not in st.session_state:
        render_module_header("Προσαρμοστική Εξάσκηση", "", "s26")
        show_active_dataset_context("S26")
        st.warning(
            "Το ενεργή πηγή έχει φορτωθεί, αλλά δεν έχει γίνει ακόμη mapping για Προσαρμοστική Εξάσκηση. "
            "Χρειάζονται πραγματικά πεδία ανά εκπαιδευόμενο, π.χ. εκπαιδευόμενος/βαθμολογία και, όπου υπάρχουν, προσπάθεια/βεβαιότητα/κλίμακα αξιολόγησης."
        )
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Τρέχουσα επίδοση","—")
        c2.metric("Κατάκτηση","—")
        c3.metric("Βεβαιότητα","—")
        c4.metric("Πρόοδος","—")
        c5.metric("Δείκτης υγείας","—")
        st.stop()

    render_module_header("Προσαρμοστική Εξάσκηση", "", "s26")
    show_active_dataset_context("S26")
    st.caption("Αναλύει την επίδοση ανά κριτήριο, την πρόοδο και τη βεβαιότητα και προτείνει το επόμενο εκπαιδευτικό βήμα.")

    with st.expander("📘 Τι είναι η Rubric και πώς περνά στο σύστημα;", expanded=False):
        st.markdown(
            "<div style='font-family:Segoe UI,Arial,sans-serif;font-size:1.06rem;line-height:1.65;color:#26374d;'>"
            "Η <strong>κλίμακα αξιολόγησης</strong> είναι ο πίνακας κριτηρίων με τον οποίο αξιολογείται η απάντηση."
            "</div>",
            unsafe_allow_html=True
        )
        st.caption("Στην προτεινόμενη δομή χρησιμοποιούνται πέντε κριτήρια:")
        rubric_df = pd.DataFrame([
            {"Κωδικός": "RC-01", "Τι αξιολογεί": "Εννοιολογική ακρίβεια", "Βάρος": "30%"},
            {"Κωδικός": "RC-02", "Τι αξιολογεί": "Μέθοδος / συλλογισμός", "Βάρος": "25%"},
            {"Κωδικός": "RC-03", "Τι αξιολογεί": "Τεκμηρίωση", "Βάρος": "15%"},
            {"Κωδικός": "RC-04", "Τι αξιολογεί": "Πληρότητα", "Βάρος": "15%"},
            {"Κωδικός": "RC-05", "Τι αξιολογεί": "Σαφήνεια επικοινωνίας", "Βάρος": "15%"},
        ])
        st.dataframe(
            rubric_df,
            use_container_width=True,
            hide_index=True,
            height=245
        )
        st.markdown(
            "<div style='font-family:Segoe UI,Arial,sans-serif;font-size:1rem;line-height:1.6;color:#41536b;'>"
            "Στο CSV κάθε προσπάθεια είναι <strong>ξεχωριστή γραμμή</strong>. "
            "Παράδειγμα: ο ίδιος εκπαιδευόμενος εμφανίζεται με προσπάθεια 1, προσπάθεια 2 κ.ο.κ.<br>"
            "Οι στήλες <strong>RC-01 έως RC-05</strong> παίρνουν επίπεδο 0–4. "
            "Η βαθμολογία είναι 0–100 και η βεβαιότητα 0–1."
            "</div>",
            unsafe_allow_html=True
        )

    mapped26_view = st.session_state["custom_s26"].copy()
    if "κατάκτηση" not in mapped26_view.columns:
        mapped26_view["κατάκτηση"] = np.where(
            pd.to_numeric(mapped26_view["score"], errors="coerce") >= 70,
            "Επαρκής",
            "Αναπτυσσόμενο"
        )
    if "κατάσταση" not in mapped26_view.columns:
        mapped26_view["κατάσταση"] = mapped26_view.apply(
            lambda r: κατάσταση_26(
                float(r.get("score", 0)) if pd.notna(r.get("score", np.nan)) else 0,
                float(r.get("βεβαιότητα", 0)) if pd.notna(r.get("βεβαιότητα", np.nan)) else 0,
                float(r.get("RC-01", 2)) if pd.notna(r.get("RC-01", np.nan)) else 2
            ),
            axis=1
        )
    mapped26_view["μαθητής"] = mapped26_view["μαθητής"].astype(str)
    learner_options = sorted(mapped26_view["μαθητής"].dropna().unique().tolist())
    selected_learner = st.selectbox(
        "Φοιτητής για προβολή",
        learner_options,
        key="s26_active_learner"
    )
    service26_current = (
        mapped26_view[mapped26_view["μαθητής"] == selected_learner]
        .sort_values("προσπάθεια")
        .reset_index(drop=True)
    )
    latest = service26_current.iloc[-1]
    prev = service26_current.iloc[0]
    gain = float(latest["score"])-float(prev["score"])

    c1,c2,c3,c4,c5 = st.columns([1, 1.35, 1, 1, 1])
    c1.metric("Τρέχουσα επίδοση",f"{latest['score']:.0f}/100")
    c2.metric("Κατάκτηση",latest["κατάκτηση"])
    c3.metric("Βεβαιότητα",f"{latest['βεβαιότητα']:.2f}")
    c4.metric("Πρόοδος",f"+{gain:.2f}")
    c5.metric("Δείκτης υγείας",f"{health26:.1f}/100")
    render_moodle_rest_service_panel(
        "Προσαρμοστική Εξάσκηση",
        summary=(
            f"Adaptive practice · mastery {latest['κατάκτηση']} · "
            f"βεβαιότητα {float(latest['βεβαιότητα']):.2f} · "
            f"progress +{gain:.2f}"
        ),
        score=float(latest["score"]),
        status=str(latest["κατάσταση"]),
        entity=selected_learner,
        enabled=True,
        intervention_type=(
            "Προσαρμοστική Εξάσκηση — Εμπλουτισμός"
            if str(latest["κατάσταση"]) == "ΕΤΟΙΜΟ"
            else "Προσαρμοστική Εξάσκηση — Ανθρώπινος έλεγχος"
            if str(latest["κατάσταση"]) == "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ"
            else "Προσαρμοστική Εξάσκηση — Νέα προσπάθεια"
        ),
        note="",
    )


    view = st.radio("Προβολή",["Μαθητής","Εκπαιδευτής"],horizontal=True)
    if view == "Μαθητής":
        tabs = st.tabs(["Προσαρμοστικό προφίλ","Κενά κριτηρίων","Πορεία προόδου","Πώς λαμβάνεται η απόφαση"])
        with tabs[0]:
            _status26 = str(latest["κατάσταση"])
            if _status26 == "ΕΤΟΙΜΟ":
                st.success("ΕΤΟΙΜΟ — η βεβαιότητα είναι επαρκής και δεν ενεργοποιείται κρίσιμος κανόνας.")
                st.write("**Προτεινόμενο επόμενο βήμα:** εμπλουτισμός / επόμενη δραστηριότητα.")
            elif _status26 == "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ":
                st.warning("ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ — απαιτείται εκπαιδευτική αξιολόγηση πριν από επόμενη ενέργεια.")
                st.write("**Προτεινόμενο επόμενο βήμα:** ανθρώπινος έλεγχος.")
            else:
                st.info("ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ — το αποτέλεσμα ανά φοιτητή υποδεικνύει στοχευμένη νέα προσπάθεια.")
                st.write("**Προτεινόμενο επόμενο βήμα:** νέα προσπάθεια / στοχευμένη εξάσκηση.")
        with tabs[1]:
            rows=[]
            plot_first, plot_latest, plot_index = [], [], []
            for _,rr in rubric.iterrows():
                code=rr["κριτήριο"]
                v1 = pd.to_numeric(pd.Series([prev.get(code, np.nan)]), errors="coerce").iloc[0]
                v2 = pd.to_numeric(pd.Series([latest.get(code, np.nan)]), errors="coerce").iloc[0]
                gap1 = (4-int(v1)) if pd.notna(v1) else np.nan
                gap2 = (4-int(v2)) if pd.notna(v2) else np.nan
                rows.append({
                    "Κριτήριο":code,
                    "Περιγραφή":rr["περιγραφή"],
                    "Κενό 1ης προσπάθειας":gap1,
                    "Κενό τελευταίας προσπάθειας":gap2,
                    "Βάρος (%)":rr["βάρος"]
                })
                if pd.notna(v1) and pd.notna(v2):
                    plot_index.append(code); plot_first.append(v1); plot_latest.append(v2)
            gd=pd.DataFrame(rows)
            st.dataframe(gd,use_container_width=True,hide_index=True)
            if plot_index:
                st.line_chart(pd.DataFrame({
                    "1η προσπάθεια":plot_first,
                    "Τελευταία προσπάθεια":plot_latest
                }, index=plot_index))
        with tabs[2]:
            st.line_chart(service26_current.set_index("προσπάθεια")[["score","βεβαιότητα"]])
            st.dataframe(service26_current,use_container_width=True,hide_index=True)
        with tabs[3]:
            st.markdown("""
- Βεβαιότητα < 0,70 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- RC-01 επίπεδο 0–1 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- Επίδοση < 70 με επαρκή βεβαιότητα → **ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ**
- Επίδοση ≥ 70 με επαρκή βεβαιότητα → **ΕΤΟΙΜΟ**
""")
    else:
        with st.expander("📋 Προβολή ουράς ενεργειών εκπαιδευτή", expanded=False):
            q = cohort26[cohort26["κατάσταση"]!="ΕΤΟΙΜΟ"].copy()
            q["προτεραιότητα"] = q["κατάσταση"].map({"ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ":1,"ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ":2})
            st.dataframe(q.sort_values("προτεραιότητα"),use_container_width=True,hide_index=True)

# ==========================================================
# SERVICE 31 — ΠΡΑΓΜΑΤΙΚΗ ΣΥΝΔΕΣΗ
# ==========================================================

# ==========================================================
# SERVICE 25 — AI-ASSISTED COURSE & ASSESSMENT AUTHORING
# ==========================================================
elif section == "Δημιουργία Μαθήματος & Αξιολόγησης":
    if ACTIVE_DF is None:
        render_module_header("Δημιουργία Μαθήματος & Αξιολόγησης", "", "s25")
        show_active_dataset_context("S25")
        st.metric("Κατάσταση","—")
        st.stop()

    render_module_header("Δημιουργία Μαθήματος & Αξιολόγησης", "", "s25")
    show_active_dataset_context("S25")

    s25_tabs = st.tabs(["Δημιουργία", "Οδηγίες", "Προσχέδιο"])

    with s25_tabs[0]:
        s25_content = st.text_area(
            "Υλικό μαθήματος / πηγή",
            height=180,
            placeholder="Επικολλήστε σημειώσεις, μαθησιακούς στόχους, κεφάλαιο ή άλλο εκπαιδευτικό περιεχόμενο.",
            key="s25_content"
        )

        use_csv_s25 = st.checkbox(
            "Χρήση του ενεργού CSV ως πρόσθετου πλαισίου",
            value=ACTIVE_DF is not None,
            key="s25_use_active_csv"
        )

        if use_csv_s25 and ACTIVE_DF is not None:
            with st.expander("Προβολή πλαισίου από το ενεργό αρχείο", expanded=False):
                p = dataset_profile(ACTIVE_DF)
                st.caption(
                    f"Αρχείο: {active_dataset_name()} · "
                    f"{p['rows']} γραμμές · {p['cols']} στήλες · "
                    f"{p['missing']} κενές τιμές · {p['duplicates']} διπλότυπα"
                )
                st.markdown("**Μεταβλητές του ενεργού αρχείου**")
                st.dataframe(
                    pd.DataFrame({
                        "Α/Α": range(1, len(ACTIVE_DF.columns) + 1),
                        "Μεταβλητή": [str(col) for col in ACTIVE_DF.columns]
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=min(800, 38 * (len(ACTIVE_DF.columns) + 1))
                )

        c1, c2, c3 = st.columns(3)
        with c1:
            s25_type = st.selectbox(
                "Τύπος προσχεδίου",
                ["Τράπεζα ερωτήσεων", "Κουίζ / Σύντομο τεστ", "Κλίμακα αξιολόγησης", "Ασκήσεις εξάσκησης"],
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
            "Μαθησιακός στόχος",
            placeholder="π.χ. Ο φοιτητής να μπορεί να εφαρμόζει...",
            key="s25_goal"
        )

        if st.button("Δημιουργία προσχεδίου", type="primary", key="s25_generate"):
            raw_manual = " ".join(s25_content.strip().split())
            raw_csv = csv_context_as_text(ACTIVE_DF, max_rows=3) if use_csv_s25 and ACTIVE_DF is not None else ""
            raw = " ".join((raw_manual + " " + raw_csv).split())
            if not raw:
                st.error("Χρειάζεται εκπαιδευτικό περιεχόμενο ή ενεργό πλαίσιο από CSV.")
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

                    if s25_type == "Τράπεζα ερωτήσεων":
                        items.append({
                            "ID": f"Q{i+1:02d}",
                            "Προσχέδιο ερώτησης": f"Εξήγησε ή εφάρμοσε την ακόλουθη έννοια: {short}",
                            "Τύπος": "Ανοικτή απάντηση",
                            "Επίπεδο": s25_level,
                            "Έλεγχος διδάσκοντα": "ΑΠΑΙΤΕΙΤΑΙ"
                        })
                    elif s25_type == "Κουίζ / Σύντομο τεστ":
                        items.append({
                            "ID": f"Q{i+1:02d}",
                            "Προσχέδιο ερώτησης": f"Ποια διατύπωση αποδίδει καλύτερα το ακόλουθο σημείο; {short}",
                            "Σωστή απάντηση / κριτήριο": short,
                            "Επίπεδο": s25_level,
                            "Έλεγχος διδάσκοντα": "ΑΠΑΙΤΕΙΤΑΙ"
                        })
                    elif s25_type == "Κλίμακα αξιολόγησης":
                        criteria = ["Ακρίβεια", "Μεθοδολογία", "Τεκμηρίωση", "Πληρότητα", "Σαφήνεια"]
                        criterion = criteria[i % len(criteria)]
                        weights = [30, 25, 15, 15, 15]
                        items.append({
                            "Κριτήριο": criterion,
                            "Προτεινόμενο βάρος": weights[i % len(weights)],
                            "Περιγραφή": (
                                f"Αξιολόγηση ως προς {criterion.lower()} σε σχέση με τον στόχο: "
                                f"{learning_goal or 'τον δηλωμένο μαθησιακό στόχο'}."
                            ),
                            "Έλεγχος διδάσκοντα": "ΑΠΑΙΤΕΙΤΑΙ"
                        })
                    else:
                        items.append({
                            "Άσκηση": f"Άσκηση {i+1}",
                            "Προσχέδιο": f"Εφάρμοσε την έννοια «{short}» σε νέο παράδειγμα και αιτιολόγησε τα βήματά σου.",
                            "Επίπεδο": s25_level,
                            "Έλεγχος διδάσκοντα": "ΑΠΑΙΤΕΙΤΑΙ"
                        })

                st.session_state["s25_draft"] = pd.DataFrame(items)
                st.success("Το προσχέδιο δημιουργήθηκε.")

    with s25_tabs[1]:
        with st.expander("🧩 Οδηγίες για Κουίζ", expanded=False):
            st.markdown("""
1. Ξεκινήστε από έναν σαφή μαθησιακό στόχο.
2. Κάθε ερώτηση να ελέγχει μία βασική γνώση ή δεξιότητα.
3. Ορίστε σαφή σωστή απάντηση ή κανόνα βαθμολόγησης.
4. Αποφύγετε αμφίσημες διατυπώσεις.
5. Μετά από πραγματικές προσπάθειες, χρησιμοποιήστε την **Ποιότητα Quiz** από το αριστερό μενού.
""")

        with st.expander("📋 Οδηγίες για Κλίμακα Αξιολόγησης", expanded=False):
            st.markdown("""
1. Ορίστε 3–6 παρατηρήσιμα κριτήρια.
2. Δώστε σαφή επίπεδα επίδοσης, π.χ. 0–4.
3. Ορίστε βάρη που αθροίζουν σε 100%.
4. Για προσαρμοστική εξάσκηση υποστηρίζεται η δομή RC-01 έως RC-05.
5. Χρησιμοποιήστε την ίδια κλίμακα με συνέπεια σε όλες τις προσπάθειες.
""")

    with s25_tabs[2]:
        if "s25_draft" not in st.session_state:
            st.info("Δεν έχει δημιουργηθεί ακόμη προσχέδιο.")
        else:
            edited_s25 = st.data_editor(
                st.session_state["s25_draft"],
                use_container_width=True,
                hide_index=True,
                key="s25_editor"
            )

            faculty_ok = st.checkbox(
                "Έλεγξα και εγκρίνω το περιεχόμενο.",
                key="s25_faculty_ok"
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Έγκριση προσχεδίου",
                    disabled=not faculty_ok,
                    use_container_width=True,
                    key="s25_review_queue"
                ):
                    st.session_state["s25_approved_draft"] = edited_s25.copy()
                    st.success("Το προσχέδιο εγκρίθηκε στην τρέχουσα συνεδρία.")

            with c2:
                st.download_button(
                    "Λήψη προσχεδίου CSV",
                    edited_s25.to_csv(index=False).encode("utf-8-sig"),
                    "INELSO_Academia_Προσχέδιο_Αξιολόγησης.csv",
                    "text/csv",
                    key="s25_download",
                    use_container_width=True
                )

# ==========================================================
# SERVICE 28 — FACULTY TEACHING AMPLIFICATION
# ==========================================================
elif section == "Υποστήριξη Διδάσκοντα":
    if ACTIVE_DF is None:
        render_module_header("Υποστήριξη Διδάσκοντα", "", "s28")
        show_active_dataset_context("S28")
        st.metric("Κατάσταση","—")
        st.stop()

    render_module_header("Υποστήριξη Διδάσκοντα", "", "s28")
    show_active_dataset_context("S28")

    s28_tabs = st.tabs(["Δημιουργία", "Προσχέδιο", "Αποθηκευμένα", "Moodle"])

    with s28_tabs[0]:
        s28_mode = st.selectbox(
            "Επιλέξτε λειτουργία",
            [
                "Προσχέδιο απάντησης σε συχνή ερώτηση",
                "Διευκρίνιση κλίμακας αξιολόγησης",
                "Προσχέδιο ανατροφοδότησης"
            ],
            key="s28_mode"
        )

        if s28_mode == "Προσχέδιο απάντησης σε συχνή ερώτηση":
            st.caption("Δημιουργεί προσχέδιο απάντησης με βάση το εγκεκριμένο πλαίσιο.")
        elif s28_mode == "Διευκρίνιση κλίμακας αξιολόγησης":
            st.caption("Δημιουργεί διευκρίνιση για κριτήρια και επίπεδα της κλίμακας αξιολόγησης.")
        else:
            st.caption("Δημιουργεί προσχέδιο ανατροφοδότησης για συγκεκριμένη περίπτωση φοιτητή.")

        source_context = st.text_area(
            "Εγκεκριμένο πλαίσιο μαθήματος / πολιτική / κλίμακα αξιολόγησης",
            height=140,
            placeholder="Επικόλλησε μόνο το εγκεκριμένο υλικό στο οποίο πρέπει να βασιστεί η απάντηση.",
            key="s28_context"
        )

        use_csv_s28 = st.checkbox(
            "Συμπερίληψη ενεργού CSV ως τεκμηριωμένου πλαισίου",
            value=ACTIVE_DF is not None,
            key="s28_use_active_csv"
        )
        if use_csv_s28 and ACTIVE_DF is not None:
            with st.expander("Προβολή πλαισίου από το ενεργό αρχείο", expanded=False):
                p = dataset_profile(ACTIVE_DF)
                st.caption(
                    f"Αρχείο: {active_dataset_name()} · "
                    f"{p['rows']} γραμμές · {p['cols']} στήλες · "
                    f"{p['missing']} κενές τιμές · {p['duplicates']} διπλότυπα"
                )
                st.markdown("**Μεταβλητές του ενεργού αρχείου**")
                st.dataframe(
                    pd.DataFrame({
                        "Α/Α": range(1, len(ACTIVE_DF.columns) + 1),
                        "Μεταβλητή": [str(col) for col in ACTIVE_DF.columns]
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=min(800, 38 * (len(ACTIVE_DF.columns) + 1))
                )

        student_input = st.text_area(
            "Ερώτηση φοιτητή / περίπτωση",
            height=100,
            placeholder="π.χ. Τι ακριβώς ζητά το κριτήριο τεκμηρίωσης;",
            key="s28_question"
        )

        if st.button("Δημιουργία προσχεδίου", type="primary", key="s28_generate"):
            combined_context = source_context.strip()
            if use_csv_s28 and ACTIVE_DF is not None:
                combined_context = (
                    combined_context + "\n\n" + csv_context_as_text(ACTIVE_DF, max_rows=3)
                ).strip()

            if not combined_context or not student_input.strip():
                st.error("Χρειάζονται πλαίσιο και ερώτηση/περίπτωση.")
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

                if s28_mode == "Προσχέδιο απάντησης σε συχνή ερώτηση":
                    draft = (
                        "Προτεινόμενο προσχέδιο απάντησης προς έλεγχο διδάσκοντα:\n\n"
                        f"Με βάση το εγκεκριμένο υλικό, τα πιο σχετικά σημεία είναι:\n{evidence_text}\n\n"
                        "Προτεινόμενη απάντηση: Παρακαλώ ακολούθησε τα παραπάνω σημεία του μαθήματος."
                    )
                elif s28_mode == "Διευκρίνιση κλίμακας αξιολόγησης":
                    draft = (
                        "Προσχέδιο διευκρίνισης κλίμακας αξιολόγησης προς έλεγχο διδάσκοντα:\n\n"
                        f"Σχετικά αποσπάσματα:\n{evidence_text}\n\n"
                        "Το κριτήριο πρέπει να ερμηνευτεί σύμφωνα με την εγκεκριμένη κλίμακα αξιολόγησης."
                    )
                else:
                    draft = (
                        "Προσχέδιο ανατροφοδότησης προς έλεγχο διδάσκοντα:\n\n"
                        f"Σχετικά σημεία του πλαισίου μαθήματος:\n{evidence_text}\n\n"
                        "Αναγνώρισε τι έγινε σωστά, σύνδεσε τη βελτίωση με το αντίστοιχο κριτήριο "
                        "και δώσε μία συγκεκριμένη επόμενη ενέργεια."
                    )

                st.session_state["s28_draft"] = draft
                st.success("Το προσχέδιο δημιουργήθηκε. Δείτε το tab «Προσχέδιο».")

    with s28_tabs[1]:
        if "s28_draft" not in st.session_state:
            st.info("Δεν έχει δημιουργηθεί ακόμη προσχέδιο.")
        else:
            s28_edited = st.text_area(
                "Προσχέδιο",
                value=st.session_state["s28_draft"],
                height=220,
                key="s28_edited"
            )

            s28_approve = st.checkbox(
                "Ο έλεγχος του διδάσκοντα ολοκληρώθηκε.",
                key="s28_approve"
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Αποθήκευση εγκεκριμένου προσχεδίου",
                    disabled=not s28_approve,
                    use_container_width=True,
                    key="s28_save"
                ):
                    if "s28_library" not in st.session_state:
                        st.session_state["s28_library"] = []
                    st.session_state["s28_library"].append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": st.session_state.get("s28_mode",""),
                        "question": st.session_state.get("s28_question",""),
                        "draft": s28_edited,
                        "status": "ΕΓΚΕΚΡΙΜΕΝΟ"
                    })
                    st.success("Το προσχέδιο αποθηκεύτηκε.")

            with c2:
                if st.button(
                    "Προετοιμασία για Moodle",
                    use_container_width=True,
                    key="s28_stage"
                ):
                    st.session_state["pending_moodle_note_text"] = s28_edited
                    st.session_state["pending_moodle_intervention_type"] = "Υποστήριξη Διδάσκοντα — Εγκεκριμένη ανατροφοδότηση"
                    st.success("Το προσχέδιο είναι έτοιμο για ελεγχόμενη καταγραφή στο Moodle.")

    with s28_tabs[2]:
        if st.session_state.get("s28_library"):
            lib_df = pd.DataFrame(st.session_state["s28_library"])
            st.dataframe(lib_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Λήψη προσχεδίων CSV",
                lib_df.to_csv(index=False).encode("utf-8-sig"),
                "INELSO_Academia_Εγκεκριμένα_Προσχέδια.csv",
                "text/csv",
                key="s28_download"
            )
        else:
            st.info("Δεν υπάρχουν ακόμη αποθηκευμένα προσχέδια.")

    with s28_tabs[3]:
        s28_scorm_ready = bool(st.session_state.get("s28_library"))
        render_moodle_rest_service_panel(
            "Υποστήριξη Διδάσκοντα",
            summary=("Εγκεκριμένο προσχέδιο" if s28_scorm_ready else "Αναμένεται έλεγχος διδάσκοντα"),
            score=None,
            status="completed" if s28_scorm_ready else "incomplete",
            entity="Υποστήριξη Διδάσκοντα",
            enabled=True,
            intervention_type="Υποστήριξη Διδάσκοντα — Εγκεκριμένη ανατροφοδότηση",
            note="",
        )


elif section == "Ενοποιημένες Μαθησιακές Ενδείξεις":
    if ACTIVE_DF is None:
        render_module_header("Ενοποιημένες Μαθησιακές Ενδείξεις", "", "s31")
        show_active_dataset_context("S31")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Συνολικά ενδείξεις","—")
        c2.metric("Υψηλή προτεραιότητα","—")
        c3.metric("Μεσαία προτεραιότητα","—")
        c4.metric("Δείκτης υγείας","—")
        st.stop()
    if ("custom_s21" not in st.session_state) and ("custom_s26" not in st.session_state):
        render_module_header("Ενοποιημένες Μαθησιακές Ενδείξεις", "", "s31")
        show_active_dataset_context("S31")
        st.warning(
            "Υπάρχει ενεργή πηγή, αλλά δεν υπάρχουν ακόμη πραγματικά αντιστοιχισμένες ενδείξεις από Ποιότητα Αξιολόγησης ή Προσαρμοστική Εξάσκηση. "
            "Η ενότητα Ενοποιημένων Μαθησιακών Ενδείξεων δεν δημιουργεί τεχνητές ενδείξεις."
        )
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Συνολικά ενδείξεις","—")
        c2.metric("Υψηλή προτεραιότητα","—")
        c3.metric("Μεσαία προτεραιότητα","—")
        c4.metric("Δείκτης υγείας","—")
        st.stop()

    render_module_header("Ενοποιημένες Μαθησιακές Ενδείξεις", "", "s31")
    show_active_dataset_context("S31")
    

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Συνολικά ενδείξεις",len(service31_integrated))
    c2.metric("Υψηλή προτεραιότητα",high31)
    c3.metric("Μεσαία προτεραιότητα",medium31)
    c4.metric("Δείκτης υγείας",f"{health31:.1f}/100")
    render_moodle_rest_service_panel(
        "Ενοποιημένες Μαθησιακές Ενδείξεις",
        summary=f"Ενοποιημένες ενδείξεις {len(service31_integrated)} · υψηλής προτεραιότητας {high31}",
        score=None,
        status="Ενοποιημένη μαθησιακή ένδειξη",
        entity="Ενοποιημένες μαθησιακές ενδείξεις",
        enabled=True,
        intervention_type="Ενοποιημένες Μαθησιακές Ενδείξεις — Μαθησιακή ένδειξη",
        note="",
    )


    if "s31_view" not in st.session_state:
        st.session_state["s31_view"] = "Ενιαίες ενδείξεις"

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Ενιαίες ενδείξεις", use_container_width=True, key="s31_btn_signals"):
        st.session_state["s31_view"] = "Ενιαίες ενδείξεις"
    if b2.button("Προτεραιότητες", use_container_width=True, key="s31_btn_priorities"):
        st.session_state["s31_view"] = "Προτεραιότητες"
    if b3.button("Προέλευση", use_container_width=True, key="s31_btn_provenance"):
        st.session_state["s31_view"] = "Προέλευση"
    if b4.button("Αλυσίδα παρέμβασης", use_container_width=True, key="s31_btn_chain"):
        st.session_state["s31_view"] = "Αλυσίδα παρέμβασης"

    display31 = service31_integrated.rename(columns={
        "οντότητα":"Οντότητα",
        "επίπεδο_ένδειξης":"Επίπεδο ένδειξης",
        "πηγή":"Πηγή",
        "τύπος":"Τύπος ένδειξης",
        "ισχύς_σήματος":"Ισχύς σήματος",
        "κατεύθυνση":"Κατεύθυνση",
        "ενέργεια":"Ενέργεια",
        "προέλευση":"Προέλευση",
        "προτεραιότητα":"Προτεραιότητα"
    })

    current_s31_view = st.session_state["s31_view"]

    if current_s31_view == "Ενιαίες ενδείξεις":
        st.dataframe(display31,use_container_width=True,hide_index=True)

    elif current_s31_view == "Προτεραιότητες":
        sorted31 = service31_integrated.sort_values(
            ["προτεραιότητα","ισχύς_σήματος"],
            ascending=[True,False]
        ).rename(columns={
            "οντότητα":"Οντότητα",
            "επίπεδο_ένδειξης":"Επίπεδο ένδειξης",
            "πηγή":"Πηγή",
            "τύπος":"Τύπος ένδειξης",
            "ισχύς_σήματος":"Ισχύς σήματος",
            "κατεύθυνση":"Κατεύθυνση",
            "ενέργεια":"Ενέργεια",
            "προέλευση":"Προέλευση",
            "προτεραιότητα":"Προτεραιότητα"
        })
        st.dataframe(sorted31,use_container_width=True,hide_index=True)
        st.warning("Υψηλή προτεραιότητα σημαίνει ότι η ένδειξη απαιτεί ανθρώπινο έλεγχο πριν από οποιαδήποτε εκπαιδευτική παρέμβαση.")

    elif current_s31_view == "Προέλευση":
        prov = service31_integrated[["οντότητα","επίπεδο_ένδειξης","πηγή","τύπος","προέλευση"]].copy()
        prov.columns = ["Οντότητα","Επίπεδο ένδειξης","Πηγή","Τύπος ένδειξης","Προέλευση"]
        st.dataframe(prov,use_container_width=True,hide_index=True)

    elif current_s31_view == "Αλυσίδα παρέμβασης":
        st.markdown("""
1. Εντοπισμός ένδειξης στην Ποιότητα Quiz ή στην Προσαρμοστική Εξάσκηση  
2. Διατήρηση της πραγματικής προέλευσης και του τύπου της ένδειξης  
3. Ιεράρχηση της ένδειξης χωρίς επινοημένη θεματική ετικέτα  
4. Έλεγχος από τον εκπαιδευτή ή τον σχεδιαστή αξιολόγησης  
5. Επιλογή κατάλληλης παρέμβασης: έλεγχος ερώτησης, νέα προσπάθεια, εμπλουτισμός ή ανθρώπινη υποστήριξη  
6. Παρακολούθηση της επόμενης προσπάθειας ή της επόμενης ομάδας εκπαιδευομένων  
7. Ολοκλήρωση της ενέργειας μόνο όταν υπάρχει νέα τεκμηρίωση
""")

# ==========================================================
# ΔΙΑΣΥΝΔΕΣΗ ΥΠΗΡΕΣΙΩΝ
# ==========================================================
elif section == "Ενιαία Ροή Ενεργειών":
    render_module_header("Ενιαία Ροή Ενεργειών", "", "flow")
    show_active_dataset_context("S31")

    actions=[]
    if "custom_s21" in st.session_state:
        for _,r in service21[
            (service21["κατάσταση"]=="ΚΡΙΣΙΜΗ") |
            (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
            (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
        ].iterrows():
            actions.append({
                "Προτεραιότητα":1 if r["κατάσταση"] in ["ΚΡΙΣΙΜΗ","ΜΗ ΕΚΤΙΜΗΣΙΜΗ"] else 2,
                "Ενότητα":"Ποιότητα Αξιολόγησης",
                "Οντότητα":r["ερώτηση"],
                "Ένδειξη":r["κατάσταση"],
                "Ενέργεια":r["ενέργεια"]
            })

    if "custom_s26" in st.session_state:
        for _,r in cohort26[cohort26["κατάσταση"]!="ΕΤΟΙΜΟ"].iterrows():
            actions.append({
                "Προτεραιότητα":1 if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else 2,
                "Ενότητα":"Προσαρμοστική Εξάσκηση",
                "Οντότητα":r["μαθητής"],
                "Ένδειξη":r["κατάσταση"],
                "Ενέργεια":"Ανθρώπινος έλεγχος" if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else "Στοχευμένη νέα προσπάθεια"
            })

    for _,r in service31_integrated[service31_integrated["προτεραιότητα"]=="ΥΨΗΛΗ"].iterrows():
        actions.append({
            "Προτεραιότητα":2,
            "Ενότητα":"Ενοποιημένες Μαθησιακές Ενδείξεις",
            "Οντότητα":r["οντότητα"],
            "Ένδειξη":r["τύπος"],
            "Ενέργεια":r["ενέργεια"]
        })

    action_df = pd.DataFrame(actions)
    if not action_df.empty:
        action_df = action_df.sort_values(["Προτεραιότητα","Ενότητα","Οντότητα"])

    with st.expander("📋 Προβολή αποτελεσμάτων", expanded=False):
        if action_df.empty:
            st.info("Δεν υπάρχουν διαθέσιμα αποτελέσματα.")
        else:
            st.dataframe(action_df,use_container_width=True,hide_index=True)

    with st.expander("🔎 Αναλυτικός έλεγχος ενέργειας", expanded=False):
        if action_df.empty:
            st.info("Δεν υπάρχουν διαθέσιμες ενέργειες για αναλυτικό έλεγχο.")
        else:
            labels=[
                f"{r['Ενότητα']} — {r['Οντότητα']} — {r['Ένδειξη']}"
                for _,r in action_df.iterrows()
            ]
            selected=st.selectbox("Επιλέξτε ενέργεια",labels,key="flow_selected_action")
            idx=labels.index(selected)
            rr=action_df.iloc[idx]
            a,b,c=st.columns(3)
            a.metric("Ενότητα",rr["Ενότητα"])
            b.metric("Προτεραιότητα",rr["Προτεραιότητα"])
            c.metric("Ένδειξη",rr["Ένδειξη"])
            st.write("**Οντότητα:**",rr["Οντότητα"])
            st.write("**Προτεινόμενη ενέργεια:**",rr["Ενέργεια"])
        


# ==========================================================
# ΚΥΚΛΟΣ ΖΩΗΣ ΕΝΕΡΓΕΙΩΝ
# ==========================================================
elif section == "Κύκλος Ζωής Ενεργειών":
    render_module_header("Κύκλος Ζωής Ενεργειών", "", "lifecycle")
    show_active_dataset_context("S31")

    actions=[]
    if "custom_s21" in st.session_state:
        for _,r in service21[
            (service21["κατάσταση"]=="ΕΛΕΓΧΟΣ") |
            (service21["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ")
        ].iterrows():
            actions.append({
                "Κωδικός":f"ΕΡΩΤΗΣΗ-{r['ερώτηση'].split()[-1]}",
                "Ενότητα":"Ποιότητα Αξιολόγησης",
                "Οντότητα":r["ερώτηση"],
                "Προτεραιότητα":1 if r["κατάσταση"]=="ΜΗ ΕΚΤΙΜΗΣΙΜΗ" else 2,
                "Ένδειξη":r["κατάσταση"],
                "Προτεινόμενη ενέργεια":r["ενέργεια"]
            })

    if "custom_s26" in st.session_state:
        for i,(_,r) in enumerate(cohort26[cohort26["κατάσταση"]!="ΕΤΟΙΜΟ"].iterrows(), start=1):
            actions.append({
                "Κωδικός":f"ΕΞΑΣΚΗΣΗ-{i:02d}",
                "Ενότητα":"Προσαρμοστική Εξάσκηση",
                "Οντότητα":r["μαθητής"],
                "Προτεραιότητα":1 if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else 2,
                "Ένδειξη":r["κατάσταση"],
                "Προτεινόμενη ενέργεια":"Ανθρώπινος έλεγχος" if r["κατάσταση"]=="ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ" else "Στοχευμένη νέα προσπάθεια"
            })

    for i,(_,r) in enumerate(service31_integrated[service31_integrated["προτεραιότητα"]=="ΥΨΗΛΗ"].iterrows(), start=1):
        actions.append({
            "Κωδικός":f"ΕΝΔΕΙΞΗ-{i:02d}",
            "Ενότητα":"Ενοποιημένες Μαθησιακές Ενδείξεις",
            "Οντότητα":r["οντότητα"],
            "Προτεραιότητα":2,
            "Ένδειξη":r["τύπος"],
            "Προτεινόμενη ενέργεια":r["ενέργεια"]
        })

    lifecycle = pd.DataFrame(actions)
    if not lifecycle.empty:
        lifecycle = lifecycle.sort_values(["Προτεραιότητα","Ενότητα","Οντότητα"])

    if "κατάσταση_ενεργειών" not in st.session_state:
        st.session_state["κατάσταση_ενεργειών"] = {}
    if not lifecycle.empty:
        for code in lifecycle["Κωδικός"].tolist():
            st.session_state["κατάσταση_ενεργειών"].setdefault(code, "ΝΕΑ")

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
        a.metric("Ενότητα", row["Ενότητα"])
        b.metric("Προτεραιότητα", row["Προτεραιότητα"])
        c.metric("Ένδειξη", row["Ένδειξη"])

        st.write("**Οντότητα:**", row["Οντότητα"])
        st.write("**Προτεινόμενη ενέργεια:**", row["Προτεινόμενη ενέργεια"])

        current = st.session_state["κατάσταση_ενεργειών"].get(code, "ΝΕΑ")
        lifecycle_widget_key = f"lifecycle_state_{code}"
        if lifecycle_widget_key not in st.session_state:
            st.session_state[lifecycle_widget_key] = current

        def _update_lifecycle_state(action_code=code, widget_key=lifecycle_widget_key):
            st.session_state["κατάσταση_ενεργειών"][action_code] = st.session_state[widget_key]

        st.radio(
            "Κατάσταση κύκλου ζωής",
            ["ΝΕΑ","ΣΕ ΕΛΕΓΧΟ","ΚΛΕΙΣΤΗ"],
            key=lifecycle_widget_key,
            horizontal=True,
            on_change=_update_lifecycle_state
        )

        export_lifecycle = lifecycle.copy()
        export_lifecycle["Κατάσταση κύκλου ζωής"] = export_lifecycle["Κωδικός"].map(
            st.session_state["κατάσταση_ενεργειών"]
        )
        st.dataframe(export_lifecycle,use_container_width=True,hide_index=True)
        st.download_button(
            "Λήψη μητρώου ενεργειών",
            export_lifecycle.to_csv(index=False).encode("utf-8-sig"),
            "INELSO Academia_Mitroo_Energeion.csv",
            "text/csv"
        )

# ==========================================================
# ΜΕΘΟΔΟΛΟΓΙΑ & ΠΡΟΕΛΕΥΣΗ
# ==========================================================
elif section == "Μεθοδολογία & Προέλευση":
    render_module_header(
        "Μεθοδολογία & Προέλευση",
        "",
        "methodology"
    )

    provenance = pd.DataFrame([
        {
            "Ενότητα":"Ποιότητα Quiz",
            "Πηγή": (f"ενεργή πηγή: {ACTIVE_NAME}" if "custom_s21" in st.session_state else (f"ενεργή πηγή: {ACTIVE_NAME} — μη συμβατή με την Ποιότητα Quiz" if ACTIVE_DF is not None else "Μη διαθέσιμο")),
            "Τύπος":"Ψυχομετρικές ενδείξεις ανά ερώτηση",
            "Χρήση":"Ποιότητα ερωτήσεων και ουρά ελέγχου",
            "Περιορισμός":"Δεν επιτρέπεται αυτόματη διαγραφή/αλλαγή ερώτησης"
        },
        {
            "Ενότητα":"Προσαρμοστική Εξάσκηση",
            "Πηγή": (f"ενεργή πηγή: {ACTIVE_NAME}" if "custom_s26" in st.session_state else (f"ενεργή πηγή: {ACTIVE_NAME} — δεν έχει αντιστοιχιστεί για την Προσαρμοστική Εξάσκηση" if ACTIVE_DF is not None else "Μη διαθέσιμο")),
            "Τύπος":"Επίπεδα κριτηρίων, επίδοση, κατάκτηση, βεβαιότητα",
            "Χρήση":"Νέα προσπάθεια / εμπλουτισμός / ανθρώπινος έλεγχος",
            "Περιορισμός":"Δεν αλλάζει επίσημο βαθμό Moodle"
        },
        {
            "Ενότητα":"Ενοποιημένες Μαθησιακές Ενδείξεις",
            "Πηγή": f"Παράγωγες ενδείξεις — ενεργό σύνολο δεδομένων: {ACTIVE_NAME}" if ACTIVE_DF is not None else "Παράγωγες ενδείξεις από Ποιότητα Quiz + Προσαρμοστική Εξάσκηση",
            "Τύπος":"Ενοποιημένες μαθησιακές ενδείξεις",
            "Χρήση":"Σύνθεση και προτεραιοποίηση",
            "Περιορισμός":"Δεν επινοούνται θεματικές αντιστοιχίσεις χωρίς τεκμηρίωση"
        },
    ])

    tab_overview, tab_source, tab_s21, tab_s26, tab_s31, tab_safety = st.tabs([
        "🧭 Επισκόπηση",
        "🗂️ Προέλευση δεδομένων",
        "🧪 Ποιότητα Quiz",
        "🎯 Προσαρμοστική Εξάσκηση",
        "🔗 Μαθησιακές Ενδείξεις",
        "🛡️ Όρια & ασφάλεια",
    ])

    with tab_overview:
        st.markdown("### Πώς λειτουργεί η μεθοδολογία")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>1 · Πηγή</div><h3>Πραγματικά δεδομένα</h3><p>Αρχεία CSV/Excel, SCORM ή επιτρεπόμενη ανάγνωση από το Moodle.</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>2 · Έλεγχος</div><h3>Συμβατότητα</h3><p>Κάθε λειτουργία χρησιμοποιεί μόνο τα πεδία που πραγματικά υποστηρίζει.</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>3 · Ανάλυση</div><h3>Διαφανείς κανόνες</h3><p>Οι δείκτες και τα όρια είναι ορατά και ελέγξιμα.</p></div>", unsafe_allow_html=True)
        with c4:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>4 · Απόφαση</div><h3>Ανθρώπινος έλεγχος</h3><p>Η πλατφόρμα υποστηρίζει· δεν αντικαθιστά την εκπαιδευτική κρίση.</p></div>", unsafe_allow_html=True)

        with st.expander("🔍 Τι σημαίνει «προέλευση» και γιατί είναι σημαντική;"):
            st.write(
                "Κάθε αποτέλεσμα πρέπει να μπορεί να συνδεθεί με την αρχική πηγή δεδομένων, "
                "τον τύπο της ανάλυσης και τον κανόνα που χρησιμοποιήθηκε. "
                "Έτσι αποφεύγονται μη τεκμηριωμένες ερμηνείες και διατηρείται πλήρης ιχνηλασιμότητα."
            )

    with tab_source:
        st.markdown("### Προέλευση δεδομένων ανά λειτουργία")
        st.dataframe(provenance, use_container_width=True, hide_index=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Ενεργή πηγή", ACTIVE_NAME if ACTIVE_DF is not None else "—")
        c2.metric("Ποιότητα Quiz", service_dataset_compatibility("S21", ACTIVE_DF) if ACTIVE_DF is not None else "—")
        c3.metric("Προσαρμοστική Εξάσκηση", service_dataset_compatibility("S26", ACTIVE_DF) if ACTIVE_DF is not None else "—")

        with st.expander("📌 Πώς διαβάζω τον πίνακα προέλευσης;"):
            st.markdown(
                "- **Πηγή:** από πού προέρχονται τα δεδομένα.\n"
                "- **Τύπος:** ποια δομή δεδομένων χρησιμοποιείται.\n"
                "- **Χρήση:** ποια ανάλυση επιτρέπεται να γίνει.\n"
                "- **Περιορισμός:** τι δεν επιτρέπεται να συμπεράνουμε ή να αλλάξουμε αυτόματα."
            )

    with tab_s21:
        st.markdown("### Κανόνες Ποιότητας Αξιολόγησης")
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Ισχυρή διάκριση", "≥ 40%")
        r2.metric("Ικανοποιητική", "20–39.99%")
        r3.metric("Έλεγχος", "0–19.99%")
        r4.metric("Κρίσιμη", "< 0%")

        with st.expander("📊 Αναλυτικοί κανόνες διάκρισης", expanded=True):
            st.markdown("""
- Διάκριση ≥ 40% → **ΙΣΧΥΡΗ**
- 20%–39,99% → **ΙΚΑΝΟΠΟΙΗΤΙΚΗ**
- 0%–19,99% → **ΕΛΕΓΧΟΣ**
- Αρνητική διάκριση → **ΚΡΙΣΙΜΗ**
- Μη εκτιμήσιμη διάκριση → απαιτεί ανθρώπινο έλεγχο scoring / response pattern
""")
    with tab_s26:
        st.markdown("### Κανόνες Προσαρμοστικής Εξάσκησης")
        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Confidence review", "< 0.70")
        a2.metric("RC-01 review", "0–1")
        a3.metric("Retry", "Score < 70")
        a4.metric("Ready", "Score ≥ 70")

        with st.expander("🎯 Αναλυτικοί κανόνες προσαρμοστικής απόφασης", expanded=True):
            st.markdown("""
- Βεβαιότητα < 0,70 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- RC-01 επίπεδο 0–1 → **ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ**
- Επίδοση < 70 με επαρκή βεβαιότητα → **ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ**
- Επίδοση ≥ 70 με επαρκή βεβαιότητα → **ΕΤΟΙΜΟ**
- Μέγιστος αριθμός προσπαθειών → **3**
""")
        with st.expander("🧩 Τι σημαίνει ο περιορισμός RC-01;"):
            st.write(
                "Το RC-01 αφορά εννοιολογική ακρίβεια. Πολύ χαμηλή επίδοση στο RC-01 "
                "δεν επιτρέπεται να θεωρηθεί υψηλή κατάκτηση ακόμη και αν ο συνολικός score είναι καλύτερος."
            )
        with st.expander("👤 Πότε απαιτείται ανθρώπινος έλεγχος;"):
            st.write(
                "Όταν η βεβαιότητα είναι χαμηλή ή όταν το κρίσιμο κριτήριο RC-01 βρίσκεται σε επίπεδο 0–1. "
                "Το σύστημα δεν παίρνει αυτόματα παιδαγωγική απόφαση."
            )

    with tab_s31:
        st.markdown("### Κανόνες Ενοποιημένων Μαθησιακών Ενδείξεων")
        i1,i2,i3 = st.columns(3)
        with i1:
            st.markdown(
                "<div class='inelso-feature-card'><div class='kicker'>Προέλευση</div>"
                "<h3>Ποιότητα Quiz + Προσαρμοστική Εξάσκηση</h3>"
                "<p>Συνδυάζονται μόνο πραγματικές ενδείξεις από τις δύο λειτουργίες.</p></div>",
                unsafe_allow_html=True
            )
        with i2:
            st.markdown(
                "<div class='inelso-feature-card'><div class='kicker'>Απόφαση</div>"
                "<h3>Όχι αυτόματα</h3>"
                "<p>Η πλατφόρμα ιεραρχεί· η τελική απόφαση παραμένει στον διδάσκοντα.</p></div>",
                unsafe_allow_html=True
            )
        with i3:
            st.markdown(
                "<div class='inelso-feature-card'><div class='kicker'>Υψηλή προτεραιότητα</div>"
                "<h3>Ανθρώπινος έλεγχος</h3>"
                "<p>Κάθε κρίσιμη ένδειξη εξετάζεται πριν από οποιαδήποτε ενέργεια.</p></div>",
                unsafe_allow_html=True
            )

        with st.expander("🔗 Πώς συντίθενται οι ενδείξεις;", expanded=True):
            st.markdown("""
- Συνθέτει ενδείξεις που ήδη υπάρχουν στην Ποιότητα Quiz και στην Προσαρμοστική Εξάσκηση.
- Διατηρεί την αρχική προέλευση κάθε ένδειξης.
- Η προτεραιότητα δεν ισοδυναμεί με αυτόματη εκπαιδευτική απόφαση.
- Κάθε υψηλής προτεραιότητας ένδειξη οδηγεί πρώτα σε ανθρώπινο έλεγχο.
""")
        with st.expander("🧾 Τι ΔΕΝ κάνει η σύνθεση;"):
            st.write(
                "Δεν δημιουργεί τεχνητές θεματικές συσχετίσεις, δεν επινοεί καταστάσεις εκπαιδευομένων "
                "και δεν μετατρέπει την προτεραιότητα σε αυτόματη ενέργεια."
            )

    with tab_safety:
        st.markdown("### Όρια, ασφάλεια και επιτρεπτές ενέργειες")
        s1,s2,s3 = st.columns(3)
        with s1:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>Βαθμοί</div><h3>Δεν αλλάζουν αυτόματα</h3><p>Η πλατφόρμα δεν τροποποιεί αυτόματα επίσημους βαθμούς στο Moodle.</p></div>", unsafe_allow_html=True)
        with s2:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>Παρεμβάσεις</div><h3>Απαιτείται ρητή έγκριση</h3><p>Κρίσιμες ενέργειες προωθούνται μόνο μετά από ανθρώπινη επιβεβαίωση.</p></div>", unsafe_allow_html=True)
        with s3:
            st.markdown("<div class='inelso-feature-card'><div class='kicker'>Καταγραφή στο Moodle</div><h3>Απαιτείται επαλήθευση</h3><p>Η καταγραφή θεωρείται ολοκληρωμένη μόνο μετά από επιβεβαίωση του αποτελέσματος.</p></div>", unsafe_allow_html=True)

        with st.expander("🛡️ Βασικοί κανόνες ασφαλούς λειτουργίας", expanded=True):
            st.markdown("""
- Δεν εμφανίζονται τεχνητά αποτελέσματα όταν δεν υπάρχει συμβατό ενεργό σύνολο δεδομένων.
- Δεν αλλάζουν αυτόματα επίσημοι βαθμοί στο Moodle.
- Δεν εκτελείται κρίσιμη καταγραφή στο Moodle χωρίς ρητή ανθρώπινη έγκριση.
- Η απόκριση του συστήματος από μόνη της δεν αποτελεί τελική επιβεβαίωση.
- Η τελική επιβεβαίωση γίνεται μετά την καταγραφή, με έλεγχο του αποτελέσματος και του ιστορικού ενεργειών.
""")

# ==========================================================
# ΕΛΕΓΧΟΣ ΕΤΟΙΜΟΤΗΤΑΣ
# ==========================================================
elif section == "Εργαστήριο Ανάλυσης Δεδομένων":
    render_module_header("Εργαστήριο Ανάλυσης Δεδομένων", "", "analytics_lab")
    show_active_dataset_context("ANALYTICS")

    source_options = ["Χρήση τελευταίου αρχείου από το Κέντρο Δεδομένων"]
    if "inelso_uploaded_original" not in st.session_state:
        st.warning("Δεν υπάρχει ακόμη ανεβασμένο σύνολο δεδομένων. Πήγαινε πρώτα στο «Κέντρο Εισαγωγής & Επεξεργασίας Δεδομένων» και φόρτωσε ένα αρχείο.")
        lab_df = None
    else:
        source = st.radio("Πηγή δεδομένων", source_options, horizontal=True)
        lab_df = st.session_state.get(
            "inelso_uploaded_clean",
            st.session_state["inelso_uploaded_original"]
        ).copy()

    if lab_df is not None and len(lab_df) > 0:
        st.success(f"Ενεργό σύνολο δεδομένων: {st.session_state.get('inelso_uploaded_name','Ανεβασμένο αρχείο')}")

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

        diag_box = st.expander("🩺 Αυτόματη διάγνωση", expanded=False)
        corr_box = st.expander("🔗 Συσχετίσεις & σχέσεις", expanded=False)
        class_box = st.expander("🎯 Κατηγοριοποίηση", expanded=False)
        reg_box = st.expander("📈 Παλινδρόμηση", expanded=False)
        cluster_box = st.expander("🧩 Ομαδοποίηση", expanded=False)
        group_box = st.expander("👥 Σύγκριση ομάδων", expanded=False)
        report_box = st.expander("📄 Αυτόματη αναφορά", expanded=False)

        # ------------------------------------------------------
        # Automated diagnosis
        # ------------------------------------------------------
        with diag_box:
            st.subheader("Αυτόματη διάγνωση συνόλου δεδομένων")

            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in lab_df.columns if c not in numeric_cols]

            diagnosis = pd.DataFrame([
                {"Έλεγχος":"Κενές τιμές","Τιμή":f"{missing_rate*100:.2f}%","Ερμηνεία":"Χαμηλότερο είναι καλύτερο"},
                {"Έλεγχος":"Διπλότυπες γραμμές","Τιμή":f"{duplicate_rate*100:.2f}%","Ερμηνεία":"Ελέγξτε αν είναι πραγματικές επαναλήψεις"},
                {"Έλεγχος":"Αριθμητικές στήλες","Τιμή":len(numeric_cols),"Ερμηνεία":"Κατάλληλες για παλινδρόμηση, συσχέτιση ή ομαδοποίηση"},
                {"Έλεγχος":"Μη αριθμητικές στήλες","Τιμή":len(categorical_cols),"Ερμηνεία":"Κατάλληλες κατηγορικές μεταβλητές"},
                {"Έλεγχος":"Σταθερές στήλες","Τιμή":constant_cols,"Ερμηνεία":"Συνήθως δεν προσφέρουν χρήσιμη πληροφορία για τη μηχανική μάθηση"},
            ])
            st.dataframe(diagnosis, use_container_width=True, hide_index=True)

            st.subheader("Προτεινόμενες αναλύσεις")
            suggestions = []
            if len(numeric_cols) >= 2:
                suggestions.append("Συσχετίσεις και παλινδρόμηση")
                suggestions.append("Ομαδοποίηση")
            if categorical_cols:
                suggestions.append("Κατηγοριοποίηση, εφόσον επιλεγεί κατηγορική μεταβλητή-στόχος")
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
        with corr_box:
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
        # Κατηγοριοποίηση
        # ------------------------------------------------------
        with class_box:
            st.subheader("Κατηγοριοποίηση")
            st.caption("Επιλέξτε μια κατηγορική μεταβλητή-στόχο. Το σύστημα εκπαιδεύει και αξιολογεί το μοντέλο με ελεγχόμενο διαχωρισμό των δεδομένων, ώστε να αποφεύγεται η διαρροή πληροφοριών.")

            target = st.selectbox(
                "Μεταβλητή-στόχος για κατηγοριοποίηση",
                ["— επιλέξτε —"] + list(lab_df.columns),
                key="class_target"
            )

            if target != "— επιλέξτε —":
                y = lab_df[target]
                unique_y = y.dropna().nunique()

                if unique_y < 2:
                    st.error("Η μεταβλητή-στόχος έχει λιγότερες από 2 κατηγορίες.")
                elif unique_y > max(30, int(0.2*len(lab_df))):
                    st.warning("Η μεταβλητή-στόχος έχει πολλές μοναδικές τιμές. Ίσως δεν είναι κατάλληλη για κατηγοριοποίηση.")
                else:
                    available = [c for c in lab_df.columns if c != target]
                    predictors = st.multiselect(
                        "Προβλεπτικές μεταβλητές",
                        available,
                        default=available[:min(8,len(available))],
                        key="class_predictors"
                    )

                    model_name = st.radio(
                        "Μοντέλο",
                        ["Λογιστική παλινδρόμηση","Τυχαίο δάσος"],
                        horizontal=True
                    )

                    test_size = st.slider("Ποσοστό συνόλου ελέγχου", 20, 40, 30, 5) / 100

                    if st.button("Εκτέλεση κατηγοριοποίησης", key="run_class"):
                        if len(predictors) == 0:
                            st.error("Επιλέξτε τουλάχιστον μία προβλεπτική μεταβλητή.")
                        else:
                            data = lab_df[predictors+[target]].dropna(subset=[target]).copy()
                            X = data[predictors]
                            y = data[target].astype(str)

                            if y.value_counts().min() < 2:
                                st.error("Υπάρχει κατηγορία της μεταβλητής-στόχου με λιγότερες από 2 εγγραφές.")
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

                                if model_name == "Λογιστική παλινδρόμηση":
                                    model = LogisticΠαλινδρόμηση(max_iter=2000)
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

                                    if model_name == "Τυχαίο δάσος":
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
                                    st.error(f"Δεν ήταν δυνατή η εκτέλεση της κατηγοριοποίησης: {e}")

        # ------------------------------------------------------
        # Παλινδρόμηση
        # ------------------------------------------------------
        with reg_box:
            st.subheader("Παλινδρόμηση")
            numeric_targets = lab_df.select_dtypes(include=np.number).columns.tolist()
            target_reg = st.selectbox(
                "Αριθμητική μεταβλητή-στόχος",
                ["— επιλέξτε —"] + numeric_targets,
                key="reg_target"
            )

            if target_reg != "— επιλέξτε —":
                available = [c for c in lab_df.columns if c != target_reg]
                predictors = st.multiselect(
                    "Προβλεπτικές μεταβλητές",
                    available,
                    default=available[:min(8,len(available))],
                    key="reg_predictors"
                )

                model_name = st.radio(
                    "Μοντέλο παλινδρόμησης",
                    ["Γραμμική παλινδρόμηση","Παλινδρόμηση με τυχαίο δάσος"],
                    horizontal=True
                )

                if st.button("Εκτέλεση παλινδρόμησης", key="run_reg"):
                    if not predictors:
                        st.error("Επιλέξτε τουλάχιστον μία προβλεπτική μεταβλητή.")
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
                            LinearΠαλινδρόμηση()
                            if model_name == "Γραμμική παλινδρόμηση"
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

                            if model_name == "Παλινδρόμηση με τυχαίο δάσος":
                                feature_names = pipe.named_steps["prep"].get_feature_names_out()
                                imp = pipe.named_steps["model"].feature_importances_
                                fi = pd.DataFrame({
                                    "Χαρακτηριστικό":feature_names,
                                    "Σημαντικότητα":imp
                                }).sort_values("Σημαντικότητα",ascending=False)
                                st.subheader("Feature importance")
                                st.dataframe(fi.head(25),use_container_width=True,hide_index=True)
                        except Exception as e:
                            st.error(f"Δεν ήταν δυνατή η εκτέλεση της παλινδρόμησης: {e}")

        # ------------------------------------------------------
        # Ομαδοποίηση
        # ------------------------------------------------------
        with cluster_box:
            st.subheader("Ομαδοποίηση")
            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()

            selected = st.multiselect(
                "Αριθμητικές μεταβλητές για ομαδοποίηση",
                numeric_cols,
                default=numeric_cols[:min(5,len(numeric_cols))],
                key="cluster_vars"
            )
            k = st.slider("Αριθμός ομάδων",2,8,3)

            if st.button("Εκτέλεση ομαδοποίησης K-means", key="run_cluster"):
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
                        st.error("Δεν υπάρχουν αρκετές εγγραφές για τον επιλεγμένο αριθμό ομάδων.")
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
                        st.subheader("Μέσο προφίλ ανά ομάδα")
                        st.dataframe(profile.round(3),use_container_width=True)

                        st.download_button(
                            "Λήψη συνόλου δεδομένων με ομάδες",
                            clustered.to_csv(index=False).encode("utf-8-sig"),
                            "INELSO_Academia_σύνολο_δεδομένων_με_ομάδες.csv",
                            "text/csv"
                        )

        # ------------------------------------------------------
        # Group comparison
        # ------------------------------------------------------
        with group_box:
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
        with report_box:
            st.subheader("Αυτόματη αναφορά")

            numeric_cols = lab_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = [c for c in lab_df.columns if c not in numeric_cols]

            report_text = f"""INELSO Academia — ΑΥΤΟΜΑΤΗ ΑΝΑΦΟΡΑ

Σύνολο δεδομένων: {st.session_state.get('inelso_uploaded_name','Ανεβασμένο αρχείο')}
Γραμμές: {len(lab_df)}
Στήλες: {len(lab_df.columns)}
Δείκτης ποιότητας: {quality_score:.1f}/100
Ποσοστό κενών τιμών: {missing_rate*100:.2f}%
Ποσοστό διπλοεγγραφών: {duplicate_rate*100:.2f}%
Αριθμητικές μεταβλητές: {len(numeric_cols)}
Κατηγορικές/λοιπές μεταβλητές: {len(categorical_cols)}
Σταθερές στήλες: {constant_cols}

Ερμηνεία:
- Ο δείκτης ποιότητας αποτελεί λειτουργικό δείκτη ελέγχου και όχι στατιστικά επικυρωμένη μετρική.
- Οι προβλεπτικές αναλύσεις χρησιμοποιούν διαχωρισμό δεδομένων εκπαίδευσης και ελέγχου, καθώς και κατάλληλη προεπεξεργασία.
- Τα αποτελέσματα πρέπει να ερμηνεύονται με βάση το ερευνητικό ερώτημα και τη δομή του συνόλου δεδομένων.
"""
            st.text_area("Προεπισκόπηση αναφοράς",report_text,height=320)
            st.download_button(
                "Λήψη αυτόματης αναφοράς",
                report_text.encode("utf-8"),
                "INELSO Academia_Analytics_Lab_Report.txt",
                "text/plain"
            )

# ==========================================================
# ΠΡΟΣΟΜΟΙΩΤΗΣ WHAT-IF
# ==========================================================
elif section == "Προσομοίωση Σεναρίων":
    render_module_header("Προσομοίωση Σεναρίων", "", "whatif")
    show_active_dataset_context("S26")

    st.subheader("Προσαρμοστική Εξάσκηση — δοκιμή κατωφλίων")

    if "custom_s26" not in st.session_state:
        render_empty_state(
            "Η προσομοίωση είναι διαθέσιμη μόλις ενεργοποιηθεί κατάλληλη πηγή δεδομένων",
            "Φορτώστε ή συνδέστε δεδομένα και επιστρέψτε εδώ για δοκιμή διαφορετικών κατωφλίων και σεναρίων.",
            "Μετά την ενεργοποίηση της πηγής, οι επιλογές προσομοίωσης εμφανίζονται αυτόματα."
        )
        st.stop()

    st.caption(f"Πηγή προσομοίωσης: {ACTIVE_NAME}")

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

    sim = st.session_state['custom_s26'].copy()

    # Το custom_s26 αποθηκεύει τα βασικά mapped δεδομένα και μπορεί να μην
    # περιλαμβάνει ακόμη την αρχική επιχειρησιακή κατάσταση. Τη δημιουργούμε
    # με την ίδια λογική που χρησιμοποιείται στην Προσαρμοστική Εξάσκηση,
    # ώστε η προσομοίωση να μπορεί να συγκρίνει παλιά και νέα κατάσταση.
    if "κατάσταση" not in sim.columns:
        sim["κατάσταση"] = sim.apply(
            lambda r: κατάσταση_26(
                float(r.get("score", 0)) if pd.notna(r.get("score", np.nan)) else 0,
                float(r.get("βεβαιότητα", 0)) if pd.notna(r.get("βεβαιότητα", np.nan)) else 0,
                float(r.get("RC-01", 2)) if pd.notna(r.get("RC-01", np.nan)) else 2
            ),
            axis=1
        )

    def simulate_status(row):
        # Η προσομοίωση μεταβάλλει μόνο τα επιλεγμένα κατώφλια βεβαιότητας και βαθμολογίας.
        if row["βεβαιότητα"] < confidence_threshold:
            return "ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ"
        if row["score"] < ready_score_threshold:
            return "ΝΕΑ ΠΡΟΣΠΑΘΕΙΑ"
        return "ΕΤΟΙΜΟ"

    sim["Νέα κατάσταση"] = sim.apply(simulate_status, axis=1)

    old_counts = sim["κατάσταση"].value_counts()
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
    st.info(f"Με τα επιλεγμένα όρια αλλάζουν κατηγορία {changed} από {len(sim)} εκπαιδευόμενους.")

    st.subheader("Σενάρια")
    scenarios = pd.DataFrame([
        {"Σενάριο":"Βασικό","Όριο βεβαιότητας":0.70,"Όριο βαθμολογίας":70},
        {"Σενάριο":"Πιο αυστηρό","Όριο βεβαιότητας":0.80,"Όριο βαθμολογίας":80},
        {"Σενάριο":"Πιο ελαστικό","Όριο βεβαιότητας":0.60,"Όριο βαθμολογίας":60},
    ])
    st.dataframe(scenarios,use_container_width=True,hide_index=True)

    st.warning("Η προσομοίωση γίνεται σε ασφαλές δοκιμαστικό περιβάλλον. Δεν επηρεάζει το Moodle ούτε τις επίσημες αποφάσεις.")


# ==========================================================
# LIVE MOODLE
# ==========================================================
elif section == "Ζωντανή Σύνδεση Moodle":
    render_module_header(
        "Ζωντανή Σύνδεση Moodle",
        "",
        "moodle_live"
    )
    show_active_dataset_context("REPORT")

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

    default_live_url = st.session_state.get("moodle_base_url", "https://contentsolutions.eu") or "https://contentsolutions.eu"
    default_live_token = st.session_state.get("moodle_shared_token", "")

    tab_conn, tab_courses, tab_course_data, tab_advanced = st.tabs([
        "🔐 Σύνδεση",
        "📚 Μαθήματα",
        "👥 Δεδομένα μαθήματος",
        "🛠️ Προχωρημένα"
    ])

    with tab_conn:
        st.markdown("### Σύνδεση Moodle")
        st.caption("Η φόρμα παραμένει κλειστή όταν δεν χρειάζεται, για καθαρότερη προβολή.")
        with st.expander("🔐 Άνοιγμα ρυθμίσεων σύνδεσης", expanded=not st.session_state.get("moodle_connection_verified", False)):
            st.markdown("### Σύνδεση Moodle")
            st.caption("Συμπληρώστε τα στοιχεία μία φορά. Παραμένουν διαθέσιμα όσο διαρκεί η τρέχουσα συνεδρία.")

            moodle_url = st.text_input(
                "Βασικό URL Moodle",
                value=default_live_url,
                key="live_shared_moodle_url"
            ).rstrip("/")

            token = st.text_input(
                "Κλειδί πρόσβασης Moodle",
                value=default_live_token,
                type="password",
                key="live_shared_moodle_token",
                help="Χρησιμοποιείται μόνο μέσα στην τρέχουσα συνεδρία."
            )

            if moodle_url:
                st.session_state["moodle_base_url"] = moodle_url
            if token:
                st.session_state["moodle_shared_token"] = token

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("🔌 Έλεγχος σύνδεσης", type="primary", key="live_check_connection"):
                    if not moodle_url or not token:
                        st.error("Συμπληρώστε URL και κλειδί πρόσβασης.")
                    else:
                        try:
                            info = moodle_rest_call(
                                moodle_url, token, "core_webservice_get_site_info"
                            )
                            st.session_state["moodle_site_info"] = info
                            st.session_state["moodle_connection_verified"] = True
                            st.success("Η σύνδεση με το Moodle λειτουργεί.")
                        except Exception as e:
                            st.session_state["moodle_connection_verified"] = False
                            st.error(f"Αποτυχία σύνδεσης: {e}")

            with c2:
                if st.session_state.get("moodle_connection_verified"):
                    info = st.session_state.get("moodle_site_info", {})
                    st.markdown(
                        f"**Συνδεδεμένο:** {info.get('sitename','Moodle')}  \\n"
                        f"**Λογαριασμός:** {info.get('fullname','—')}  \\n"
                        f"**Έκδοση:** {info.get('release','—')}"
                    )
                else:
                    st.caption("Μετά τον επιτυχή έλεγχο, δεν χρειάζεται να ξαναδώσετε τα στοιχεία στις άλλες καρτέλες.")


    # Shared values are refreshed after the connection tab widgets run.
    moodle_url = st.session_state.get("moodle_base_url", default_live_url)
    token = st.session_state.get("moodle_shared_token", default_live_token)

    with tab_courses:
        st.markdown("### Μαθήματα")
        st.caption("Ανακτήστε μία φορά τη λίστα και επιλέξτε το μάθημα που θα χρησιμοποιηθεί στις επόμενες καρτέλες.")

        if st.button("📥 Ανάκτηση μαθημάτων", type="primary", key="live_load_courses"):
            if not moodle_url or not token:
                st.error("Πηγαίνετε πρώτα στην καρτέλα «Σύνδεση».")
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
            display_cols = [x for x in ["id","shortname","fullname","visible"] if x in courses_df.columns]
            st.dataframe(
                courses_df[display_cols] if display_cols else courses_df,
                use_container_width=True,
                hide_index=True
            )
            if "id" in courses_df.columns and len(courses_df):
                course_options = {
                    f"{r.get('shortname','')} — {r.get('fullname','')} (ID {r['id']})": int(r["id"])
                    for _, r in courses_df.iterrows()
                }
                labels = list(course_options.keys())
                previous_id = st.session_state.get("moodle_selected_course_id")
                default_idx = 0
                if previous_id is not None:
                    for idx, lab in enumerate(labels):
                        if course_options[lab] == previous_id:
                            default_idx = idx
                            break
                selected_course_label = st.selectbox(
                    "Επιλέξτε μάθημα",
                    labels,
                    index=default_idx,
                    key="live_course_selector"
                )
                st.session_state["moodle_selected_course_id"] = course_options[selected_course_label]
                st.success(f"Ενεργό μάθημα: {selected_course_label}")

    with tab_course_data:
        st.markdown("### Δεδομένα μαθήματος")
        selected_course_id = st.session_state.get("moodle_selected_course_id")
        if not selected_course_id:
            st.info("Επιλέξτε πρώτα μάθημα από την καρτέλα «Μαθήματα».")
        else:
            st.caption(f"Ενεργό Course ID: {selected_course_id}")
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("👥 Εγγεγραμμένοι χρήστες", key="live_users_btn"):
                    try:
                        users = moodle_rest_call(
                            moodle_url, token,
                            "core_enrol_get_enrolled_users",
                            {"courseid": int(selected_course_id)}
                        )
                        st.session_state["moodle_enrolled_users"] = pd.DataFrame(users)
                    except Exception as e:
                        st.error(f"Αποτυχία: {e}")

            with c2:
                if st.button("📝 Quiz", key="live_quizzes_btn"):
                    try:
                        quizzes = moodle_rest_call(
                            moodle_url, token,
                            "mod_quiz_get_quizzes_by_courses",
                            {"courseids[0]": int(selected_course_id)}
                        )
                        qlist = quizzes.get("quizzes", quizzes) if isinstance(quizzes, dict) else quizzes
                        st.session_state["moodle_quizzes"] = pd.DataFrame(qlist)
                    except Exception as e:
                        st.error(f"Αποτυχία: {e}")

            with c3:
                if st.button("🗂️ Δομή μαθήματος", key="live_contents_btn"):
                    try:
                        contents = moodle_rest_call(
                            moodle_url, token,
                            "core_course_get_contents",
                            {"courseid": int(selected_course_id)}
                        )
                        st.session_state["moodle_course_contents"] = contents
                    except Exception as e:
                        st.error(f"Αποτυχία: {e}")

            data_tab1, data_tab2, data_tab3 = st.tabs(["Χρήστες", "Quiz", "Δομή"])

            with data_tab1:
                if "moodle_enrolled_users" in st.session_state:
                    users = st.session_state["moodle_enrolled_users"]
                    safe_cols = [x for x in ["id","fullname","email","lastaccess"] if x in users.columns]
                    st.dataframe(users[safe_cols] if safe_cols else users, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Λήψη χρηστών CSV",
                        users.to_csv(index=False).encode("utf-8-sig"),
                        "moodle_live_users.csv",
                        "text/csv"
                    )
                else:
                    st.caption("Δεν έχουν ανακτηθεί ακόμη χρήστες.")

            with data_tab2:
                if "moodle_quizzes" in st.session_state:
                    qdf = st.session_state["moodle_quizzes"]
                    show_cols = [x for x in ["id","name","course","grade","sumgrades","timeopen","timeclose"] if x in qdf.columns]
                    st.dataframe(qdf[show_cols] if show_cols else qdf, use_container_width=True, hide_index=True)
                else:
                    st.caption("Δεν έχουν ανακτηθεί ακόμη quiz.")

            with data_tab3:
                if "moodle_course_contents" in st.session_state:
                    sections = st.session_state["moodle_course_contents"]
                    flat = []
                    for sec in sections if isinstance(sections, list) else []:
                        sec_name = sec.get("name","")
                        for mod in sec.get("modules",[]):
                            flat.append({
                                "Ενότητα": sec_name,
                                "Όνομα": mod.get("name"),
                                "Τύπος": mod.get("modname"),
                                "ID": mod.get("id"),
                                "URL": mod.get("url")
                            })
                    if flat:
                        st.dataframe(pd.DataFrame(flat), use_container_width=True, hide_index=True)
                    else:
                        st.caption("Δεν βρέθηκε διαθέσιμη δομή.")
                else:
                    st.caption("Δεν έχει ανακτηθεί ακόμη η δομή του μαθήματος.")

    with tab_advanced:
        st.markdown("### Προχωρημένο αίτημα")
        st.caption("Χρησιμοποιήστε το μόνο για ειδικές λειτουργίες Moodle που δεν καλύπτονται από τα έτοιμα εργαλεία.")
        with st.expander("🛠️ Άνοιγμα προχωρημένου αιτήματος", expanded=False):
            custom_function = st.text_input(
                "Λειτουργία διασύνδεσης Moodle",
                placeholder="π.χ. gradereport_user_get_grade_items",
                key="live_custom_function"
            )
            custom_params_text = st.text_area(
                "Παράμετροι ως JSON",
                value="{}",
                height=100,
                key="live_custom_params"
            )
            if st.button("▶️ Εκτέλεση αιτήματος", key="live_custom_execute"):
                if not moodle_url or not token:
                    st.error("Πηγαίνετε πρώτα στην καρτέλα «Σύνδεση».")
                else:
                    try:
                        params = json.loads(custom_params_text or "{}")
                        result = moodle_rest_call(moodle_url, token, custom_function, params)
                        st.session_state["moodle_custom_result"] = result
                        st.success("Το αίτημα ολοκληρώθηκε.")
                        if isinstance(result, list):
                            st.dataframe(pd.DataFrame(result), use_container_width=True, hide_index=True)
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
    render_module_header(
        "Ελεγχόμενη Εγγραφή Moodle",
        "",
        "moodle_write"
    )
    pending_meta = st.session_state.get("pending_moodle_result_meta")
    if isinstance(pending_meta, dict):
        meta_preview = pd.DataFrame([{
            "Λειτουργία": pending_meta.get("service"),
            "Οντότητα": pending_meta.get("entity"),
            "Κατάσταση": pending_meta.get("status"),
            "Βαθμολογία": pending_meta.get("score"),
            "Σύνοψη": pending_meta.get("summary"),
        }])
        st.dataframe(meta_preview, use_container_width=True, hide_index=True)


    moodle_url = st.session_state.get("moodle_base_url", "")
    token = st.session_state.get("moodle_shared_token", "")

    st.markdown("### Σύνδεση Moodle")
    connection_ready = bool(moodle_url and token)

    if connection_ready:
        st.success("Υπάρχουν διαθέσιμα στοιχεία σύνδεσης Moodle για την τρέχουσα συνεδρία.")
    else:
        st.warning(
            "Δεν υπάρχουν ακόμη ενεργά στοιχεία σύνδεσης Moodle. "
            "Συμπληρώστε τα παρακάτω πριν από καταγραφή ή ανάκτηση δεδομένων."
        )

    with st.expander("Ρυθμίσεις σύνδεσης Moodle", expanded=not connection_ready):
        write_url = st.text_input(
            "Βασικό URL Moodle",
            value=moodle_url or "https://contentsolutions.eu",
            key="controlled_moodle_url"
        ).rstrip("/")

        write_token = st.text_input(
            "Κλειδί πρόσβασης Moodle",
            value=token,
            type="password",
            key="controlled_moodle_token",
            help="Χρησιμοποιείται μόνο στην τρέχουσα συνεδρία."
        )

        c_conn1, c_conn2 = st.columns([1, 2])
        with c_conn1:
            if st.button("Έλεγχος σύνδεσης", type="primary", key="controlled_check_connection"):
                if not write_url or not write_token:
                    st.error("Συμπληρώστε URL και κλειδί πρόσβασης.")
                else:
                    try:
                        endpoint = f"{write_url}/webservice/rest/server.php"
                        response = requests.post(
                            endpoint,
                            data={
                                "wstoken": write_token,
                                "wsfunction": "core_webservice_get_site_info",
                                "moodlewsrestformat": "json"
                            },
                            timeout=20
                        )
                        response.raise_for_status()
                        info = response.json()
                        if isinstance(info, dict) and ("exception" in info or "errorcode" in info):
                            raise RuntimeError(info.get("message", str(info)))
                        st.session_state["moodle_base_url"] = write_url
                        st.session_state["moodle_shared_token"] = write_token
                        st.session_state["moodle_connection_verified"] = True
                        moodle_url = write_url
                        token = write_token
                        st.success(
                            f"Η σύνδεση με το Moodle επιβεβαιώθηκε"
                            + (f": {info.get('sitename')}" if isinstance(info, dict) and info.get("sitename") else ".")
                        )
                    except Exception as e:
                        st.session_state["moodle_connection_verified"] = False
                        st.error(f"Η σύνδεση με το Moodle απέτυχε: {e}")

        with c_conn2:
            st.caption(
                "Μετά τον επιτυχή έλεγχο, τα ίδια στοιχεία χρησιμοποιούνται για "
                "καταγραφή παρέμβασης και ανάκτηση σημειώσεων."
            )

    # Επαναφόρτωση μετά από πιθανή επιτυχή σύνδεση.
    moodle_url = st.session_state.get("moodle_base_url", moodle_url)
    token = st.session_state.get("moodle_shared_token", token)

    if "moodle_write_audit" not in st.session_state:
        st.session_state["moodle_write_audit"] = pd.DataFrame(columns=[
            "Ημερομηνία","Ενέργεια","Λειτουργία API","Στόχος",
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
            "Λειτουργία API":function_name,
            "Στόχος":target,
            "Προεπισκόπηση":preview,
            "Αποτέλεσμα":result,
            "Κατάσταση":status
        }])
        st.session_state["moodle_write_audit"] = pd.concat(
            [st.session_state["moodle_write_audit"], row],
            ignore_index=True
        )

    if "moodle_write_action" not in st.session_state:
        st.session_state["moodle_write_action"] = None

    st.markdown("### Επιλέξτε ενέργεια")
    a1, a2, a3, a4 = st.columns([1, 1, 1, 0.7])
    with a1:
        if st.button("📝 Σημείωση", use_container_width=True, key="open_note_action"):
            st.session_state["moodle_write_action"] = "Σημείωση"
    with a2:
        if st.button("💬 Μήνυμα", use_container_width=True, key="open_message_action"):
            st.session_state["moodle_write_action"] = "Μήνυμα"
    with a3:
        if st.button("🎓 Βαθμός", use_container_width=True, key="open_grade_action"):
            st.session_state["moodle_write_action"] = "Βαθμός"
    with a4:
        if st.button("Κλείσιμο", use_container_width=True, key="close_write_action"):
            st.session_state["moodle_write_action"] = None

    action = st.session_state.get("moodle_write_action")
    if action is None:
        st.caption("Επιλέξτε μία ενέργεια για να εμφανιστεί η αντίστοιχη φόρμα.")

    # ------------------------------------------------------
    # EDUCATIONAL NOTE / INTERVENTION
    # ------------------------------------------------------
    if action == "Σημείωση":

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
            note_course_id = st.number_input("ID Μαθήματος", min_value=1, step=1, value=6, key="note_course_manual")

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
            selected_user_label = st.selectbox("Φοιτητής / χρήστης", list(user_labels.keys()), key="note_user")
            note_user_id = user_labels[selected_user_label]
        else:
            note_user_id = st.number_input("ID Χρήστη", min_value=1, step=1, value=208, key="note_user_manual")

        intervention_options = [
            "Μαθησιακή Ανάλυση — Έγκαιρη υποστήριξη / Ανθρώπινος έλεγχος",
            "Ποιότητα Αξιολόγησης — Έλεγχος ποιότητας ερώτησης",
            "Δημιουργία Μαθήματος & Αξιολόγησης — Έλεγχος προσχεδίου",
            "Προσαρμοστική Εξάσκηση — Νέα προσπάθεια",
            "Προσαρμοστική Εξάσκηση — Εμπλουτισμός",
            "Προσαρμοστική Εξάσκηση — Ανθρώπινος έλεγχος",
            "Υποστήριξη Διδάσκοντα — Εγκεκριμένη ανατροφοδότηση",
            "Ενοποιημένες Μαθησιακές Ενδείξεις — Μαθησιακή ένδειξη",
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
        if "controlled_moodle_note_text" not in st.session_state:
            st.session_state["controlled_moodle_note_text"] = pending_note
        elif pending_note and st.session_state.get("controlled_moodle_note_text") != pending_note:
            # Νέο draft από άλλη λειτουργία: το εμφανίζουμε αμέσως και δεν
            # αφήνουμε παλιό περιεχόμενο του widget να το υπερκαλύψει.
            st.session_state["controlled_moodle_note_text"] = pending_note

        note_text = st.text_area(
            "Κείμενο σημείωσης",
            height=120,
            placeholder="Γράψτε τη σημείωση ή την ανατροφοδότηση.",
            key="controlled_moodle_note_text"
        )

        if pending_note:
            st.caption(
                "Το κείμενο έχει προ-συμπληρωθεί από άλλο λειτουργία INELSO Academia. "
                "Ελέγξτε το πριν από οποιαδήποτε πραγματική εγγραφή."
            )

        formatted_note = (
            f"[INELSO Academia] {intervention_type}\n\n"
            f"{note_text.strip()}\n\n"
            f"Καταγραφή μέσω INELSO Academia Πλατφόρμα Έξυπνης Μάθησης."
            + (f"\nΠηγή ανάλυσης: {ACTIVE_NAME}." if ACTIVE_DF is not None else "")
        )

        note_preview = pd.DataFrame([{
            "Ενέργεια":"Καταγραφή Moodle note",
            "ID Μαθήματος":int(note_course_id),
            "ID Χρήστη":int(note_user_id),
            "Τύπος παρέμβασης":intervention_type,
            "Κείμενο":formatted_note,
            "Λειτουργία Moodle":"core_notes_create_notes",
            "Πηγή ανάλυσης": ACTIVE_NAME if ACTIVE_DF is not None else "Μη διαθέσιμο"
        }])

        with st.expander("Προεπισκόπηση", expanded=False):
            st.dataframe(note_preview, use_container_width=True, hide_index=True)

        note_confirm1 = st.checkbox(
            "Έλεγξα το μάθημα, τον εκπαιδευόμενο και το κείμενο.",
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
                st.error("Δεν υπάρχει ενεργή σύνδεση Moodle. Ανοίξτε τις «Ρυθμίσεις σύνδεσης Moodle» παραπάνω και εκτελέστε «Έλεγχος σύνδεσης».")
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
                        st.markdown(
                            "<div class='inelso-action-banner' style='border-left:5px solid #2e9d78'>"
                            "<strong>Η εκπαιδευτική παρέμβαση καταγράφηκε επιτυχώς στο Moodle.</strong>"
                            f"<div style='margin-top:.35rem;color:#52606d'>Αναγνωριστικό σημείωσης: "
                            f"<strong>{note_id}</strong></div></div>",
                            unsafe_allow_html=True
                        )
                        audit_status = "ΕΠΙΤΥΧΙΑ"
                        st.session_state.pop("pending_moodle_note_text", None)
                        st.session_state.pop("pending_moodle_intervention_type", None)
                    else:
                        st.warning(
                            "Το Moodle απάντησε χωρίς σαφές θετικό αναγνωριστικό σημείωσης. "
                            "Ελέγξτε τη λίστα σημειώσεων πριν θεωρηθεί επιτυχής η καταγραφή."
                        )
                        audit_status = "ΠΡΟΣ ΕΛΕΓΧΟ"

                    with st.expander("Τεχνικές λεπτομέρειες απόκρισης Moodle", expanded=False):
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


        if st.button("Ανάκτηση σημειώσεων", key="load_notes"):
            if not moodle_url or not token:
                st.error("Δεν υπάρχει ενεργή σύνδεση Moodle. Ανοίξτε τις «Ρυθμίσεις σύνδεσης Moodle» παραπάνω και εκτελέστε «Έλεγχος σύνδεσης».")
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
                    "courseid":"ID Μαθήματος",
                    "userid":"ID Χρήστη",
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

                latest=notes_list[-1]
                st.markdown(
                    "<div class='inelso-action-banner' style='border-left:5px solid #2e9d78'>"
                    "<strong>Η καταγραφή επαληθεύτηκε από το Moodle.</strong>"
                    "<div style='margin-top:.35rem;color:#52606d'>"
                    "Η σημείωση ανακτήθηκε επιτυχώς για τον επιλεγμένο εκπαιδευόμενο και μάθημα."
                    "</div></div>",
                    unsafe_allow_html=True
                )
                st.session_state.pop("pending_moodle_result_meta", None)

                c1,c2,c3,c4=st.columns(4)
                c1.metric("Note ID",str(latest.get("id","—")))
                c2.metric("ID Μαθήματος",str(latest.get("courseid","—")))
                c3.metric("ID Χρήστη",str(latest.get("userid","—")))
                c4.metric("Κατάσταση","ΕΠΑΛΗΘΕΥΜΕΝΟ")

                with st.expander("Προβολή ανακτημένων σημειώσεων", expanded=True):
                    st.dataframe(clean_notes,use_container_width=True,hide_index=True)
            else:
                st.info("Δεν βρέθηκαν σημειώσεις για τον επιλεγμένο χρήστη/μάθημα.")

    # ------------------------------------------------------
    # MESSAGE
    # ------------------------------------------------------
    elif action == "Μήνυμα":
        st.subheader("Αποστολή προσωπικού μηνύματος")
        st.caption(
            "Επιλέξτε πρώτα μάθημα και πραγματικό εγγεγραμμένο χρήστη. "
            "Η αποστολή πραγματοποιείται μόνο μετά από ρητή επιβεβαίωση."
        )

        message_course_id = st.number_input(
            "ID Μαθήματος",
            min_value=1,
            step=1,
            value=int(st.session_state.get("moodle_selected_course_id", 6) or 6),
            key="message_course_id"
        )

        if st.button(
            "Φόρτωση εγγεγραμμένων χρηστών",
            use_container_width=True,
            key="message_load_course_users"
        ):
            if not moodle_url or not token:
                st.error(
                    "Δεν υπάρχει ενεργή σύνδεση Moodle. Ανοίξτε τις «Ρυθμίσεις σύνδεσης Moodle» "
                    "και εκτελέστε «Έλεγχος σύνδεσης»."
                )
            else:
                try:
                    users_result = moodle_write_call(
                        moodle_url,
                        token,
                        "core_enrol_get_enrolled_users",
                        {"courseid": int(message_course_id)}
                    )
                    if isinstance(users_result, list) and users_result:
                        st.session_state["message_course_users"] = pd.DataFrame(users_result)
                        st.success(f"Φορτώθηκαν {len(users_result)} εγγεγραμμένοι χρήστες.")
                    else:
                        st.session_state.pop("message_course_users", None)
                        st.warning("Δεν επιστράφηκαν εγγεγραμμένοι χρήστες για το επιλεγμένο μάθημα.")
                except Exception as e:
                    st.error(f"Δεν ήταν δυνατή η φόρτωση χρηστών: {e}")

        live_users = st.session_state.get("message_course_users")
        if not isinstance(live_users, pd.DataFrame) or live_users.empty:
            fallback_users = st.session_state.get("moodle_enrolled_users")
            if isinstance(fallback_users, pd.DataFrame) and not fallback_users.empty:
                live_users = fallback_users

        selected_label = None
        if isinstance(live_users, pd.DataFrame) and len(live_users) > 0 and "id" in live_users.columns:
            user_labels = {}
            for _, ur in live_users.iterrows():
                uid = int(ur["id"])
                fullname = str(ur.get("fullname", f"Χρήστης {uid}"))
                email = str(ur.get("email", "") or "")
                label = f"{fullname} — ID {uid}"
                if email and email.lower() != "nan":
                    label += f" — {email}"
                user_labels[label] = uid

            selected_label = st.selectbox(
                "Παραλήπτης",
                list(user_labels.keys()),
                key="message_recipient_select"
            )
            user_id = user_labels[selected_label]
        else:
            st.info(
                "Πατήστε «Φόρτωση εγγεγραμμένων χρηστών» για ασφαλή επιλογή παραλήπτη. "
                "Η χειροκίνητη εισαγωγή ID παραμένει διαθέσιμη μόνο ως εναλλακτική."
            )
            user_id = st.number_input(
                "ID παραλήπτη",
                min_value=1,
                step=1,
                value=1,
                key="message_manual_user_id"
            )

        message_text = st.text_area(
            "Μήνυμα",
            height=160,
            placeholder="Γράψτε την ανατροφοδότηση ή την ειδοποίηση που θα σταλεί.",
            key="controlled_message_text"
        )

        retry_as_contact = st.checkbox(
            "Αν το Moodle μπλοκάρει την αποστολή επειδή ο παραλήπτης δεν είναι επαφή, "
            "προσπάθησε να τον προσθέσεις στις επαφές και να επαναλάβεις μία φορά.",
            value=False,
            key="message_retry_contact"
        )
        preview = pd.DataFrame([{
            "Ενέργεια":"Αποστολή προσωπικού μηνύματος",
            "Μάθημα":int(message_course_id),
            "Παραλήπτης":selected_label if selected_label else f"ID {int(user_id)}",
            "Κείμενο":message_text,
            "Λειτουργία Moodle":"core_message_send_instant_messages"
        }])
        st.subheader("Προεπισκόπηση")
        st.dataframe(preview,use_container_width=True,hide_index=True)

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
                st.error(
                    "Δεν υπάρχει ενεργή σύνδεση Moodle. Ανοίξτε τις «Ρυθμίσεις σύνδεσης Moodle» "
                    "και εκτελέστε «Έλεγχος σύνδεσης»."
                )
            elif not message_text.strip():
                st.error("Το μήνυμα είναι κενό.")
            else:
                function_name = "core_message_send_instant_messages"

                def _send_message_once():
                    return moodle_write_call(
                        moodle_url,
                        token,
                        function_name,
                        {
                            "messages[0][touserid]": int(user_id),
                            "messages[0][text]": message_text,
                            "messages[0][textformat]": 1
                        }
                    )

                def _parse_delivery(result_obj):
                    ok = False
                    msgid = None
                    error_text = ""
                    first = None
                    if isinstance(result_obj, list) and result_obj:
                        first = result_obj[0] if isinstance(result_obj[0], dict) else {}
                    elif isinstance(result_obj, dict):
                        first = result_obj
                    if isinstance(first, dict):
                        msgid = first.get("msgid")
                        error_text = str(first.get("errormessage", "") or "").strip()
                        try:
                            ok = int(msgid) > 0 and not error_text
                        except Exception:
                            ok = False
                    return ok, msgid, error_text

                try:
                    result = _send_message_once()
                    delivery_ok, msgid, delivery_error = _parse_delivery(result)
                    contact_attempted = False
                    contact_result = None

                    contact_block = (
                        (not delivery_ok)
                        and delivery_error
                        and (
                            "contact" in delivery_error.lower()
                            or "επαφ" in delivery_error.lower()
                        )
                    )

                    if contact_block and retry_as_contact:
                        contact_attempted = True
                        try:
                            contact_result = moodle_write_call(
                                moodle_url,
                                token,
                                "core_message_create_contacts",
                                {"userids[0]": int(user_id)}
                            )
                            result = _send_message_once()
                            delivery_ok, msgid, delivery_error = _parse_delivery(result)
                        except Exception as contact_exc:
                            delivery_error = (
                                f"{delivery_error} Η προσπάθεια προσθήκης στις επαφές δεν ολοκληρώθηκε: "
                                f"{contact_exc}"
                            )

                    if delivery_ok:
                        st.markdown(
                            "<div class='inelso-action-banner' style='border-left:5px solid #2e9d78'>"
                            "<strong>Το μήνυμα στάλθηκε επιτυχώς στο Moodle.</strong>"
                            f"<div style='margin-top:.35rem;color:#52606d'>Message ID: "
                            f"<strong>{msgid}</strong></div></div>",
                            unsafe_allow_html=True
                        )
                        if contact_attempted:
                            st.caption("Πριν από την επιτυχή αποστολή έγινε ελεγχόμενη προσπάθεια προσθήκης του παραλήπτη στις επαφές.")
                        audit_status = "ΕΠΙΤΥΧΙΑ"
                    else:
                        if not delivery_error:
                            delivery_error = (
                                "Το Moodle δεν επέστρεψε θετικό αναγνωριστικό μηνύματος."
                            )
                        st.error("Το μήνυμα δεν στάλθηκε.")
                        st.info(delivery_error)
                        if contact_block and not retry_as_contact:
                            st.info(
                                "Το Moodle ζητά σχέση επαφής για τον συγκεκριμένο παραλήπτη. "
                                "Μπορείτε να ενεργοποιήσετε την επιλογή προσθήκης επαφής και να ξαναδοκιμάσετε."
                            )
                        elif contact_attempted:
                            st.warning(
                                "Έγινε προσπάθεια προσθήκης στις επαφές, αλλά το Moodle εξακολουθεί να μην επιτρέπει την αποστολή. "
                                "Δεν παρακάμπτονται οι ρυθμίσεις απορρήτου του παραλήπτη."
                            )
                        audit_status = "ΑΠΟΤΥΧΙΑ"

                    add_write_audit(
                        "Αποστολή προσωπικού μηνύματος",
                        function_name,
                        f"Course {int(message_course_id)} / User ID {int(user_id)}",
                        message_text,
                        json.dumps(result, ensure_ascii=False),
                        audit_status
                    )

                except Exception as e:
                    st.error(f"Η αποστολή απέτυχε τεχνικά: {e}")
                    add_write_audit(
                        "Αποστολή προσωπικού μηνύματος",
                        function_name,
                        f"Course {int(message_course_id)} / User ID {int(user_id)}",
                        message_text,
                        str(e),
                        "ΑΠΟΤΥΧΙΑ"
                    )

    # ------------------------------------------------------
    # GRADE UPDATE
    # ------------------------------------------------------
    elif action == "Βαθμός":
        st.subheader("Ενημέρωση βαθμού — διαχειριστική λειτουργία")
        st.error(
            "Η αλλαγή επίσημου βαθμού είναι υψηλής επίδρασης ενέργεια. "
            "Χρησιμοποιήστε την μόνο σε δοκιμαστικό μάθημα μέχρι να επαληθευτούν πλήρως "
            "τα αναγνωριστικά, η δραστηριότητα και η καταχώριση βαθμού."
        )

        course_id = st.number_input(
            "ID Μαθήματος",
            min_value=1,
            step=1,
            value=int(st.session_state.get("moodle_selected_course_id", 6) or 6),
            key="grade_course_id"
        )

        c_load1, c_load2 = st.columns(2)
        with c_load1:
            if st.button(
                "Φόρτωση εργασιών μαθήματος",
                use_container_width=True,
                key="grade_load_assignments"
            ):
                if not moodle_url or not token:
                    st.error("Δεν υπάρχει ενεργή σύνδεση Moodle.")
                else:
                    assignments = []
                    primary_error = None

                    # 1) Προτιμώμενη μέθοδος: ειδική λειτουργία Assignments.
                    try:
                        assign_result = moodle_write_call(
                            moodle_url,
                            token,
                            "mod_assign_get_assignments",
                            {"courseids[0]": int(course_id)}
                        )
                        if isinstance(assign_result, dict):
                            for course_obj in assign_result.get("courses", []) or []:
                                assignments.extend(course_obj.get("assignments", []) or [])
                    except Exception as e:
                        primary_error = e

                    # 2) Fallback: περιεχόμενα μαθήματος. Χρησιμοποιούμε το instance
                    # της δραστηριότητας ως Assignment ID για το grade API.
                    if not assignments:
                        try:
                            contents = moodle_write_call(
                                moodle_url,
                                token,
                                "core_course_get_contents",
                                {"courseid": int(course_id)}
                            )
                            if isinstance(contents, list):
                                for section in contents:
                                    if not isinstance(section, dict):
                                        continue
                                    for module in section.get("modules", []) or []:
                                        if not isinstance(module, dict):
                                            continue
                                        if str(module.get("modname", "")).lower() == "assign":
                                            instance_id = module.get("instance")
                                            if instance_id is None:
                                                continue
                                            assignments.append({
                                                "id": int(instance_id),
                                                "cmid": module.get("id"),
                                                "name": module.get("name", f"Εργασία {instance_id}")
                                            })
                        except Exception as fallback_error:
                            detail = (
                                f"mod_assign_get_assignments: {primary_error}; "
                                f"core_course_get_contents: {fallback_error}"
                            )
                            st.error(
                                "Δεν ήταν δυνατή η φόρτωση των εργασιών με καμία από τις διαθέσιμες μεθόδους. "
                                f"Λεπτομέρεια: {detail}"
                            )

                    if assignments:
                        # Αποφυγή διπλοεγγραφών αν η ίδια εργασία εμφανιστεί περισσότερες φορές.
                        unique_assignments = {}
                        for a in assignments:
                            if isinstance(a, dict) and a.get("id") is not None:
                                unique_assignments[int(a["id"])] = a
                        assignments = list(unique_assignments.values())
                        st.session_state["grade_assignments"] = assignments
                        st.success(f"Φορτώθηκαν {len(assignments)} εργασίες.")
                        if primary_error is not None:
                            st.caption("Οι εργασίες φορτώθηκαν μέσω των περιεχομένων του μαθήματος.")
                    elif primary_error is None:
                        st.session_state.pop("grade_assignments", None)
                        st.warning("Δεν βρέθηκαν εργασίες τύπου Assignment στο επιλεγμένο μάθημα.")

        with c_load2:
            if st.button(
                "Φόρτωση εκπαιδευομένων",
                use_container_width=True,
                key="grade_load_users"
            ):
                if not moodle_url or not token:
                    st.error("Δεν υπάρχει ενεργή σύνδεση Moodle.")
                else:
                    try:
                        users_result = moodle_write_call(
                            moodle_url,
                            token,
                            "core_enrol_get_enrolled_users",
                            {"courseid": int(course_id)}
                        )
                        if isinstance(users_result, list) and users_result:
                            st.session_state["grade_course_users"] = pd.DataFrame(users_result)
                            st.success(f"Φορτώθηκαν {len(users_result)} εγγεγραμμένοι χρήστες.")
                        else:
                            st.session_state.pop("grade_course_users", None)
                            st.warning("Δεν βρέθηκαν εγγεγραμμένοι χρήστες.")
                    except Exception as e:
                        st.error(f"Δεν ήταν δυνατή η φόρτωση χρηστών: {e}")

        assignments = st.session_state.get("grade_assignments", [])
        selected_assignment_label = None
        if isinstance(assignments, list) and assignments:
            assignment_map = {}
            for a in assignments:
                if not isinstance(a, dict):
                    continue
                aid = a.get("id")
                if aid is None:
                    continue
                name = str(a.get("name", f"Εργασία {aid}"))
                cmid = a.get("cmid")
                label = f"{name} — Assignment ID {aid}"
                if cmid is not None:
                    label += f" — Activity ID {cmid}"
                assignment_map[label] = a
            selected_assignment_label = st.selectbox(
                "Εργασία Moodle",
                list(assignment_map.keys()),
                key="grade_assignment_select"
            )
            selected_assignment = assignment_map[selected_assignment_label]
            component = "mod_assign"
            assignment_instance_id = int(selected_assignment.get("id"))
            cmid_value = selected_assignment.get("cmid")
            if cmid_value is None:
                st.error(
                    "Δεν βρέθηκε Course Module ID για την επιλεγμένη εργασία. "
                    "Δεν είναι ασφαλές να εκτελεστεί αλλαγή βαθμού."
                )
                activity_id = 0
            else:
                activity_id = int(cmid_value)
            st.caption(
                f"Θα χρησιμοποιηθεί η εργασία «{selected_assignment.get('name','')}» · "
                f"Assignment ID {assignment_instance_id} · Activity ID {activity_id}."
            )
        else:
            st.info(
                "Πατήστε «Φόρτωση εργασιών μαθήματος» για να επιλέξετε πραγματική εργασία. "
                "Η χειροκίνητη εισαγωγή παραμένει διαθέσιμη μόνο ως εναλλακτική."
            )
            component = st.text_input(
                "Στοιχείο Moodle",
                value="mod_assign",
                key="grade_component_manual"
            )
            activity_id = st.number_input(
                "ID Δραστηριότητας",
                min_value=1,
                step=1,
                value=1,
                key="grade_activity_manual"
            )

        users_df = st.session_state.get("grade_course_users")
        selected_user_label = None
        if isinstance(users_df, pd.DataFrame) and not users_df.empty and "id" in users_df.columns:
            user_map = {}
            for _, ur in users_df.iterrows():
                uid = int(ur["id"])
                fullname = str(ur.get("fullname", f"Χρήστης {uid}"))
                email = str(ur.get("email", "") or "")
                label = f"{fullname} — ID {uid}"
                if email and email.lower() != "nan":
                    label += f" — {email}"
                user_map[label] = uid
            selected_user_label = st.selectbox(
                "Εκπαιδευόμενος",
                list(user_map.keys()),
                key="grade_user_select"
            )
            student_id = user_map[selected_user_label]
        else:
            student_id = st.number_input(
                "ID Εκπαιδευόμενου",
                min_value=1,
                step=1,
                value=208,
                key="grade_student_manual"
            )

        item_number = 0
        assignment_max_grade = None
        if selected_assignment_label:
            raw_max_grade = selected_assignment.get("grade")
            try:
                if raw_max_grade is not None and float(raw_max_grade) > 0:
                    assignment_max_grade = float(raw_max_grade)
            except Exception:
                assignment_max_grade = None

        if assignment_max_grade is not None:
            st.caption(f"Μέγιστος βαθμός εργασίας: {assignment_max_grade:g}")
            grade_value = st.number_input(
                "Νέος βαθμός",
                min_value=0.0,
                max_value=float(assignment_max_grade),
                value=min(float(st.session_state.get("grade_new_value", 0.0) or 0.0), float(assignment_max_grade)),
                step=0.5,
                key="grade_new_value"
            )
        else:
            grade_value = st.number_input(
                "Νέος βαθμός",
                min_value=0.0,
                value=0.0,
                step=0.5,
                key="grade_new_value"
            )

        submission_ready = False
        submission_attempt = -1
        if selected_assignment_label:
            if st.button(
                "Έλεγχος κατάστασης υποβολής",
                use_container_width=True,
                key="grade_check_submission"
            ):
                if not moodle_url or not token:
                    st.error("Δεν υπάρχει ενεργή σύνδεση Moodle.")
                else:
                    try:
                        submission_status = moodle_write_call(
                            moodle_url,
                            token,
                            "mod_assign_get_submission_status",
                            {
                                "assignid": int(assignment_instance_id),
                                "userid": int(student_id)
                            }
                        )
                        st.session_state["grade_submission_status"] = submission_status
                    except Exception as e:
                        st.session_state.pop("grade_submission_status", None)
                        st.error(
                            "Δεν ήταν δυνατός ο έλεγχος της υποβολής. "
                            "Προσθέστε τη λειτουργία mod_assign_get_submission_status στο external service "
                            f"και ξαναδοκιμάστε. Λεπτομέρεια: {e}"
                        )

            submission_status = st.session_state.get("grade_submission_status")
            if isinstance(submission_status, dict):
                lastattempt = submission_status.get("lastattempt") or {}
                submission = lastattempt.get("submission") or {}
                status_text = str(submission.get("status", "") or "")
                try:
                    submission_attempt = int(submission.get("attemptnumber", -1))
                except Exception:
                    submission_attempt = -1
                if submission:
                    submission_ready = True
                    st.success(
                        f"Βρέθηκε υποβολή για τον εκπαιδευόμενο. "
                        f"Κατάσταση: {status_text or '—'} · Attempt: {submission_attempt}."
                    )
                else:
                    st.warning(
                        "Δεν βρέθηκε υπάρχουσα υποβολή/attempt για τον συγκεκριμένο εκπαιδευόμενο "
                        "στην επιλεγμένη εργασία."
                    )
        feedback = st.text_area(
            "Προαιρετική ανατροφοδότηση",
            height=100,
            key="grade_feedback"
        )
        source = "inelso_control_center"

        grade_preview = pd.DataFrame([{
            "ID Μαθήματος":int(course_id),
            "Εργασία":selected_assignment_label if selected_assignment_label else component,
            "Activity ID":int(activity_id),
            "Assignment ID":int(assignment_instance_id) if selected_assignment_label else "—",
            "ID Εκπαιδευόμενου":int(student_id),
            "Νέος βαθμός":float(grade_value),
            "Ανατροφοδότηση":feedback
        }])

        st.subheader("Προεπισκόπηση αλλαγής")
        st.dataframe(grade_preview,use_container_width=True,hide_index=True)

        typed_phrase = st.text_input(
            "Για επιβεβαίωση πληκτρολογήστε ακριβώς: ΑΛΛΑΓΗ ΒΑΘΜΟΥ",
            key="grade_phrase"
        )
        grade_confirm = st.checkbox(
            "Έλεγξα το μάθημα, την εργασία, τον εκπαιδευόμενο και τον νέο βαθμό.",
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
            and float(grade_value) >= 0
            and int(activity_id) > 0
            and (
                (not selected_assignment_label)
                or submission_ready
            )
            and (
                assignment_max_grade is None
                or float(grade_value) <= float(assignment_max_grade)
            )
        )

        if st.button(
            "ΕΚΤΕΛΕΣΗ ΑΛΛΑΓΗΣ ΒΑΘΜΟΥ",
            disabled=not can_submit,
            type="primary"
        ):
            if not moodle_url or not token:
                st.error("Δεν υπάρχει ενεργή σύνδεση Moodle.")
            else:
                # Για Moodle Assignment χρησιμοποιούμε πρώτα το ειδικό API βαθμολόγησης.
                # Αυτό δουλεύει με Assignment ID (instance id), όχι με Course Module ID.
                function_name = "mod_assign_save_grade"
                params = {
                    "assignmentid": int(assignment_instance_id) if selected_assignment_label else int(activity_id),
                    "userid": int(student_id),
                    "grade": float(grade_value),
                    "attemptnumber": int(submission_attempt) if submission_ready else -1,
                    "addattempt": 0,
                    "workflowstate": "",
                    "applytoall": 0
                }
                if feedback.strip():
                    params["plugindata[assignfeedbackcomments_editor][text]"] = feedback
                    params["plugindata[assignfeedbackcomments_editor][format]"] = 1

                preview_text = json.dumps(params, ensure_ascii=False)

                try:
                    result = moodle_write_call(
                        moodle_url, token, function_name, params
                    )
                    st.markdown(
                        "<div class='inelso-action-banner' style='border-left:5px solid #2e9d78'>"
                        "<strong>Η ενημέρωση βαθμού ολοκληρώθηκε χωρίς σφάλμα από το Moodle.</strong>"
                        "<div style='margin-top:.35rem;color:#52606d'>Ελέγξτε το Βαθμολόγιο του μαθήματος για επιβεβαίωση.</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                    add_write_audit(
                        "Ενημέρωση βαθμού",
                        function_name,
                        f"Course {course_id} / Assignment {params['assignmentid']} / Student {student_id}",
                        preview_text,
                        json.dumps(result, ensure_ascii=False),
                        "ΕΠΙΤΥΧΙΑ"
                    )
                except Exception as primary_error:
                    # Fallback στο γενικό grade API μόνο αν το ειδικό API δεν είναι διαθέσιμο.
                    fallback_name = "core_grade_update_grades"
                    fallback_params = {
                        "source":source,
                        "courseid":int(course_id),
                        "component":component,
                        "activityid":int(activity_id),
                        "itemnumber":int(item_number),
                        "grades[0][studentid]":int(student_id),
                        "grades[0][grade]":float(grade_value)
                    }
                    if feedback.strip():
                        fallback_params["grades[0][str_feedback]"] = feedback

                    try:
                        result = moodle_write_call(
                            moodle_url, token, fallback_name, fallback_params
                        )
                        st.markdown(
                            "<div class='inelso-action-banner' style='border-left:5px solid #2e9d78'>"
                            "<strong>Η ενημέρωση βαθμού ολοκληρώθηκε μέσω της εναλλακτικής λειτουργίας Moodle.</strong>"
                            "<div style='margin-top:.35rem;color:#52606d'>Ελέγξτε το Βαθμολόγιο του μαθήματος για επιβεβαίωση.</div>"
                            "</div>",
                            unsafe_allow_html=True
                        )
                        add_write_audit(
                            "Ενημέρωση βαθμού",
                            fallback_name,
                            f"Course {course_id} / Activity {activity_id} / Student {student_id}",
                            json.dumps(fallback_params, ensure_ascii=False),
                            json.dumps(result, ensure_ascii=False),
                            "ΕΠΙΤΥΧΙΑ"
                        )
                    except Exception as fallback_error:
                        st.error(
                            "Η ενημέρωση βαθμού απέτυχε και με τις δύο διαθέσιμες μεθόδους. "
                            f"mod_assign_save_grade: {primary_error} · "
                            f"core_grade_update_grades: {fallback_error}"
                        )
                        add_write_audit(
                            "Ενημέρωση βαθμού",
                            function_name,
                            f"Course {course_id} / Assignment {params['assignmentid']} / Student {student_id}",
                            preview_text,
                            f"Primary: {primary_error}; Fallback: {fallback_error}",
                            "ΑΠΟΤΥΧΙΑ"
                        )

    # ------------------------------------------------------
    # AUDIT LOG
    # ------------------------------------------------------
    st.markdown("---")
    audit = st.session_state["moodle_write_audit"]

    if len(audit) > 0:
        st.dataframe(audit,use_container_width=True,hide_index=True)
        st.download_button(
            "Λήψη audit log CSV",
            audit.to_csv(index=False).encode("utf-8-sig"),
            "INELSO Academia_Moodle_Write_Audit.csv",
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
                "Το τοπικό session audit log δεν περιέχει νέα ενέργεια καταγραφής στην παρούσα συνεδρία."
            )
        else:
            pass

# ==========================================================
# EARLY WARNING ENGINE
# ==========================================================
elif section == "Μαθησιακή Ανάλυση & Έγκαιρη Υποστήριξη":

    def _s16_go(target_label):
        st.session_state["inelso_main_navigation"] = target_label

    render_module_header(
        "Μαθησιακή Ανάλυση & Έγκαιρη Υποστήριξη",
        "",
        "s16"
    )
    show_active_dataset_context("S16")

    if ACTIVE_DF is None:
        no_data_tabs = st.tabs(["Τι κάνει", "Τι χρειάζεται"])
        with no_data_tabs[0]:
            a,b,c1,c2 = st.columns(4)
            with a:
                st.markdown(
                    "<div class='inelso-feature-card'><h3>Παρακολουθεί</h3>"
                    "<p>Επίδοση, αδράνεια, προσπάθειες και συμμετοχή, μόνο όταν υπάρχουν πραγματικά πεδία.</p></div>",
                    unsafe_allow_html=True
                )
            with b:
                st.markdown(
                    "<div class='inelso-feature-card'><h3>Ιεραρχεί</h3>"
                    "<p>Συνδυάζει διαθέσιμες ενδείξεις σε διαφανή λειτουργικό δείκτη προτεραιότητας.</p></div>",
                    unsafe_allow_html=True
                )
            with c1:
                st.markdown(
                    "<div class='inelso-feature-card'><h3>Εξηγεί</h3>"
                    "<p>Δείχνει ποιες μεταβλητές και ποια βάρη συνέβαλαν στην τελική ένδειξη.</p></div>",
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    "<div class='inelso-feature-card'><h3>Προτείνει</h3>"
                    "<p>Παράγει επόμενο βήμα για ανθρώπινο έλεγχο, επικοινωνία ή απλή παρακολούθηση.</p></div>",
                    unsafe_allow_html=True
                )

            st.markdown("#### Τι θα εμφανιστεί όταν υπάρχουν δεδομένα")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Φοιτητές", "—")
            k2.metric("Υψηλή προτεραιότητα", "—")
            k3.metric("Μεσαία προτεραιότητα", "—")
            k4.metric("Κατάσταση", "Αναμονή δεδομένων")

        with no_data_tabs[1]:
            st.markdown("### Ελάχιστα χρήσιμα πεδία")
            st.markdown(
                "<div class='inelso-process'>"
                "<div class='step active'><strong>1 · Αναγνωριστικό μαθητή</strong><span>Ανώνυμο ή εσωτερικό αναγνωριστικό μαθητή.</span></div>"
                "<div class='step'><strong>2 · Επίδοση</strong><span>Βαθμολογία ή επίδοση, εφόσον είναι διαθέσιμη.</span></div>"
                "<div class='step'><strong>3 · Συμμετοχή</strong><span>Αδράνεια, υποβολές ή δραστηριότητα/συμμετοχή.</span></div>"
                "<div class='step safe'><strong>4 · Ανθρώπινος έλεγχος</strong><span>Η ένδειξη δεν εκτελεί καμία αυτόματη παρέμβαση.</span></div>"
                "</div>",
                unsafe_allow_html=True
            )
            x1,x2 = st.columns(2)
            with x1:
                st.button(
                    "Εισαγωγή CSV / Excel",
                    use_container_width=True,
                    on_click=_s16_go,
                    args=("Εισαγωγή Δεδομένων",),
                    key="s16_empty_import"
                )
            with x2:
                st.button(
                    "Ανάλυση SCORM",
                    use_container_width=True,
                    on_click=_s16_go,
                    args=("Ανάλυση SCORM",),
                    key="s16_empty_scorm"
                )

        st.stop()

    # ------------------------------------------------------
    # ACTIVE DATASET MODE
    # ------------------------------------------------------
    ew_df = st.session_state.get(
        "inelso_uploaded_clean",
        st.session_state.get("inelso_uploaded_original", ACTIVE_DF)
    ).copy()

    # Αυτόματος αρχικός υπολογισμός για unified datasets.
    # Χρησιμοποιεί μόνο πραγματικά διαθέσιμα πεδία και μία εγγραφή ανά εκπαιδευόμενο.
    _auto_source = ACTIVE_NAME
    if st.session_state.get("early_warning_auto_source") != _auto_source:
        _learner_col = detect_column(ew_df, ["learner_id", "μαθητής", "learner", "user_id"])
        _grade_col = detect_column(ew_df, ["score", "final_score", "grade", "final score"])
        _inactivity_col = detect_column(ew_df, ["inactivity_days", "ημέρες αδράνειας", "inactivity"])
        _attempts_col = detect_column(ew_df, ["submissions", "attempt", "προσπάθεια"])
        _engagement_col = detect_column(ew_df, ["engagement_score", "engagement", "activity score"])

        _available = [
            x for x in [_grade_col, _inactivity_col, _attempts_col, _engagement_col]
            if x is not None
        ]

        if _learner_col is not None and _available:
            _auto_df = ew_df.copy()

            # Long-format unified CSV: μία γραμμή ανά learner για να μην μετριέται
            # ο ίδιος εκπαιδευόμενος πολλές φορές λόγω ερωτήσεων.
            if _learner_col in _auto_df.columns:
                _auto_df = _auto_df.drop_duplicates(subset=[_learner_col], keep="last").copy()

            def _norm_low(series):
                s = pd.to_numeric(series, errors="coerce")
                if s.notna().sum() == 0:
                    return pd.Series(0.0, index=series.index)
                lo, hi = s.min(), s.max()
                if hi == lo:
                    return pd.Series(0.0, index=series.index)
                return 100 * (hi - s) / (hi - lo)

            def _norm_high(series):
                s = pd.to_numeric(series, errors="coerce")
                if s.notna().sum() == 0:
                    return pd.Series(0.0, index=series.index)
                lo, hi = s.min(), s.max()
                if hi == lo:
                    return pd.Series(0.0, index=series.index)
                return 100 * (s - lo) / (hi - lo)

            _components = []
            _weights = []
            _signals = []

            if _grade_col is not None:
                _components.append(_norm_low(_auto_df[_grade_col])); _weights.append(35); _signals.append("επίδοση")
            if _inactivity_col is not None:
                _components.append(_norm_high(_auto_df[_inactivity_col])); _weights.append(30); _signals.append("αδράνεια")
            if _attempts_col is not None:
                _components.append(_norm_high(_auto_df[_attempts_col])); _weights.append(15); _signals.append("προσπάθειες")
            if _engagement_col is not None:
                _components.append(_norm_low(_auto_df[_engagement_col])); _weights.append(20); _signals.append("συμμετοχή")

            if _components:
                _total_w = max(1, sum(_weights))
                _risk = sum(comp*w for comp, w in zip(_components, _weights)) / _total_w
                _auto_df["Δείκτης κινδύνου"] = _risk.round(1)
                _auto_df["Κατηγορία κινδύνου"] = _auto_df["Δείκτης κινδύνου"].apply(
                    lambda x: "ΥΨΗΛΟΣ" if x >= 70 else ("ΜΕΣΑΙΟΣ" if x >= 45 else "ΧΑΜΗΛΟΣ")
                )
                _auto_df["Μαθητής"] = _auto_df[_learner_col].astype(str)
                _auto_df["Προτεινόμενη ενέργεια"] = _auto_df["Κατηγορία κινδύνου"].map({
                    "ΥΨΗΛΟΣ": "Άμεσος ανθρώπινος έλεγχος",
                    "ΜΕΣΑΙΟΣ": "Στοχευμένη επικοινωνία / παρακολούθηση",
                    "ΧΑΜΗΛΟΣ": "Κανονική παρακολούθηση"
                })
                _auto_df["Εξήγηση"] = "Αυτόματος αρχικός δείκτης από: " + ", ".join(_signals) + "."
                st.session_state["early_warning_results"] = _auto_df
                st.session_state["early_warning_config"] = {
                    "signals": _signals,
                    "high_thr": 70,
                    "med_thr": 45,
                    "weights": dict(zip(_signals, _weights)),
                    "automatic": True
                }
                st.session_state["early_warning_auto_source"] = _auto_source
        else:
            st.session_state.pop("early_warning_results", None)
            st.session_state.pop("early_warning_config", None)
            st.session_state["early_warning_auto_source"] = _auto_source

    s16_tabs = st.tabs(["Επισκόπηση", "Ρύθμιση δείκτη", "Αποτελέσματα", "Παρέμβαση & Moodle"])

    with s16_tabs[0]:
        compat = service_dataset_compatibility("S16", ACTIVE_DF)
        p = dataset_profile(ACTIVE_DF)
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Εγγραφές", p["rows"])
        r2.metric("Πεδία", p["cols"])
        r3.metric("Συμβατότητα", compat)
        if "early_warning_results" in st.session_state:
            _ew_current = st.session_state["early_warning_results"]
            _high_n = int((_ew_current["Κατηγορία κινδύνου"]=="ΥΨΗΛΟΣ").sum()) if "Κατηγορία κινδύνου" in _ew_current.columns else 0
            r4.metric("Υψηλή προτεραιότητα", _high_n)
        else:
            r4.metric("Υψηλή προτεραιότητα", "—")

        st.markdown("#### Τι χρησιμοποιεί η λειτουργία")
        u1,u2,u3,u4 = st.columns(4)
        with u1:
            st.markdown("<div class='inelso-feature-card'><h3>Επίδοση</h3><p>Η χαμηλότερη επίδοση μπορεί να αυξήσει την προτεραιότητα, ιδιαίτερα όταν συνδυάζεται με άλλες ενδείξεις μαθησιακής δυσκολίας.</p></div>", unsafe_allow_html=True)
        with u2:
            st.markdown("<div class='inelso-feature-card'><h3>Αδράνεια</h3><p>Οι περισσότερες ημέρες αδράνειας μπορεί να υποδηλώνουν μειωμένη εμπλοκή και να αυξάνουν την προτεραιότητα για έγκαιρο έλεγχο.</p></div>", unsafe_allow_html=True)
        with u3:
            st.markdown("<div class='inelso-feature-card'><h3>Προσπάθειες</h3><p>Ο αριθμός προσπαθειών μπορεί να χρησιμοποιηθεί ως συμπληρωματική ένδειξη, ανάλογα με τη δομή και το περιεχόμενο του συνόλου δεδομένων.</p></div>", unsafe_allow_html=True)
        with u4:
            st.markdown("<div class='inelso-feature-card'><h3>Συμμετοχή</h3><p>Η χαμηλή συμμετοχή μπορεί να λειτουργήσει ως πρόσθετη ένδειξη περιορισμένης εμπλοκής και να ενισχύσει τη συνολική προτεραιότητα.</p></div>", unsafe_allow_html=True)

        with st.expander("Μεθοδολογική σημείωση", expanded=False):
            st.write(
                "Ο δείκτης είναι παραμετρικός και διαφανής. Δεν παρουσιάζεται ως επικυρωμένο predictive model "
                "χωρίς ανεξάρτητη αξιολόγηση. Τα βάρη και τα κατώφλια ορίζονται ρητά από τον χρήστη."
            )

    with s16_tabs[1]:
        st.markdown("### 1. Αντιστοίχιση πεδίων")
        cols = ["— καμία —"] + list(ew_df.columns.astype(str))
        m1,m2 = st.columns(2)
        with m1:
            learner_col = st.selectbox("Αναγνωριστικό μαθητή", cols, key="ew_learner")
            grade_col = st.selectbox("Βαθμός / επίδοση", cols, key="ew_grade")
            inactivity_col = st.selectbox("Ημέρες αδράνειας", cols, key="ew_inactivity")
        with m2:
            attempts_col = st.selectbox("Αριθμός προσπαθειών", cols, key="ew_attempts")
            engagement_col = st.selectbox("Δείκτης συμμετοχής / δραστηριότητας", cols, key="ew_engagement")

        st.markdown("### 2. Βάρη")
        w1,w2,w3,w4 = st.columns(4)
        with w1:
            w_grade = st.slider("Χαμηλή επίδοση",0,100,35,5,key="s16_w_grade")
        with w2:
            w_inactivity = st.slider("Αδράνεια",0,100,30,5,key="s16_w_inactivity")
        with w3:
            w_attempts = st.slider("Προσπάθειες",0,100,15,5,key="s16_w_attempts")
        with w4:
            w_engagement = st.slider("Χαμηλή συμμετοχή",0,100,20,5,key="s16_w_engagement")

        st.markdown("### 3. Όρια κατηγοριοποίησης προτεραιότητας")
        med_thr, high_thr = st.slider(
            "Όρια Μεσαίας και Υψηλής προτεραιότητας",
            min_value=0,
            max_value=100,
            value=(45, 75),
            step=5,
            key="s16_priority_thresholds"
        )
        st.caption(
            f"Χαμηλή: 0–{med_thr-1} · Μεσαία: {med_thr}–{high_thr-1} · Υψηλή: {high_thr}–100"
        )
        st.caption("Τα βάρη κανονικοποιούνται μόνο στα πεδία που έχουν επιλεγεί.")

        if st.button("Υπολογισμός προτεραιότητας", type="primary", use_container_width=True, key="s16_compute"):
            out_df = ew_df.copy()

            # Σε long-format σύνολα δεδομένων ο ίδιος εκπαιδευόμενος μπορεί να
            # εμφανίζεται πολλές φορές (π.χ. ανά ερώτηση ή δραστηριότητα).
            # Για την Έγκαιρη Υποστήριξη κρατάμε μία εγγραφή ανά εκπαιδευόμενο,
            # όπως και στον αυτόματο αρχικό υπολογισμό.
            if learner_col != "— καμία —" and learner_col in out_df.columns:
                out_df = out_df.drop_duplicates(subset=[learner_col], keep="last").copy()

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
            active_signals = []

            if grade_col != "— καμία —":
                components.append(norm_low(out_df[grade_col])); weights.append(w_grade); active_signals.append("επίδοση")
            if inactivity_col != "— καμία —":
                components.append(norm_high(out_df[inactivity_col])); weights.append(w_inactivity); active_signals.append("αδράνεια")
            if attempts_col != "— καμία —":
                components.append(norm_high(out_df[attempts_col])); weights.append(w_attempts); active_signals.append("προσπάθειες")
            if engagement_col != "— καμία —":
                components.append(norm_low(out_df[engagement_col])); weights.append(w_engagement); active_signals.append("συμμετοχή")

            if not components:
                st.error("Επίλεξε τουλάχιστον μία πραγματική ένδειξη.")
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
                out_df["Μαθητής"] = (
                    out_df[learner_col].astype(str)
                    if learner_col != "— καμία —"
                    else [f"Εγγραφή {i+1}" for i in range(len(out_df))]
                )
                out_df["Προτεινόμενη ενέργεια"] = out_df["Κατηγορία κινδύνου"].map({
                    "ΥΨΗΛΟΣ":"Άμεσος ανθρώπινος έλεγχος",
                    "ΜΕΣΑΙΟΣ":"Στοχευμένη επικοινωνία / παρακολούθηση",
                    "ΧΑΜΗΛΟΣ":"Κανονική παρακολούθηση"
                })
                out_df["Εξήγηση"] = (
                    "Risk score από: " + ", ".join(active_signals) +
                    ". Βάρη και κατώφλια ορίζονται από τον χρήστη."
                )

                st.session_state["early_warning_results"] = out_df
                st.session_state["early_warning_config"] = {
                    "signals": active_signals,
                    "high_thr": high_thr,
                    "med_thr": med_thr,
                    "weights": {
                        "grade":w_grade,
                        "inactivity":w_inactivity,
                        "attempts":w_attempts,
                        "engagement":w_engagement,
                    }
                }
                st.success("Η ανάλυση ολοκληρώθηκε. Τα αποτελέσματα είναι διαθέσιμα στο tab «Αποτελέσματα».")

    with s16_tabs[2]:
        if "early_warning_results" not in st.session_state:
            render_empty_state(
                "Δεν έχει εκτελεστεί ακόμη ανάλυση",
                "Ρυθμίστε τις πραγματικές μεταβλητές και τα κατώφλια στο tab «Ρύθμιση δείκτη».",
                "Πατήστε «Υπολογισμός προτεραιότητας»."
            )
        else:
            out_df = st.session_state["early_warning_results"]
            counts = out_df["Κατηγορία κινδύνου"].value_counts()
            a,b,c3,d = st.columns(4)
            a.metric("Σύνολο", len(out_df))
            b.metric("Υψηλή", int(counts.get("ΥΨΗΛΟΣ",0)))
            c3.metric("Μεσαία", int(counts.get("ΜΕΣΑΙΟΣ",0)))
            d.metric("Χαμηλή", int(counts.get("ΧΑΜΗΛΟΣ",0)))

            ranked = out_df.sort_values("Δείκτης κινδύνου",ascending=False)
            top = ranked.iloc[0] if len(ranked) else None
            if top is not None:
                st.markdown(
                    f"<div class='inelso-action-banner'><strong>Υψηλότερη τρέχουσα προτεραιότητα:</strong> "
                    f"{top['Μαθητής']} · δείκτης {top['Δείκτης κινδύνου']}/100 · {top['Κατηγορία κινδύνου']}.</div>",
                    unsafe_allow_html=True
                )

            view_cols = ["Μαθητής","Δείκτης κινδύνου","Κατηγορία κινδύνου","Προτεινόμενη ενέργεια"]
            st.dataframe(ranked[view_cols], use_container_width=True, hide_index=True)

            st.download_button(
                "Λήψη αποτελεσμάτων CSV",
                out_df.to_csv(index=False).encode("utf-8-sig"),
                "INELSO Academia_Early_Warning.csv",
                "text/csv",
                use_container_width=True,
                key="s16_download_results"
            )

    with s16_tabs[3]:
        if "early_warning_results" not in st.session_state:
            render_empty_state(
                "Δεν υπάρχει ακόμη περίπτωση για παρέμβαση",
                "Η παρέμβαση εμφανίζεται μόνο αφού ολοκληρωθεί πραγματική ανάλυση.",
                "Εκτελέστε πρώτα τον υπολογισμό προτεραιότητας."
            )
        else:
            ranked = st.session_state["early_warning_results"].sort_values("Δείκτης κινδύνου",ascending=False)
            st.markdown("### Παρέμβαση με ανθρώπινο έλεγχο")
            selected_risk_learner = st.selectbox(
                "Εκπαιδευόμενος",
                ranked["Μαθητής"].astype(str).drop_duplicates().tolist(),
                key="s16_intervention_learner"
            )
            selected_row = ranked[
                ranked["Μαθητής"].astype(str) == str(selected_risk_learner)
            ].iloc[0]

            p1,p2,p3 = st.columns(3)
            p1.metric("Δείκτης", f"{selected_row['Δείκτης κινδύνου']}/100")
            p2.metric("Προτεραιότητα", selected_row["Κατηγορία κινδύνου"])
            with p3:
                st.markdown(
                    "<div class='inelso-feature-card' style='min-height:auto;padding:.75rem .9rem'>"
                    "<div class='kicker'>Επόμενο βήμα</div>"
                    f"<div style='font-size:1.02rem;font-weight:700;color:#173b5e;line-height:1.3;white-space:normal'>"
                    f"{selected_row['Προτεινόμενη ενέργεια']}</div></div>",
                    unsafe_allow_html=True
                )

            default_s16_note = (
                f"Ο εκπαιδευόμενος {selected_risk_learner} εμφανίζει "
                f"{str(selected_row['Κατηγορία κινδύνου']).lower()} προτεραιότητα υποστήριξης "
                f"με δείκτη {selected_row['Δείκτης κινδύνου']}/100. "
                "Συνιστάται έλεγχος της πρόσφατης επίδοσης, της συμμετοχής και της δραστηριότητάς του "
                "πριν αποφασιστεί οποιαδήποτε παρέμβαση."
            )

            with st.expander("Κείμενο παρέμβασης", expanded=True):
                note_key = f"s16_note_preview_{str(selected_risk_learner)}"
                if note_key not in st.session_state:
                    st.session_state[note_key] = default_s16_note
                else:
                    # Κάθε εκπαιδευόμενος διατηρεί το δικό του προσχέδιο,
                    # αλλά η αρχική τιμή προέρχεται πάντα από τον τρέχοντα υπολογισμό.
                    existing_note = str(st.session_state.get(note_key, ""))
                    # Αν υπάρχει παλιό τεχνικό draft από προηγούμενη έκδοση,
                    # αντικαθίσταται μία φορά με το νέο καθαρό ελληνικό πρότυπο.
                    if (
                        "early-warning review" in existing_note
                        or "Δείκτης κινδύνου:" in existing_note
                    ):
                        st.session_state[note_key] = default_s16_note

                s16_note_text = st.text_area(
                    "Προσχέδιο προς έλεγχο",
                    height=130,
                    key=note_key
                )

            st.info("Το παρακάτω βήμα δημιουργεί μόνο draft. Δεν γράφει αυτόματα βαθμό ή παρέμβαση στο Moodle.")
            if st.button("Προώθηση draft στην Ελεγχόμενη Εγγραφή Moodle", use_container_width=True, key="s16_stage_note"):
                # Μεταφέρουμε ακριβώς το τρέχον, ορατό κείμενο του editor.
                st.session_state["pending_moodle_note_text"] = str(
                    st.session_state.get(note_key, s16_note_text)
                )
                st.session_state["pending_moodle_intervention_type"] = "Μαθησιακή Ανάλυση — Έγκαιρη υποστήριξη / Ανθρώπινος έλεγχος"
                # Καθαρίζουμε τυχόν παλιά widget state της φόρμας Moodle ώστε
                # το νέο draft να εμφανιστεί ακριβώς όπως προωθήθηκε.
                st.session_state.pop("controlled_moodle_note_text", None)
                st.success("Το draft προωθήθηκε στην Ελεγχόμενη Εγγραφή Moodle. Απαιτείται ρητή ανθρώπινη επιβεβαίωση.")

# ==========================================================
# ΜΗΤΡΩΟ ΕΝΕΡΓΕΙΩΝ
# ==========================================================
elif section == "Μητρώο Ενεργειών":
    render_module_header("Μητρώο Ενεργειών", "", "registry")
    show_active_dataset_context("REPORT")

    base_columns = [
        "Ημερομηνία",
        "Ενότητα",
        "Οντότητα",
        "Ένδειξη",
        "Προτεραιότητα",
        "Απόφαση",
        "Υπεύθυνος",
        "Κατάσταση",
        "Σημειώσεις",
        "Πηγή δεδομένων"
    ]

    if "action_log" not in st.session_state:
        st.session_state["action_log"] = pd.DataFrame(columns=base_columns)

    _registry = st.session_state["action_log"]
    r1, r2, r3 = st.columns(3)
    r1.metric("Συνολικές ενέργειες", len(_registry))
    r2.metric(
        "Ανοικτές",
        int(_registry["Κατάσταση"].isin(["ΝΕΑ","ΣΕ ΕΛΕΓΧΟ"]).sum()) if len(_registry) else 0
    )
    r3.metric(
        "Ολοκληρωμένες",
        int((_registry["Κατάσταση"]=="ΟΛΟΚΛΗΡΩΘΗΚΕ").sum()) if len(_registry) else 0
    )

    with st.expander("＋ Νέα ενέργεια", expanded=False):
        c1,c2 = st.columns(2)

        with c1:
            log_service = st.selectbox(
                "Ενότητα",
                ["Μαθησιακή Ανάλυση","Ποιότητα Αξιολόγησης","Δημιουργία Μαθήματος & Αξιολόγησης","Προσαρμοστική Εξάσκηση","Υποστήριξη Διδάσκοντα","Ενοποιημένες Μαθησιακές Ενδείξεις","Άλλο"],
                key="log_service"
            )
            log_entity = st.text_input("Οντότητα / μαθητής / ερώτηση",key="log_entity")
            log_ένδειξη = st.text_input("Ένδειξη / αιτία",key="log_ένδειξη")
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

        if st.button("＋ Καταχώριση ενέργειας", type="primary"):
            new_row = pd.DataFrame([{
                "Ημερομηνία":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ενότητα":log_service,
                "Οντότητα":log_entity,
                "Ένδειξη":log_ένδειξη,
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

    with st.expander("📋 Καταχωρίσεις", expanded=False):
        with st.expander("📥 Εισαγωγή υπάρχοντος μητρώου", expanded=False):
            imported_log = st.file_uploader(
                "Αρχείο CSV μητρώου",
                type=["csv"],
                key="action_log_import"
            )
            if imported_log is not None:
                try:
                    imported_df = pd.read_csv(imported_log)
                    missing = [c for c in base_columns if c not in imported_df.columns]
                    if missing:
                        for c in missing:
                            imported_df[c] = ""
                    st.session_state["action_log"] = imported_df[base_columns]
                    st.success("Το μητρώο εισήχθη επιτυχώς.")
                except Exception as e:
                    st.error(f"Αποτυχία εισαγωγής: {e}")

        logdf = st.session_state["action_log"]

        if len(logdf) > 0:
            st.dataframe(logdf,use_container_width=True,hide_index=True)

            status_counts = logdf["Κατάσταση"].value_counts()
            c1,c2,c3 = st.columns(3)
            c1.metric("Σύνολο",len(logdf))
            c2.metric("Ανοικτές",int(status_counts.get("ΝΕΑ",0)+status_counts.get("ΣΕ ΕΛΕΓΧΟ",0)))
            c3.metric("Ολοκληρωμένες",int(status_counts.get("ΟΛΟΚΛΗΡΩΘΗΚΕ",0)))

            st.download_button(
                "Λήψη μητρώου CSV",
                logdf.to_csv(index=False).encode("utf-8-sig"),
                "INELSO Academia_Action_Log.csv",
                "text/csv"
            )
        else:
            st.info("Δεν υπάρχουν ακόμη καταχωρισμένες ενέργειες.")

# ==========================================================
# ΕΙΣΑΓΩΓΗ ΔΕΔΟΜΕΝΩΝ
# ==========================================================
elif section == "Εισαγωγή CSV / Excel":
    render_module_header(
        "Εισαγωγή Δεδομένων",
        "",
        "csv_import"
    )

    st.markdown("### Αρχείο δεδομένων")
    uploaded_file = st.file_uploader(
        "Επιλέξτε αρχείο CSV ή Excel",
        type=["csv", "xlsx", "xls"],
        key="main_data_file_uploader"
    )

    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            lower_name = file_name.lower()

            if lower_name.endswith(".csv"):
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    uploaded_df = pd.read_csv(uploaded_file, encoding="latin-1")
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            if uploaded_df.empty:
                st.error("Το αρχείο δεν περιέχει εγγραφές.")
            else:
                st.session_state["inelso_uploaded_original"] = uploaded_df.copy()
                st.session_state["inelso_uploaded_clean"] = uploaded_df.copy()
                st.session_state["inelso_uploaded_name"] = file_name
                mapped = auto_map_unified_services(uploaded_df, file_name)
                st.success(
                    f"Το αρχείο «{file_name}» φορτώθηκε επιτυχώς "
                    f"({len(uploaded_df)} εγγραφές × {len(uploaded_df.columns)} πεδία)."
                )
                st.rerun()
        except Exception as e:
            st.error(f"Δεν ήταν δυνατή η ανάγνωση του αρχείου: {e}")

    if ACTIVE_DF is None:
        st.info("Δεν υπάρχει ακόμη ενεργό αρχείο.")
    else:
        p = dataset_profile(ACTIVE_DF)
        q = dataset_quality_metrics(ACTIVE_DF)

        st.markdown("### Ενεργό αρχείο")
        st.write(f"**{ACTIVE_NAME}**")

        a,b,d,e = st.columns(4)
        a.metric("Εγγραφές", p["rows"])
        b.metric("Πεδία", p["cols"])
        d.metric("Κενές τιμές", p["missing"])
        e.metric("Διπλότυπα", p["duplicates"])

        if q:
            st.metric("Ποιότητα δεδομένων", f"{q['quality']:.1f}/100")

        with st.expander("🔎 Αυτόματη αναγνώριση συμβατότητας", expanded=False):
            compatibility = pd.DataFrame([
                {
                    "Λειτουργία":"Έγκαιρη Υποστήριξη",
                    "Συμβατότητα":service_dataset_compatibility("S16", ACTIVE_DF)
                },
                {
                    "Λειτουργία":"Ποιότητα Αξιολόγησης",
                    "Συμβατότητα":service_dataset_compatibility("S21", ACTIVE_DF)
                },
                {
                    "Λειτουργία":"Δημιουργία Αξιολόγησης",
                    "Συμβατότητα":service_dataset_compatibility("S25", ACTIVE_DF)
                },
                {
                    "Λειτουργία":"Προσαρμοστική Εξάσκηση",
                    "Συμβατότητα":service_dataset_compatibility("S26", ACTIVE_DF)
                },
                {
                    "Λειτουργία":"Υποστήριξη Διδάσκοντα",
                    "Συμβατότητα":service_dataset_compatibility("S28", ACTIVE_DF)
                },
                {
                    "Λειτουργία":"Μαθησιακές Ενδείξεις",
                    "Συμβατότητα":service_dataset_compatibility("S31", ACTIVE_DF)
                },
            ])
            st.dataframe(compatibility, use_container_width=True, hide_index=True)

        with st.expander("🧾 Προεπισκόπηση δεδομένων", expanded=False):
            st.dataframe(ACTIVE_DF.head(20), use_container_width=True, hide_index=True)

# ==========================================================
# ΚΕΝΤΡΙΚΗ ΑΝΑΦΟΡΑ
# ==========================================================
elif section == "Κεντρική Αναφορά":
    render_module_header(
        "Κεντρική Αναφορά",
        "",
        "report"
    )
    show_active_dataset_context("REPORT", compact=False)

    if ACTIVE_DF is None:
        st.info("Δεν υπάρχει ενεργή πηγή δεδομένων. Χρησιμοποιήστε την «Εισαγωγή Δεδομένων» για να φορτώσετε αρχείο.")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Συνολικός δείκτης υγείας", f"{overall:.1f}" if ACTIVE_DF is not None else "—")
    c2.metric("Ενέργειες Ποιότητας Αξιολόγησης", open21 if ACTIVE_DF is not None and "custom_s21" in st.session_state else "—")
    c3.metric("Ενέργειες Προσαρμοστικής Εξάσκησης", open26 if ACTIVE_DF is not None and "custom_s26" in st.session_state else "—")
    c4.metric("Ενέργειες Μαθησιακών Ενδείξεων", open31 if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "—")

    report=pd.DataFrame([
        {"Ενότητα":"Μαθησιακή Ανάλυση","Ρόλος":"Έγκαιρη υποστήριξη","Κατάσταση / δείκτης":"Διαθέσιμη" if ACTIVE_DF is not None else "Μη διαθέσιμη","Ανοικτές ενέργειες":"Ρύθμιση δείκτη" if ACTIVE_DF is not None else "Μη διαθέσιμο","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None else "Μη διαθέσιμο"},
        {"Ενότητα":"Ποιότητα Αξιολόγησης","Ρόλος":"Ποιότητα αξιολόγησης","Κατάσταση / δείκτης":round(health21,1) if ACTIVE_DF is not None and "custom_s21" in st.session_state else "Μη διαθέσιμο","Ανοικτές ενέργειες":open21 if ACTIVE_DF is not None and "custom_s21" in st.session_state else "Μη διαθέσιμο","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None and "custom_s21" in st.session_state else "Μη διαθέσιμο"},
        {"Ενότητα":"Δημιουργία Μαθήματος & Αξιολόγησης","Ρόλος":"Υποβοηθούμενη δημιουργία περιεχομένου","Κατάσταση / δείκτης":"Διαθέσιμη" if ACTIVE_DF is not None else "Μη διαθέσιμη","Ανοικτές ενέργειες":"Έλεγχος διδάσκοντα","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None else "Χειροκίνητο πλαίσιο"},
        {"Ενότητα":"Προσαρμοστική Εξάσκηση","Ρόλος":"Προσαρμοστική εξάσκηση","Κατάσταση / δείκτης":round(health26,1) if ACTIVE_DF is not None and "custom_s26" in st.session_state else "Μη διαθέσιμο","Ανοικτές ενέργειες":open26 if ACTIVE_DF is not None and "custom_s26" in st.session_state else "Μη διαθέσιμο","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None and "custom_s26" in st.session_state else "Μη διαθέσιμο"},
        {"Ενότητα":"Υποστήριξη Διδάσκοντα","Ρόλος":"Υποστήριξη διδάσκοντα","Κατάσταση / δείκτης":"Διαθέσιμη" if ACTIVE_DF is not None else "Μη διαθέσιμη","Ανοικτές ενέργειες":"Έλεγχος διδάσκοντα","Προέλευση":ACTIVE_NAME if ACTIVE_DF is not None else "Χειροκίνητο πλαίσιο"},
        {"Ενότητα":"Ενοποιημένες Μαθησιακές Ενδείξεις","Ρόλος":"Σύνθεση μαθησιακών ενδείξεων","Κατάσταση / δείκτης":round(health31,1) if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "Μη διαθέσιμο","Ανοικτές ενέργειες":open31 if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) else "Μη διαθέσιμο","Προέλευση":f"Ενοποιημένη πηγή — {ACTIVE_NAME}" if ACTIVE_DF is not None else "Μη διαθέσιμο"},
    ])
    st.markdown("#### Τι χρειάζεται προσοχή")
    _attention = []
    if ACTIVE_DF is not None and "custom_s21" in st.session_state and open21:
        _attention.append(f"{open21} έλεγχοι ποιότητας αξιολόγησης")
    if ACTIVE_DF is not None and "custom_s26" in st.session_state and open26:
        _attention.append(f"{open26} ενέργειες προσαρμοστικής εξάσκησης")
    if ACTIVE_DF is not None and (("custom_s21" in st.session_state) or ("custom_s26" in st.session_state)) and open31:
        _attention.append(f"{open31} μαθησιακές ενδείξεις υψηλής προτεραιότητας")
    if _attention:
        st.warning(" • ".join(_attention))
    elif ACTIVE_DF is not None:
        st.success("Δεν υπάρχουν αυτή τη στιγμή καταγεγραμμένες ενέργειες υψηλής προτεραιότητας.")

    with st.expander("Αναλυτική κατάσταση λειτουργιών", expanded=False):
        st.dataframe(report,use_container_width=True,hide_index=True)

    summary=f"""INELSO Academia ΚΕΝΤΡΙΚΟ ΣΥΣΤΗΜΑ ΕΞΥΠΝΗΣ ΜΑΘΗΣΗΣ — ULTIMATE PRO WRITE

ενεργό σύνολο δεδομένων: {ACTIVE_NAME}
Dataset type: {ACTIVE_KIND}
Συνολικός δείκτης υγείας: {f"{overall:.1f}/100" if ACTIVE_DF is not None else "—"}

Ποιότητα Αξιολόγησης:
- δείκτης υγείας: {health21:.1f}
- ανοικτές ενέργειες: {open21}

Προσαρμοστική Εξάσκηση:
- δείκτης υγείας: {health26:.1f}
- ανοικτές ενέργειες: {open26}

Ενοποιημένες Μαθησιακές Ενδείξεις:
- δείκτης υγείας: {health31:.1f}
- ανοικτές ενέργειες: {open31}
- προέλευση: σύνθεση πραγματικών/ελεγχόμενων ενδείξεις από Ποιότητα Αξιολόγησης και Προσαρμοστική Εξάσκηση

Αρχιτεκτονική:
Moodle -> Ποιότητα Αξιολόγησης / 26 ενδείξεις -> Ενοποιημένες Μαθησιακές Ενδείξεις synthesis -> ανθρώπινος έλεγχος -> παρέμβαση / retry / revision -> αναφορά

Δεν πραγματοποιείται αυτόματη αλλαγή επίσημων βαθμών στο Moodle.
"""
    st.download_button(
        "Λήψη κεντρικής σύνοψης",
        summary.encode("utf-8"),
        "INELSO Academia_Kentriki_Synopsi_ULTIMATE_PRO_WRITE.txt",
        "text/plain"
    )
