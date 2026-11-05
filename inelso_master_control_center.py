
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
# SERVICE 31: ΠΛΕΟΝ ΠΑΡΑΓΕΤΑΙ ΑΠΟ SIGNALS 21 + 26
# ==========================================================
# Μόνο το Q08 έχει τεκμηριωμένη σημασιολογική ετικέτα από το υπάρχον workflow (CTR).
# Τα Q09/Q07 διατηρούνται ως item-level signals χωρίς επινοημένη θεματική ονομασία.

# 1) Signals από Service 21
s21_signals = pd.DataFrame([
    {
        "οντότητα":"CTR interpretation (Q08)",
        "πηγή":"Service 21",
        "τύπος":"Μαθησιακή έννοια με τεκμηριωμένη αντιστοίχιση",
        "ισχύς_σήματος":round(service21.loc[service21["ερώτηση"]=="S21 Marketing Q08","διάκριση"].iloc[0],2),
        "κατεύθυνση":"ΘΕΤΙΚΗ",
        "ενέργεια":"Διατήρηση / χρήση ως ισχυρό σημείο αναφοράς",
        "προέλευση":"Ελεγχόμενο Moodle validation"
    },
    {
        "οντότητα":"S21 Marketing Q09",
        "πηγή":"Service 21",
        "τύπος":"Αδύναμο item-level signal",
        "ισχύς_σήματος":round(100-service21.loc[service21["ερώτηση"]=="S21 Marketing Q09","διάκριση"].iloc[0],2),
        "κατεύθυνση":"ΑΡΝΗΤΙΚΗ",
        "ενέργεια":"Ανθρώπινος έλεγχος item πριν από θεματική ερμηνεία",
        "προέλευση":"Ελεγχόμενο Moodle validation"
    },
    {
        "οντότητα":"S21 Marketing Q07",
        "πηγή":"Service 21",
        "τύπος":"Μη εκτιμήσιμο item-level signal",
        "ισχύς_σήματος":100.0,
        "κατεύθυνση":"ΑΒΕΒΑΙΗ",
        "ενέργεια":"Έλεγχος scoring / response pattern",
        "προέλευση":"Ελεγχόμενο Moodle validation"
    },
])

# 2) Signals από Service 26: gaps στην 1η προσπάθεια και επίλυση στη 2η
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
# SERVICE 21
# ==========================================================
elif section == "Service 21 — Ευφυής Ποιότητα Αξιολόγησης":
    st.title("Service 21 — Ευφυής Ποιότητα Αξιολόγησης")
    st.caption("Από ψυχομετρικούς δείκτες σε τεκμηριωμένη ουρά ελέγχου ερωτήσεων.")

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
