
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="INELSO Smart Learning Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CSS / embedded-friendly UI
# ----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.9rem;}
[data-testid="stMetricLabel"] {font-size: .95rem;}
h1 {font-size: 2.15rem !important;}
h2 {font-size: 1.55rem !important;}
h3 {font-size: 1.25rem !important;}
.small-note {font-size:0.9rem;color:#666;}
.inelso-card {
    padding: 1rem 1.1rem;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    background: white;
    margin-bottom: .8rem;
}
.status-ready {padding:.7rem 1rem;background:#eaf7ec;border-radius:10px;}
.status-review {padding:.7rem 1rem;background:#fff5d6;border-radius:10px;}
.status-critical {padding:.7rem 1rem;background:#fde8e8;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Shared demo data
# ----------------------------
service21 = pd.DataFrame([
    {"question":"S21 Marketing Q01","facility":2.00,"discrimination":34.92,"efficiency":100.00},
    {"question":"S21 Marketing Q02","facility":2.00,"discrimination":34.92,"efficiency":100.00},
    {"question":"S21 Marketing Q04","facility":0.67,"discrimination":39.68,"efficiency":100.00},
    {"question":"S21 Marketing Q07","facility":0.00,"discrimination":None,"efficiency":None},
    {"question":"S21 Marketing Q10","facility":2.00,"discrimination":34.92,"efficiency":100.00},
    {"question":"S21 Marketing Q05","facility":20.00,"discrimination":44.11,"efficiency":61.54},
    {"question":"S21 Marketing Q09","facility":72.00,"discrimination":12.88,"efficiency":21.49},
    {"question":"S21 Marketing Q03","facility":44.00,"discrimination":73.22,"efficiency":92.35},
    {"question":"S21 Marketing Q06","facility":32.00,"discrimination":44.38,"efficiency":52.03},
    {"question":"S21 Marketing Q08","facility":52.00,"discrimination":60.89,"efficiency":80.71},
])

def s21_status(d):
    if pd.isna(d):
        return "N/A"
    if d >= 40:
        return "ΙΣΧΥΡΗ"
    if d >= 20:
        return "ΙΚΑΝΟΠΟΙΗΤΙΚΗ"
    if d >= 0:
        return "ΕΛΕΓΧΟΣ"
    return "ΚΡΙΣΙΜΗ"

service21["status"] = service21["discrimination"].apply(s21_status)

rubric = pd.DataFrame([
    {"criterion":"RC-01","name":"Εννοιολογική ακρίβεια","weight":30},
    {"criterion":"RC-02","name":"Μέθοδος / συλλογισμός","weight":25},
    {"criterion":"RC-03","name":"Τεκμηρίωση / υποστήριξη","weight":15},
    {"criterion":"RC-04","name":"Πληρότητα","weight":15},
    {"criterion":"RC-05","name":"Σαφήνεια / επικοινωνία","weight":15},
])

service26_history = pd.DataFrame([
    {"attempt":1,"RC-01":3,"RC-02":2,"RC-03":2,"RC-04":2,"RC-05":3,
     "score":61.25,"mastery":"Developing","confidence":0.65,"status":"ESCALATE"},
    {"attempt":2,"RC-01":4,"RC-02":4,"RC-03":4,"RC-04":4,"RC-05":4,
     "score":100.00,"mastery":"Mastery","confidence":0.85,"status":"READY"},
])

cohort26 = pd.DataFrame([
    {"learner":"Demo Student 0201","score":100.00,"mastery":"Mastery","confidence":0.85,"status":"READY"},
    {"learner":"Demo Student 0202","score":78.75,"mastery":"Proficient","confidence":0.81,"status":"READY"},
    {"learner":"Demo Student 0203","score":66.25,"mastery":"Developing","confidence":0.68,"status":"ESCALATE"},
    {"learner":"Demo Student 0204","score":91.25,"mastery":"Mastery","confidence":0.88,"status":"READY"},
    {"learner":"Demo Student 0205","score":57.50,"mastery":"Developing","confidence":0.72,"status":"RETRY"},
])

# ----------------------------
# Navigation
# ----------------------------
st.sidebar.title("INELSO Control Center")
section = st.sidebar.radio(
    "Υπηρεσία",
    [
        "Αρχική",
        "Service 21 — Ποιότητα αξιολόγησης",
        "Service 26 — Προσαρμοστική εξάσκηση",
        "Service 31 — Learning Insights",
        "Master αναφορά"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Master demo • read-only decision support")
st.sidebar.caption("Moodle = σημείο υποβολής / επίσημης διαχείρισης")

# ----------------------------
# Home
# ----------------------------
if section == "Αρχική":
    st.title("INELSO Smart Learning Control Center")
    st.caption("Ενιαία εποπτεία ποιότητας αξιολόγησης, προσαρμοστικής εξάσκησης και learning insights.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Service 21", "10 items")
    c2.metric("Service 26", "2 attempts")
    c3.metric("Learners σε cohort demo", len(cohort26))
    c4.metric("Open actions", int((cohort26["status"]!="READY").sum()) + int((service21["status"]=="ΕΛΕΓΧΟΣ").sum()))

    st.subheader("Master workflow")
    st.markdown("""
**Moodle → δεδομένα αξιολόγησης / submissions → Service analytics → decision rules → human review όπου απαιτείται → feedback / retry / revision → reports**
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="inelso-card">', unsafe_allow_html=True)
        st.markdown("### Service 21")
        st.write("Ψυχομετρικός έλεγχος quiz και question-level ποιότητα.")
        st.write("• Facility / difficulty")
        st.write("• Discrimination")
        st.write("• Priority review")
        st.write("• Keep / revise / inspect logic")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="inelso-card">', unsafe_allow_html=True)
        st.markdown("### Service 26")
        st.write("Adaptive formative practice και learner progression.")
        st.write("• Rubric profile")
        st.write("• Mastery")
        st.write("• Confidence / safety gate")
        st.write("• Retry / enrichment / human review")
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Το Service 31 εμφανίζεται ως επόμενο module της master αρχιτεκτονικής και θα συνδεθεί στο ίδιο περιβάλλον.")

# ----------------------------
# Service 21
# ----------------------------
elif section == "Service 21 — Ποιότητα αξιολόγησης":
    st.title("Service 21 — Ποιότητα αξιολόγησης")

    valid_disc = service21["discrimination"].dropna()
    review_count = int((service21["status"]=="ΕΛΕΓΧΟΣ").sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ερωτήσεις", len(service21))
    c2.metric("Μέσος facility", f"{service21['facility'].mean():.1f}%")
    c3.metric("Μέσο discrimination", f"{valid_disc.mean():.1f}%")
    c4.metric("Χρειάζονται έλεγχο", review_count)

    tabs = st.tabs(["Σύνοψη","Τεχνική ανάλυση","Priority review","Decision support"])

    with tabs[0]:
        counts = service21["status"].value_counts().rename_axis("Κατηγορία").reset_index(name="Πλήθος")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        chart = service21.dropna(subset=["discrimination"]).sort_values("discrimination", ascending=False)
        st.bar_chart(chart.set_index("question")[["discrimination"]])

    with tabs[1]:
        st.dataframe(
            service21.rename(columns={
                "question":"Ερώτηση","facility":"Facility index",
                "discrimination":"Discrimination index",
                "efficiency":"Discriminative efficiency","status":"Κατάσταση"
            }),
            use_container_width=True,
            hide_index=True
        )

    with tabs[2]:
        priority = service21[(service21["status"].isin(["ΕΛΕΓΧΟΣ","ΚΡΙΣΙΜΗ"])) | service21["discrimination"].isna()].copy()
        st.dataframe(priority, use_container_width=True, hide_index=True)
        if not priority.empty:
            st.warning("Προτεραιότητα σε items με χαμηλή ή μη εκτιμήσιμη διάκριση. Η τελική απόφαση αναθεώρησης παραμένει ανθρώπινη.")

    with tabs[3]:
        choice = st.selectbox("Επιλέξτε ερώτηση", service21["question"].tolist())
        row = service21.loc[service21["question"]==choice].iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("Facility", f"{row['facility']:.2f}%")
        c2.metric("Discrimination", "N/A" if pd.isna(row["discrimination"]) else f"{row['discrimination']:.2f}%")
        c3.metric("Status", row["status"])
        if pd.isna(row["discrimination"]):
            st.info("Δεν υπάρχει επαρκής εκτίμηση discrimination. Προτείνεται επιθεώρηση του item και των response patterns.")
        elif row["discrimination"] < 20:
            st.warning("Ασθενής διάκριση. Προτείνεται έλεγχος stem, distractors, scoring και alignment με το learning objective.")
        elif row["discrimination"] < 40:
            st.info("Ικανοποιητική διάκριση. Διατηρείται, με παρακολούθηση σε επόμενο cohort.")
        else:
            st.success("Ισχυρή διάκριση. Το item λειτουργεί καλά με βάση το συγκεκριμένο δείγμα.")

# ----------------------------
# Service 26
# ----------------------------
elif section == "Service 26 — Προσαρμοστική εξάσκηση":
    st.title("Service 26 — Προσαρμοστική εξάσκηση")
    view = st.radio("Προβολή", ["Learner","Teacher"], horizontal=True)

    if view == "Learner":
        latest = service26_history.iloc[-1]
        prev = service26_history.iloc[-2]
        gain = latest["score"] - prev["score"]

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Score", f"{latest['score']:.0f}/100")
        c2.metric("Mastery", latest["mastery"])
        c3.metric("Confidence", f"{latest['confidence']:.2f}")
        c4.metric("Status", latest["status"])
        c5.metric("Progress", f"+{gain:.2f}")

        tabs = st.tabs(["Rubric","Πρόοδος","Next action"])

        with tabs[0]:
            display = rubric.copy()
            display["Attempt 1"] = [service26_history.iloc[0][c] for c in display["criterion"]]
            display["Attempt 2"] = [service26_history.iloc[1][c] for c in display["criterion"]]
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.bar_chart(display.set_index("criterion")[["Attempt 1","Attempt 2"]])

        with tabs[1]:
            st.line_chart(service26_history.set_index("attempt")[["score"]])
            st.success(f"Βελτίωση {prev['score']:.2f} → {latest['score']:.2f} ({gain:+.2f}).")

        with tabs[2]:
            st.success("READY — confidence ≥ 0.70, RC-01 δεν ενεργοποιεί critical rule και το mastery είναι Mastery.")
            st.write("**Προτεινόμενη ενέργεια:** Δεν απαιτείται νέο retry.")
            st.write("**Προτεινόμενο resource:** Enrichment / επόμενη δραστηριότητα.")

    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Learners", len(cohort26))
        c2.metric("READY", int((cohort26["status"]=="READY").sum()))
        c3.metric("RETRY", int((cohort26["status"]=="RETRY").sum()))
        c4.metric("ESCALATE", int((cohort26["status"]=="ESCALATE").sum()))
        st.dataframe(cohort26, use_container_width=True, hide_index=True)
        st.subheader("Learners needing action")
        st.dataframe(cohort26[cohort26["status"]!="READY"], use_container_width=True, hide_index=True)

# ----------------------------
# Service 31 placeholder
# ----------------------------
elif section == "Service 31 — Learning Insights":
    st.title("Service 31 — Learning Insights")
    st.info("Το Service 31 είναι το επόμενο module για σύνδεση στη master αρχιτεκτονική.")
    st.write("Προτεινόμενη master κατεύθυνση:")
    st.write("• concept hotspots")
    st.write("• cohort-level learning insights")
    st.write("• faculty action workflow")
    st.write("• privacy / governance")
    st.write("• σύνδεση με signals από Service 21 και Service 26")

# ----------------------------
# Master report
# ----------------------------
else:
    st.title("Master αναφορά")
    s21_actions = int((service21["status"]=="ΕΛΕΓΧΟΣ").sum()) + int(service21["discrimination"].isna().sum())
    s26_actions = int((cohort26["status"]!="READY").sum())

    c1,c2,c3 = st.columns(3)
    c1.metric("Service 21 actions", s21_actions)
    c2.metric("Service 26 actions", s26_actions)
    c3.metric("Total open actions", s21_actions + s26_actions)

    master_actions = pd.DataFrame([
        {"Service":"21","Area":"Assessment quality","Action":"Review weak / non-estimable items","Open":s21_actions},
        {"Service":"26","Area":"Adaptive practice","Action":"Review RETRY / ESCALATE learners","Open":s26_actions},
        {"Service":"31","Area":"Learning insights","Action":"Pending master integration","Open":1},
    ])
    st.dataframe(master_actions, use_container_width=True, hide_index=True)

    st.download_button(
        "Λήψη master action list",
        master_actions.to_csv(index=False).encode("utf-8-sig"),
        "INELSO_Master_Action_List.csv",
        "text/csv"
    )

    summary = f"""INELSO SMART LEARNING CONTROL CENTER
Master summary

Service 21 open actions: {s21_actions}
Service 26 open actions: {s26_actions}
Service 31: pending integration

Architecture:
Moodle -> analytics / decision support -> human review where needed -> feedback / revision / retry -> reporting

The control center is read-only and does not automatically alter official Moodle grades.
"""
    st.download_button(
        "Λήψη master summary",
        summary.encode("utf-8"),
        "INELSO_Master_Summary.txt",
        "text/plain"
    )
