
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="INELSO Smart Learning Control Center v4",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown("""
<style>
:root{
 --navy:#102a43; --navy2:#173b5e; --blue:#2a6f9e; --teal:#2a8c99;
 --soft:#f5f8fb; --line:#dde6ee; --muted:#667085;
 --green:#2f855a; --amber:#b7791f; --red:#c53030;
}
.block-container{padding-top:.8rem;padding-bottom:3rem;max-width:1550px}
h1{color:var(--navy);font-size:2.15rem!important}
h2{color:var(--navy2);font-size:1.5rem!important}
h3{font-size:1.15rem!important}
[data-testid="stMetricValue"]{font-size:1.75rem}
[data-testid="stMetricLabel"]{font-size:.9rem}
.hero{
 background:linear-gradient(135deg,#102a43,#1f5f87);
 color:white;border-radius:20px;padding:1.25rem 1.45rem;margin:.15rem 0 1rem 0;
}
.hero h1,.hero h2,.hero h3{color:white!important;margin:0}
.hero p{margin:.45rem 0 0;opacity:.92}
.ribbon{
 display:inline-block;padding:.22rem .62rem;border-radius:999px;font-size:.78rem;
 font-weight:700;margin-right:.35rem
}
.live{background:#e7f6ec;color:#236b3a}
.controlled{background:#eaf2ff;color:#24527a}
.synthetic{background:#fff4da;color:#8a5a00}
.pending{background:#f2f4f7;color:#59636e}
.card{
 border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;background:white
}
.callout{
 border-left:5px solid var(--teal);background:#f4fbfc;padding:.85rem 1rem;border-radius:10px
}
.flow{
 border:1px dashed #a9bdcf;background:#f8fbfd;padding:.75rem 1rem;border-radius:12px;
 font-weight:650
}
.compact{font-size:.9rem;color:#475467}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# DATA
# ==========================================================
service21 = pd.DataFrame([
    {"question":"S21 Marketing Q01","facility":2.00,"discrimination":34.92,"efficiency":100.00},
    {"question":"S21 Marketing Q02","facility":2.00,"discrimination":34.92,"efficiency":100.00},
    {"question":"S21 Marketing Q04","facility":0.67,"discrimination":39.68,"efficiency":100.00},
    {"question":"S21 Marketing Q07","facility":0.00,"discrimination":np.nan,"efficiency":np.nan},
    {"question":"S21 Marketing Q10","facility":2.00,"discrimination":34.92,"efficiency":100.00},
    {"question":"S21 Marketing Q05","facility":20.00,"discrimination":44.11,"efficiency":61.54},
    {"question":"S21 Marketing Q09","facility":72.00,"discrimination":12.88,"efficiency":21.49},
    {"question":"S21 Marketing Q03","facility":44.00,"discrimination":73.22,"efficiency":92.35},
    {"question":"S21 Marketing Q06","facility":32.00,"discrimination":44.38,"efficiency":52.03},
    {"question":"S21 Marketing Q08","facility":52.00,"discrimination":60.89,"efficiency":80.71},
])

def s21_state(d):
    if pd.isna(d): return "N/A"
    if d >= 40: return "STRONG"
    if d >= 20: return "OK"
    if d >= 0: return "REVIEW"
    return "CRITICAL"

def s21_action(r):
    d, f = r["discrimination"], r["facility"]
    if pd.isna(d): return "Inspect scoring / response pattern"
    if d < 0: return "Urgent review of key/scoring"
    if d < 20 and (f <= 10 or f >= 90): return "Revise: extreme facility + weak discrimination"
    if d < 20: return "Review stem, distractors and alignment"
    if d < 40: return "Keep with monitoring"
    return "Keep"

service21["state"] = service21["discrimination"].apply(s21_state)
service21["action"] = service21.apply(s21_action, axis=1)

rubric = pd.DataFrame([
    {"criterion":"RC-01","name":"Εννοιολογική ακρίβεια","weight":30},
    {"criterion":"RC-02","name":"Μέθοδος / συλλογισμός","weight":25},
    {"criterion":"RC-03","name":"Τεκμηρίωση / υποστήριξη","weight":15},
    {"criterion":"RC-04","name":"Πληρότητα","weight":15},
    {"criterion":"RC-05","name":"Σαφήνεια / επικοινωνία","weight":15},
])

s26 = pd.DataFrame([
    {"attempt":1,"RC-01":3,"RC-02":2,"RC-03":2,"RC-04":2,"RC-05":3,
     "score":61.25,"mastery":"Developing","confidence":0.65},
    {"attempt":2,"RC-01":4,"RC-02":4,"RC-03":4,"RC-04":4,"RC-05":4,
     "score":100.00,"mastery":"Mastery","confidence":0.85},
])

def adaptive_status(score, confidence, rc01):
    if confidence < .70 or rc01 <= 1: return "ESCALATE"
    if score < 70: return "RETRY"
    return "READY"

s26["status"] = s26.apply(lambda r: adaptive_status(r["score"],r["confidence"],r["RC-01"]), axis=1)

cohort26 = pd.DataFrame([
    {"learner":"Demo Student 0201","score":100.00,"confidence":0.85,"mastery":"Mastery","status":"READY"},
    {"learner":"Demo Student 0202","score":78.75,"confidence":0.81,"mastery":"Proficient","status":"READY"},
    {"learner":"Demo Student 0203","score":66.25,"confidence":0.68,"mastery":"Developing","status":"ESCALATE"},
    {"learner":"Demo Student 0204","score":91.25,"confidence":0.88,"mastery":"Mastery","status":"READY"},
    {"learner":"Demo Student 0205","score":57.50,"confidence":0.72,"mastery":"Developing","status":"RETRY"},
])

concept31 = pd.DataFrame([
    {"concept":"Audience targeting","mastery":82,"engagement":88,"risk":12,"trend":6},
    {"concept":"CTR interpretation","mastery":76,"engagement":81,"risk":18,"trend":3},
    {"concept":"Campaign metrics","mastery":69,"engagement":74,"risk":29,"trend":1},
    {"concept":"Conversion funnel","mastery":58,"engagement":62,"risk":43,"trend":-4},
    {"concept":"Attribution logic","mastery":51,"engagement":55,"risk":52,"trend":-7},
    {"concept":"B2B channel strategy","mastery":64,"engagement":68,"risk":35,"trend":-2},
])

def concept_priority(r):
    score=(100-r["mastery"])*.45+(100-r["engagement"])*.25+r["risk"]*.30
    label="HIGH" if score>=45 else ("MEDIUM" if score>=30 else "LOW")
    return pd.Series([round(score,1),label])

concept31[["priority_score","priority"]] = concept31.apply(concept_priority,axis=1)

# ==========================================================
# GLOBAL HEALTH / CROSS-SERVICE
# ==========================================================
def clamp(x): return max(0,min(100,x))

s21_valid = service21["discrimination"].dropna()
s21_health = clamp(
    55
    + 0.35*s21_valid.mean()
    - 6*int((service21["state"]=="REVIEW").sum())
    - 8*int(service21["discrimination"].isna().sum())
)
s26_ready_rate = (cohort26["status"]=="READY").mean()*100
s26_health = clamp(
    .55*cohort26["score"].mean()
    + .25*(cohort26["confidence"].mean()*100)
    + .20*s26_ready_rate
)
s31_health = clamp(
    .45*concept31["mastery"].mean()
    + .25*concept31["engagement"].mean()
    + .30*(100-concept31["risk"].mean())
)
overall_health = round((s21_health+s26_health+s31_health)/3,1)

s21_open = int(((service21["state"]=="REVIEW") | service21["discrimination"].isna()).sum())
s26_open = int((cohort26["status"]!="READY").sum())
s31_open = int((concept31["priority"]=="HIGH").sum())

# Cross-service evidence chain
evidence_chain = pd.DataFrame([
    {"signal":"Assessment item weakness","source":"Service 21","evidence":"Q09 weak discrimination","linked_action":"Check concept alignment / targeted practice","target":"Service 31 / Service 26"},
    {"signal":"Learner uncertainty","source":"Service 26","evidence":"ESCALATE case","linked_action":"Human review before retry","target":"Instructor"},
    {"signal":"Concept hotspot","source":"Service 31","evidence":"Attribution logic high priority","linked_action":"Inspect teaching + assessment + practice","target":"Services 21 & 26"},
])

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown("## INELSO Control Center")
st.sidebar.caption("Master v4 • decision-support architecture")
section = st.sidebar.radio(
    "Πλοήγηση",
    [
        "Executive Command Center",
        "Service 21 — Assessment Intelligence",
        "Service 26 — Adaptive Practice Intelligence",
        "Service 31 — Learning Insights",
        "Cross-Service Intelligence",
        "Presentation Mode",
        "Master Report"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="ribbon live">LIVE-capable</span>'
    '<span class="ribbon controlled">CONTROLLED</span>'
    '<span class="ribbon synthetic">SYNTHETIC</span>',
    unsafe_allow_html=True
)
st.sidebar.caption("Service 31 values in this build are synthetic demo values.")
st.sidebar.caption("No automatic official grade writes.")

# ==========================================================
# EXECUTIVE
# ==========================================================
if section == "Executive Command Center":
    st.markdown("""
    <div class="hero">
      <h1>INELSO Smart Learning Control Center</h1>
      <p>Assessment quality + adaptive practice + learning insights σε ένα ενιαίο, ερμηνεύσιμο control layer.</p>
    </div>
    """, unsafe_allow_html=True)

    a,b,c,d,e = st.columns(5)
    a.metric("Overall health", f"{overall_health:.1f}/100")
    b.metric("Service 21 health", f"{s21_health:.1f}")
    c.metric("Service 26 health", f"{s26_health:.1f}")
    d.metric("Service 31 health", f"{s31_health:.1f}")
    e.metric("Open actions", s21_open+s26_open+s31_open)

    st.markdown('<div class="flow">Moodle → signals → analytics → transparent decision rules → human review where needed → intervention / retry / revision → audit & report</div>', unsafe_allow_html=True)

    st.write("")
    c1,c2,c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.subheader("Service 21")
            st.markdown('<span class="ribbon live">LIVE-capable</span>',unsafe_allow_html=True)
            st.write("Assessment Intelligence")
            st.metric("Open item actions", s21_open)
            st.write("Question quality • discrimination • monitoring • item review")
    with c2:
        with st.container(border=True):
            st.subheader("Service 26")
            st.markdown('<span class="ribbon controlled">CONTROLLED</span>',unsafe_allow_html=True)
            st.write("Adaptive Practice Intelligence")
            st.metric("Learners needing action", s26_open)
            st.write("Rubric gaps • mastery • confidence • retry / escalation")
    with c3:
        with st.container(border=True):
            st.subheader("Service 31")
            st.markdown('<span class="ribbon synthetic">SYNTHETIC DEMO</span>',unsafe_allow_html=True)
            st.write("Learning Insights")
            st.metric("High-priority concepts", s31_open)
            st.write("Concept hotspots • cohort insights • faculty workflow")

    st.subheader("Executive attention list")
    attention = pd.DataFrame([
        {"Priority":"1","Service":"21","Issue":"Non-estimable / weak item signal","Owner":"Assessment designer"},
        {"Priority":"1","Service":"26","Issue":"ESCALATE learner case","Owner":"Instructor"},
        {"Priority":"2","Service":"31","Issue":"High-priority concept hotspot","Owner":"Faculty / learning designer"},
    ])
    st.dataframe(attention,use_container_width=True,hide_index=True)

# ==========================================================
# SERVICE 21
# ==========================================================
elif section == "Service 21 — Assessment Intelligence":
    st.title("Service 21 — Assessment Intelligence")
    st.caption("Από psychometrics σε actionable item-level quality assurance.")

    a,b,c,d,e = st.columns(5)
    a.metric("Items",len(service21))
    b.metric("Facility mean",f"{service21['facility'].mean():.1f}%")
    c.metric("Discrimination mean",f"{s21_valid.mean():.1f}%")
    d.metric("Health score",f"{s21_health:.1f}/100")
    e.metric("Open actions",s21_open)

    tabs=st.tabs(["Executive","Health matrix","Priority queue","Item inspector","Export"])
    with tabs[0]:
        counts=service21["state"].value_counts().rename_axis("State").reset_index(name="Count")
        st.dataframe(counts,use_container_width=True,hide_index=True)
        st.bar_chart(service21.dropna(subset=["discrimination"]).sort_values("discrimination",ascending=False).set_index("question")[["discrimination"]])
        st.info("Η διάκριση χρησιμοποιείται ως decision-support signal, όχι ως αυτόματος κανόνας διαγραφής item.")

    with tabs[1]:
        m=service21.copy()
        m["difficulty_band"]=pd.cut(
            m["facility"],[-1,20,40,60,80,101],
            labels=["Very difficult","Difficult","Balanced","Easy","Very easy"]
        )
        st.dataframe(m[["question","facility","difficulty_band","discrimination","efficiency","state"]],use_container_width=True,hide_index=True)
        st.caption("Συνδυασμός facility + discrimination + efficiency για ποιοτικό έλεγχο.")

    with tabs[2]:
        q=service21[(service21["state"]=="REVIEW")|service21["discrimination"].isna()].copy()
        q["priority"]=q.apply(lambda r:1 if pd.isna(r["discrimination"]) else 2,axis=1)
        q=q.sort_values(["priority","facility"])
        st.dataframe(q[["priority","question","facility","discrimination","action"]],use_container_width=True,hide_index=True)
        st.warning("Priority 1: μη εκτιμήσιμο signal. Priority 2: weak discrimination.")

    with tabs[3]:
        choice=st.selectbox("Ερώτηση",service21["question"].tolist())
        r=service21.loc[service21["question"]==choice].iloc[0]
        x1,x2,x3,x4=st.columns(4)
        x1.metric("Facility",f"{r['facility']:.2f}%")
        x2.metric("Discrimination","N/A" if pd.isna(r["discrimination"]) else f"{r['discrimination']:.2f}%")
        x3.metric("Efficiency","N/A" if pd.isna(r["efficiency"]) else f"{r['efficiency']:.2f}%")
        x4.metric("State",r["state"])
        st.write("**Recommended action:**",r["action"])
        st.markdown('<div class="callout"><b>Human decision checkpoint:</b> ελέγξτε item stem, key, distractors, alignment και response distribution πριν από αλλαγή.</div>',unsafe_allow_html=True)

    with tabs[4]:
        st.download_button("Download Service 21 action table",service21.to_csv(index=False).encode("utf-8-sig"),"S21_action_table.csv","text/csv")

# ==========================================================
# SERVICE 26
# ==========================================================
elif section == "Service 26 — Adaptive Practice Intelligence":
    st.title("Service 26 — Adaptive Practice Intelligence")
    st.caption("Criterion-level adaptivity + confidence gating + teacher action queue.")

    latest=s26.iloc[-1]; prev=s26.iloc[-2]; gain=latest["score"]-prev["score"]
    a,b,c,d,e=st.columns(5)
    a.metric("Current score",f"{latest['score']:.0f}/100")
    b.metric("Mastery",latest["mastery"])
    c.metric("Confidence",f"{latest['confidence']:.2f}")
    d.metric("Progress",f"+{gain:.2f}")
    e.metric("Health score",f"{s26_health:.1f}/100")

    view=st.radio("View",["Learner","Teacher"],horizontal=True)
    if view=="Learner":
        tabs=st.tabs(["Adaptive profile","Rubric gaps","Trajectory","Decision engine","Export"])
        with tabs[0]:
            st.success("READY — confidence gate passed, RC-01 critical rule inactive.")
            st.write("**Progress:** 61.25 → 100.00")
            st.write("**Recommended next step:** Enrichment / next practice task.")
        with tabs[1]:
            gap=[]
            for _,rr in rubric.iterrows():
                level=int(latest[rr["criterion"]]); g=4-level
                gap.append({"criterion":rr["criterion"],"name":rr["name"],"level":level,"gap":g,"weighted_gap":g*rr["weight"]})
            gd=pd.DataFrame(gap).sort_values("weighted_gap",ascending=False)
            st.dataframe(gd,use_container_width=True,hide_index=True)
            st.line_chart(pd.DataFrame({
                "Attempt 1":[s26.iloc[0][c] for c in rubric["criterion"]],
                "Attempt 2":[s26.iloc[1][c] for c in rubric["criterion"]],
            },index=rubric["criterion"]))
        with tabs[2]:
            st.line_chart(s26.set_index("attempt")[["score","confidence"]])
            st.dataframe(s26,use_container_width=True,hide_index=True)
        with tabs[3]:
            st.markdown("""
**Decision engine**
- confidence < 0.70 → ESCALATE  
- RC-01 Level 0–1 → ESCALATE + mastery cap  
- score < 70, safety OK → RETRY  
- score ≥ 70, safety OK → READY  
- RETRY → largest weighted rubric gap  
- READY → enrichment / next activity
""")
            st.info("Deterministic, explainable, human-in-the-loop.")
        with tabs[4]:
            st.download_button("Download learner trajectory",s26.to_csv(index=False).encode("utf-8-sig"),"S26_trajectory.csv","text/csv")
    else:
        t1,t2,t3,t4=st.columns(4)
        t1.metric("Learners",len(cohort26))
        t2.metric("READY",int((cohort26["status"]=="READY").sum()))
        t3.metric("RETRY",int((cohort26["status"]=="RETRY").sum()))
        t4.metric("ESCALATE",int((cohort26["status"]=="ESCALATE").sum()))
        queue=cohort26.copy()
        queue["priority"]=queue["status"].map({"ESCALATE":1,"RETRY":2,"READY":9})
        st.dataframe(queue.sort_values(["priority","score"]),use_container_width=True,hide_index=True)
        st.warning("ESCALATE always precedes RETRY in the teacher queue.")

# ==========================================================
# SERVICE 31
# ==========================================================
elif section == "Service 31 — Learning Insights":
    st.title("Service 31 — Learning Insights")
    st.markdown('<span class="ribbon synthetic">CONTROLLED SYNTHETIC DEMO</span>',unsafe_allow_html=True)
    st.caption("Concept-level and cohort-level decision support — not yet operational Moodle analytics.")

    a,b,c,d=st.columns(4)
    a.metric("Concepts",len(concept31))
    b.metric("Health score",f"{s31_health:.1f}/100")
    c.metric("High priority",int((concept31["priority"]=="HIGH").sum()))
    d.metric("Negative trends",int((concept31["trend"]<0).sum()))

    tabs=st.tabs(["Hotspots","Signal map","Priority engine","Faculty workflow","Governance"])
    with tabs[0]:
        ranked=concept31.sort_values("priority_score",ascending=False)
        st.dataframe(ranked,use_container_width=True,hide_index=True)
        st.bar_chart(ranked.set_index("concept")[["mastery","engagement","risk"]])
    with tabs[1]:
        st.subheader("Concept signal map")
        sm=concept31[["concept","mastery","engagement","risk","trend"]].copy()
        sm["learning_signal"]=sm["mastery"]-sm["risk"]+sm["trend"]
        st.dataframe(sm.sort_values("learning_signal"),use_container_width=True,hide_index=True)
        st.caption("Demo-derived composite for prioritization; not a validated predictive model.")
    with tabs[2]:
        st.markdown("""
**Transparent priority score**

`0.45 × mastery gap + 0.25 × engagement gap + 0.30 × risk`

The score is used only to order instructional review.
""")
        st.dataframe(concept31[["concept","priority_score","priority","trend"]].sort_values("priority_score",ascending=False),use_container_width=True,hide_index=True)
    with tabs[3]:
        st.markdown("""
1. Detect hotspot  
2. Check consistency with Service 21 assessment signals  
3. Check consistency with Service 26 learner/practice signals  
4. Inspect learning material and assessment alignment  
5. Select intervention: reteach / targeted practice / resource / human support  
6. Monitor next attempt or next cohort  
7. Close action only after evidence of improvement
""")
    with tabs[4]:
        st.info("Synthetic/demo data are explicitly separated from operational data.")
        st.write("• Read-only")
        st.write("• Human-in-the-loop")
        st.write("• Minimum necessary personal data")
        st.write("• Auditable rules")
        st.write("• No official grade modification")

# ==========================================================
# CROSS-SERVICE
# ==========================================================
elif section == "Cross-Service Intelligence":
    st.title("Cross-Service Intelligence")
    st.caption("Το νέο επίπεδο της v4: όχι τρία ανεξάρτητα dashboards, αλλά μία evidence chain.")

    st.subheader("Evidence chain")
    st.dataframe(evidence_chain,use_container_width=True,hide_index=True)

    st.subheader("Unified action queue")
    actions=[]
    for _,r in service21[(service21["state"]=="REVIEW")|service21["discrimination"].isna()].iterrows():
        actions.append({"priority":1 if pd.isna(r["discrimination"]) else 2,"service":"21","entity":r["question"],"signal":"Assessment quality","action":r["action"]})
    for _,r in cohort26[cohort26["status"]!="READY"].iterrows():
        actions.append({"priority":1 if r["status"]=="ESCALATE" else 2,"service":"26","entity":r["learner"],"signal":r["status"],"action":"Human review" if r["status"]=="ESCALATE" else "Targeted retry"})
    for _,r in concept31[concept31["priority"]=="HIGH"].iterrows():
        actions.append({"priority":2,"service":"31","entity":r["concept"],"signal":"Concept hotspot","action":"Instructional review"})
    adf=pd.DataFrame(actions).sort_values(["priority","service","entity"])
    st.dataframe(adf,use_container_width=True,hide_index=True)

    st.subheader("Cross-service consistency checks")
    checks=pd.DataFrame([
        {"Check":"Weak item ↔ concept hotspot","Meaning":"If both appear, review assessment alignment and teaching together","State":"Enabled"},
        {"Check":"Learner escalation ↔ hotspot","Meaning":"If clustered around one concept, prioritize instructional review","State":"Enabled"},
        {"Check":"Strong item ↔ mastery gain","Meaning":"Positive evidence that assessment and practice may be aligned","State":"Enabled"},
    ])
    st.dataframe(checks,use_container_width=True,hide_index=True)

# ==========================================================
# PRESENTATION MODE
# ==========================================================
elif section == "Presentation Mode":
    st.title("Presentation Mode")
    st.caption("Σύντομη customer-facing αφήγηση για live demo μέσα στο Moodle.")

    step=st.radio("Βήμα",["1 · Πρόβλημα","2 · Service 21","3 · Service 26","4 · Service 31","5 · Integration","6 · Customer value"],horizontal=True)

    if step.startswith("1"):
        st.markdown("## Τα LMS έχουν δεδομένα. Το δύσκολο είναι να μετατραπούν σε αποφάσεις.")
        st.write("Ποια ερώτηση θέλει έλεγχο; Ποιος learner χρειάζεται retry; Ποιο concept χρειάζεται παρέμβαση;")
    elif step.startswith("2"):
        st.markdown("## Service 21: από item statistics σε assessment intelligence")
        st.metric("Items analysed",len(service21))
        st.metric("Open review actions",s21_open)
        st.write("Psychometrics → priority queue → human item review.")
    elif step.startswith("3"):
        st.markdown("## Service 26: adaptive formative practice")
        st.metric("Progress","+38.75")
        st.metric("Current status","READY")
        st.write("Rubric → mastery → confidence gate → retry / enrichment / escalation.")
    elif step.startswith("4"):
        st.markdown("## Service 31: learning insights")
        st.metric("Concept hotspots",s31_open)
        st.warning("Controlled synthetic demo until operational Moodle signals are connected.")
        st.write("Concept hotspot → faculty workflow → targeted intervention.")
    elif step.startswith("5"):
        st.markdown("## Integration")
        st.write("Moodle + Streamlit analytics + native plugin + LTI-capable architecture.")
        st.markdown('<div class="flow">One control center — multiple services — explicit provenance — no automatic grade writes</div>',unsafe_allow_html=True)
    else:
        st.markdown("## Customer value")
        st.write("• Γρήγορη πρώτη εικόνα")
        st.write("• Explainable decision support")
        st.write("• Human review where needed")
        st.write("• Modular architecture")
        st.write("• Moodle-embedded presentation")
        st.write("• Clear distinction between live, controlled and synthetic evidence")

# ==========================================================
# MASTER REPORT
# ==========================================================
else:
    st.title("Master Report")
    a,b,c,d=st.columns(4)
    a.metric("Overall health",f"{overall_health:.1f}")
    b.metric("Open Service 21",s21_open)
    c.metric("Open Service 26",s26_open)
    d.metric("Open Service 31",s31_open)

    report=pd.DataFrame([
        {"Service":"21","Capability":"Assessment Intelligence","Health":round(s21_health,1),"Open actions":s21_open,"Provenance":"Controlled / live-capable"},
        {"Service":"26","Capability":"Adaptive Practice Intelligence","Health":round(s26_health,1),"Open actions":s26_open,"Provenance":"Controlled"},
        {"Service":"31","Capability":"Learning Insights","Health":round(s31_health,1),"Open actions":s31_open,"Provenance":"Synthetic demo"},
    ])
    st.dataframe(report,use_container_width=True,hide_index=True)

    summary=f"""INELSO SMART LEARNING CONTROL CENTER v4

Overall health: {overall_health:.1f}/100
Service 21 health: {s21_health:.1f}
Service 26 health: {s26_health:.1f}
Service 31 demo health: {s31_health:.1f}

Open actions:
Service 21: {s21_open}
Service 26: {s26_open}
Service 31 synthetic demo: {s31_open}

Architecture:
Moodle -> signals -> analytics -> transparent rules -> human review -> intervention/retry/revision -> audit/report

Important:
Service 31 values in this build are controlled synthetic demo values.
No official Moodle grades are automatically modified.
"""
    st.download_button("Download master summary",summary.encode("utf-8"),"INELSO_Master_v4.txt","text/plain")
