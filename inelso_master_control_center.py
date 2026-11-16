
import re
import io
import json
import math
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="INELSO SCORM Interaction Analyzer", layout="wide")

st.title("INELSO — Ανάλυση SCORM Αλληλεπιδράσεων")
st.caption("Αντλεί ή διαβάζει δεδομένα SCORM προσπαθειών και υπολογίζει στατιστικά ανά ερώτηση.")

with st.expander("ℹ️ Τι κάνει αυτή η εφαρμογή", expanded=False):
    st.markdown("""
    Η εφαρμογή μπορεί να δουλέψει με δύο τρόπους:

    **1. CSV export από Moodle/SCORM**
    - ανεβάζεις το αρχείο,
    - γίνεται προσπάθεια αυτόματης αναγνώρισης των πεδίων,
    - υπολογίζονται στατιστικά ανά ερώτηση.

    **2. Ζωντανά μέσω Moodle REST**
    - χρησιμοποιεί `mod_scorm_get_scorm_attempt_count`,
    - χρησιμοποιεί `mod_scorm_get_scorm_sco_tracks`,
    - ανακτά score, status και `cmi.interactions.*`,
    - δημιουργεί ενιαίο dataset.

    Τα βασικά αποτελέσματα είναι:
    - αριθμός απαντήσεων,
    - σωστές / λανθασμένες,
    - ποσοστό σωστών (facility),
    - δείκτης διάκρισης (upper 27% − lower 27%), όταν υπάρχει αρκετό δείγμα,
    - συνολικός βαθμός και κατάσταση προσπάθειας ανά εκπαιδευόμενο.
    """)

def moodle_call(base_url, token, wsfunction, params=None, timeout=30):
    params = params or {}
    url = base_url.rstrip("/") + "/webservice/rest/server.php"
    payload = {
        "wstoken": token,
        "wsfunction": wsfunction,
        "moodlewsrestformat": "json",
        **params,
    }
    r = requests.get(url, params=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("exception"):
        raise RuntimeError(data.get("message", data["exception"]))
    return data

def normalize_track_rows(user_id, attempt, sco_data):
    rows = []
    if not isinstance(sco_data, dict):
        return rows

    data = sco_data.get("data", sco_data.get("tracks", sco_data.get("trackdata", [])))
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = []
        for obj in data:
            if isinstance(obj, dict):
                k = obj.get("element") or obj.get("name") or obj.get("track")
                v = obj.get("value")
                if k is not None:
                    items.append((k, v))
    else:
        items = []

    tracks = {str(k): v for k, v in items}

    score = None
    status = None
    for key in ("cmi.core.score.raw", "cmi.score.raw"):
        if key in tracks:
            score = tracks[key]
            break
    for key in ("cmi.core.lesson_status", "cmi.completion_status", "cmi.success_status"):
        if key in tracks:
            status = tracks[key]
            break

    interaction_ids = {}
    for key, value in tracks.items():
        m = re.match(r"cmi\.interactions\.(\d+)\.id$", key)
        if m:
            interaction_ids[int(m.group(1))] = value

    for idx, qid in sorted(interaction_ids.items()):
        def get_any(*suffixes):
            for suffix in suffixes:
                k = f"cmi.interactions.{idx}.{suffix}"
                if k in tracks:
                    return tracks[k]
            return None

        rows.append({
            "user_id": user_id,
            "attempt": attempt,
            "question_id": qid,
            "student_response": get_any("student_response", "learner_response"),
            "result": get_any("result"),
            "interaction_type": get_any("type"),
            "weighting": get_any("weighting"),
            "latency": get_any("latency"),
            "attempt_score": score,
            "lesson_status": status,
        })
    return rows

def fetch_scorm_interactions(base_url, token, scorm_id, user_ids):
    all_rows = []
    progress = st.progress(0)
    total = max(len(user_ids), 1)

    scoes = moodle_call(
        base_url,
        token,
        "mod_scorm_get_scorm_scoes",
        {"scormid": scorm_id},
    )
    sco_list = scoes.get("scoes", []) if isinstance(scoes, dict) else []

    for i, user_id in enumerate(user_ids, start=1):
        count_data = moodle_call(
            base_url,
            token,
            "mod_scorm_get_scorm_attempt_count",
            {"scormid": scorm_id, "userid": user_id},
        )
        if isinstance(count_data, dict):
            attempts = int(count_data.get("attemptscount", count_data.get("attempts", 0)) or 0)
        else:
            attempts = 0

        for attempt in range(1, attempts + 1):
            for sco in sco_list:
                sco_id = sco.get("id")
                if not sco_id:
                    continue
                track_data = moodle_call(
                    base_url,
                    token,
                    "mod_scorm_get_scorm_sco_tracks",
                    {"scoid": sco_id, "userid": user_id, "attempt": attempt},
                )
                all_rows.extend(normalize_track_rows(user_id, attempt, track_data))
        progress.progress(i / total)

    progress.empty()
    return pd.DataFrame(all_rows)

def canonicalize_uploaded(df):
    cols = {c.lower().strip(): c for c in df.columns}

    def first_match(candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        for low, original in cols.items():
            if any(cand in low for cand in candidates):
                return original
        return None

    mapping = {
        "user_id": first_match(["user_id", "userid", "learner", "student", "user"]),
        "attempt": first_match(["attempt", "try"]),
        "question_id": first_match(["question_id", "interaction_id", "question", "item"]),
        "student_response": first_match(["student_response", "learner_response", "response", "answer"]),
        "result": first_match(["result", "correctness", "status"]),
        "attempt_score": first_match(["attempt_score", "score", "grade"]),
        "lesson_status": first_match(["lesson_status", "completion", "success"]),
    }

    out = pd.DataFrame()
    for target, source in mapping.items():
        out[target] = df[source] if source is not None else pd.NA
    return out, mapping

def mark_correct(result):
    if pd.isna(result):
        return pd.NA
    s = str(result).strip().lower()
    if s in {"correct", "right", "true", "1", "passed", "pass"}:
        return 1
    if s in {"wrong", "incorrect", "false", "0", "failed", "fail"}:
        return 0
    try:
        val = float(s)
        if 0 <= val <= 1:
            return 1 if val >= 0.999 else 0
    except Exception:
        pass
    return pd.NA

def item_statistics(df):
    x = df.copy()
    x["correct"] = x["result"].apply(mark_correct)
    x["attempt_score_num"] = pd.to_numeric(x["attempt_score"], errors="coerce")

    rows = []
    for qid, g in x.groupby("question_id", dropna=False):
        valid = g["correct"].dropna()
        n = int(valid.shape[0])
        correct_n = int(valid.sum()) if n else 0
        facility = float(valid.mean() * 100) if n else math.nan

        disc = math.nan
        eligible = g.dropna(subset=["correct", "attempt_score_num"]).copy()
        if len(eligible) >= 8:
            eligible = eligible.sort_values("attempt_score_num")
            k = max(1, int(round(len(eligible) * 0.27)))
            low = eligible.head(k)["correct"].mean()
            high = eligible.tail(k)["correct"].mean()
            disc = float((high - low) * 100)

        rows.append({
            "question_id": qid,
            "responses_n": n,
            "correct_n": correct_n,
            "incorrect_n": n - correct_n,
            "facility_pct": round(facility, 2) if not math.isnan(facility) else pd.NA,
            "discrimination_pct": round(disc, 2) if not math.isnan(disc) else pd.NA,
        })
    return pd.DataFrame(rows)

tab1, tab2 = st.tabs(["📄 CSV / Excel", "🔗 Moodle REST"])

interaction_df = None

with tab1:
    up = st.file_uploader(
        "Ανέβασε export SCORM",
        type=["csv", "xlsx", "xls"],
        help="Μπορεί να είναι export με attempts/interactions. Η εφαρμογή θα προσπαθήσει να αναγνωρίσει αυτόματα τις στήλες.",
    )
    if up is not None:
        raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        interaction_df, mapping = canonicalize_uploaded(raw)
        st.success(f"Φορτώθηκαν {len(raw)} γραμμές.")
        with st.expander("Αυτόματη αντιστοίχιση στηλών"):
            st.json({k: (v if v else "—") for k, v in mapping.items()})

with tab2:
    st.info(
        "Για live ανάκτηση πρέπει στο Moodle Web Service να είναι ενεργές οι SCORM functions "
        "`mod_scorm_get_scorm_attempt_count`, `mod_scorm_get_scorm_scoes` και "
        "`mod_scorm_get_scorm_sco_tracks`."
    )
    base_url = st.text_input("Moodle URL", value="https://contentsolutions.eu")
    token = st.text_input("Web service token", type="password")
    scorm_id = st.number_input("SCORM ID", min_value=1, step=1, value=1)
    user_ids_text = st.text_area("User IDs", placeholder="208, 209, 210")

    if st.button("Ανάκτηση SCORM δεδομένων", type="primary"):
        try:
            user_ids = [
                int(x.strip()) for x in user_ids_text.replace("\n", ",").split(",")
                if x.strip()
            ]
            if not token or not user_ids:
                st.error("Χρειάζεται token και τουλάχιστον ένα User ID.")
            else:
                interaction_df = fetch_scorm_interactions(base_url, token, int(scorm_id), user_ids)
                if interaction_df.empty:
                    st.warning("Δεν βρέθηκαν cmi.interactions για τα επιλεγμένα attempts.")
                else:
                    st.session_state["scorm_interaction_df"] = interaction_df
                    st.success(f"Ανακτήθηκαν {len(interaction_df)} interactions.")
        except Exception as e:
            st.error(f"Σφάλμα Moodle REST: {e}")

if interaction_df is None and "scorm_interaction_df" in st.session_state:
    interaction_df = st.session_state["scorm_interaction_df"]

if interaction_df is not None and not interaction_df.empty:
    st.divider()
    st.subheader("SCORM Interactions")
    st.dataframe(interaction_df, use_container_width=True, hide_index=True)

    stats = item_statistics(interaction_df)

    st.subheader("Στατιστικά ανά ερώτηση")
    st.dataframe(stats, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Ερωτήσεις", int(stats["question_id"].nunique()))
    c2.metric("Interactions", len(interaction_df))
    c3.metric(
        "Μέση επιτυχία",
        f"{stats['facility_pct'].dropna().mean():.1f}%"
        if stats["facility_pct"].notna().any()
        else "—",
    )

    st.download_button(
        "⬇️ Λήψη στατιστικών CSV",
        data=stats.to_csv(index=False).encode("utf-8-sig"),
        file_name="INELSO_SCORM_question_statistics.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Λήψη καθαρισμένων interactions CSV",
        data=interaction_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="INELSO_SCORM_interactions_clean.csv",
        mime="text/csv",
    )

    with st.expander("ℹ️ Ερμηνεία δεικτών"):
        st.markdown("""
        - **Facility %**: ποσοστό σωστών απαντήσεων. Υψηλή τιμή = ευκολότερη ερώτηση.
        - **Discrimination %**: διαφορά επιτυχίας μεταξύ ανώτερου και κατώτερου 27%,
          βάσει συνολικού score. Υπολογίζεται μόνο όταν υπάρχουν τουλάχιστον 8 έγκυρες παρατηρήσεις.
        - Για επίσημη ψυχομετρική αξιολόγηση χρειάζεται επαρκές δείγμα και σταθερό scoring.
        """)
else:
    st.caption("Δεν υπάρχουν ακόμη SCORM interaction δεδομένα για ανάλυση.")
