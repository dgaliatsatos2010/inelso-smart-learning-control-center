
import re
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="INELSO SCORM Live Test", layout="wide")
st.title("INELSO — Live έλεγχος πραγματικού SCORM")
st.caption("Εντοπίζει SCORM στο μάθημα, διαβάζει attempts και εμφανίζει τα πραγματικά cmi.interactions που έχει αποθηκεύσει το Moodle.")

DEFAULT_MOODLE_URL = "https://contentsolutions.eu"
DEFAULT_COURSE_ID = 6

def get_secret(*keys):
    for key in keys:
        try:
            if key in st.secrets and st.secrets[key]:
                return str(st.secrets[key])
        except Exception:
            pass
    return ""

def moodle_call(base_url, token, function, params=None):
    params = params or {}
    url = base_url.rstrip("/") + "/webservice/rest/server.php"
    payload = {
        "wstoken": token,
        "wsfunction": function,
        "moodlewsrestformat": "json",
        **params,
    }
    r = requests.get(url, params=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("exception"):
        raise RuntimeError(f"{data.get('errorcode','Moodle error')}: {data.get('message', data)}")
    return data

def tracks_to_dict(track_response):
    if not isinstance(track_response, dict):
        return {}
    for key in ("data", "tracks", "trackdata"):
        data = track_response.get(key)
        if isinstance(data, list):
            out = {}
            for item in data:
                if isinstance(item, dict):
                    element = item.get("element") or item.get("name") or item.get("track")
                    if element is not None:
                        out[str(element)] = item.get("value")
            if out:
                return out
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
    return {}

def extract_interactions(tracks):
    indexes = set()
    for key in tracks:
        m = re.match(r"cmi\.interactions\.(\d+)\.", key)
        if m:
            indexes.add(int(m.group(1)))
    rows = []
    for idx in sorted(indexes):
        def val(*suffixes):
            for suffix in suffixes:
                k = f"cmi.interactions.{idx}.{suffix}"
                if k in tracks:
                    return tracks[k]
            return None
        rows.append({
            "interaction_index": idx,
            "question_id": val("id"),
            "type": val("type"),
            "student_response": val("student_response", "learner_response"),
            "result": val("result"),
            "weighting": val("weighting"),
            "latency": val("latency"),
            "time": val("time"),
        })
    return pd.DataFrame(rows)

base_url = st.text_input("Moodle URL", value=DEFAULT_MOODLE_URL)
course_id = st.number_input("Course ID", min_value=1, value=DEFAULT_COURSE_ID, step=1)

secret_token = get_secret("MOODLE_TOKEN", "moodle_token", "WSTOKEN", "wstoken")
token = st.text_input(
    "Web service token",
    value=secret_token,
    type="password",
    help="Αν το token υπάρχει ήδη στα Streamlit Secrets, θα συμπληρωθεί αυτόματα.",
)

st.markdown("### 1. Εντοπισμός SCORM στο μάθημα")
if st.button("🔎 Βρες τα SCORM του μαθήματος", type="primary"):
    if not token:
        st.error("Δεν βρέθηκε Web Service token.")
    else:
        try:
            contents = moodle_call(
                base_url, token, "core_course_get_contents",
                {"courseid": int(course_id)}
            )
            found = []
            for section in contents:
                for mod in section.get("modules", []):
                    if str(mod.get("modname", "")).lower() == "scorm":
                        found.append({
                            "section": section.get("name"),
                            "name": mod.get("name"),
                            "cmid": mod.get("id"),
                            "scorm_id": mod.get("instance"),
                            "url": mod.get("url"),
                        })
            st.session_state["scorm_found"] = found
            if found:
                st.success(f"Βρέθηκαν {len(found)} SCORM δραστηριότητες.")
            else:
                st.warning("Δεν βρέθηκε SCORM στο συγκεκριμένο μάθημα.")
        except Exception as e:
            st.error(str(e))

found = st.session_state.get("scorm_found", [])
if found:
    st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
    options = {
        f"{x['name']} — SCORM ID {x['scorm_id']} (cmid {x['cmid']})": x
        for x in found
    }
    selected = st.selectbox("Επίλεξε SCORM", list(options.keys()))
    scorm_id = int(options[selected]["scorm_id"])
else:
    scorm_id = int(st.number_input("SCORM ID (αν το ξέρεις ήδη)", min_value=1, value=1, step=1))

st.markdown("### 2. Έλεγχος πραγματικής προσπάθειας")
user_id = st.number_input("Moodle User ID", min_value=1, value=208, step=1)

if st.button("📥 Διάβασε attempts και interactions"):
    if not token:
        st.error("Δεν βρέθηκε Web Service token.")
    else:
        try:
            count = moodle_call(
                base_url, token, "mod_scorm_get_scorm_attempt_count",
                {"scormid": scorm_id, "userid": int(user_id)}
            )
            attempts = int(count.get("attemptscount", count.get("attempts", 0)) or 0)
            st.write("Attempt count response:")
            st.json(count)

            if attempts == 0:
                st.warning("Δεν υπάρχει καταγεγραμμένη προσπάθεια για αυτόν τον χρήστη.")
            else:
                st.success(f"Βρέθηκαν {attempts} προσπάθειες.")
                scoes = moodle_call(
                    base_url, token, "mod_scorm_get_scorm_scoes",
                    {"scormid": scorm_id}
                )
                sco_list = scoes.get("scoes", []) if isinstance(scoes, dict) else []
                st.write(f"SCOs: {len(sco_list)}")

                all_tracks = []
                all_interactions = []

                for attempt in range(1, attempts + 1):
                    for sco in sco_list:
                        sco_id = sco.get("id")
                        if not sco_id:
                            continue
                        response = moodle_call(
                            base_url, token, "mod_scorm_get_scorm_sco_tracks",
                            {"scoid": int(sco_id), "userid": int(user_id), "attempt": attempt}
                        )
                        tracks = tracks_to_dict(response)

                        for element, value in tracks.items():
                            all_tracks.append({
                                "attempt": attempt,
                                "sco_id": sco_id,
                                "sco_title": sco.get("title") or sco.get("name"),
                                "element": element,
                                "value": value,
                            })

                        inter = extract_interactions(tracks)
                        if not inter.empty:
                            inter.insert(0, "attempt", attempt)
                            inter.insert(1, "sco_id", sco_id)
                            all_interactions.append(inter)

                tracks_df = pd.DataFrame(all_tracks)
                st.session_state["tracks_df"] = tracks_df

                if not tracks_df.empty:
                    score_keys = [
                        "cmi.core.score.raw",
                        "cmi.score.raw",
                        "cmi.core.lesson_status",
                        "cmi.completion_status",
                        "cmi.success_status",
                    ]
                    summary = tracks_df[tracks_df["element"].isin(score_keys)]
                    st.subheader("Score / completion / status")
                    if summary.empty:
                        st.info("Δεν βρέθηκαν standard score/status fields.")
                    else:
                        st.dataframe(summary, use_container_width=True, hide_index=True)

                if all_interactions:
                    interactions_df = pd.concat(all_interactions, ignore_index=True)
                    st.session_state["interactions_df"] = interactions_df
                    st.subheader("Πραγματικά SCORM interactions")
                    st.dataframe(interactions_df, use_container_width=True, hide_index=True)
                    st.success("ΕΠΙΤΥΧΙΑ: Το Moodle επιστρέφει δεδομένα ανά ερώτηση.")
                else:
                    st.warning("Βρέθηκε προσπάθεια, αλλά δεν επιστράφηκαν cmi.interactions.*.")

        except Exception as e:
            st.error(f"Σφάλμα live SCORM test: {e}")

if "tracks_df" in st.session_state and not st.session_state["tracks_df"].empty:
    with st.expander("🔬 Όλα τα raw SCORM tracks"):
        st.dataframe(st.session_state["tracks_df"], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Εξαγωγή raw tracks σε CSV",
        data=st.session_state["tracks_df"].to_csv(index=False).encode("utf-8-sig"),
        file_name="INELSO_real_SCORM_tracks.csv",
        mime="text/csv",
    )

if "interactions_df" in st.session_state and not st.session_state["interactions_df"].empty:
    st.download_button(
        "⬇️ Εξαγωγή interactions σε CSV",
        data=st.session_state["interactions_df"].to_csv(index=False).encode("utf-8-sig"),
        file_name="INELSO_real_SCORM_interactions.csv",
        mime="text/csv",
    )
