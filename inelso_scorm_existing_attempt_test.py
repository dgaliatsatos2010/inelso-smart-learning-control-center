
import re
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="INELSO SCORM Live Test", layout="wide")
st.title("INELSO — Live έλεγχος πραγματικού SCORM")
st.caption("Χρησιμοποιεί υπάρχουσες SCORM προσπάθειες. Δεν χρειάζεται νέα προσπάθεια.")

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
    help="Αν το token υπάρχει στα Streamlit Secrets, θα συμπληρωθεί αυτόματα.",
)

if st.button("🔎 Αυτόματος εντοπισμός SCORM και χρηστών", type="primary"):
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

            users = moodle_call(
                base_url, token, "core_enrol_get_enrolled_users",
                {"courseid": int(course_id)}
            )
            user_rows = []
            for u in users if isinstance(users, list) else []:
                user_rows.append({
                    "id": u.get("id"),
                    "fullname": u.get("fullname"),
                    "username": u.get("username"),
                    "email": u.get("email"),
                })
            st.session_state["course_users"] = user_rows

            st.success(f"Βρέθηκαν {len(found)} SCORM δραστηριότητες και {len(user_rows)} χρήστες.")
        except Exception as e:
            st.error(str(e))

found = st.session_state.get("scorm_found", [])
users = st.session_state.get("course_users", [])

if found:
    st.subheader("1. Επίλεξε SCORM")
    st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
    scorm_options = {
        f"{x['name']} — SCORM ID {x['scorm_id']}": x
        for x in found
    }
    scorm_label = st.selectbox("SCORM δραστηριότητα", list(scorm_options.keys()))
    scorm_id = int(scorm_options[scorm_label]["scorm_id"])
else:
    scorm_id = int(st.number_input("SCORM ID", min_value=1, value=1, step=1))

if users:
    st.subheader("2. Επίλεξε τον χρήστη που έχει ήδη κάνει την προσπάθεια")
    user_options = {
        f"{u['fullname']} — ID {u['id']}": u
        for u in users if u.get("id")
    }
    user_label = st.selectbox("Χρήστης", list(user_options.keys()))
    user_id = int(user_options[user_label]["id"])
else:
    user_id = int(st.number_input("Moodle User ID", min_value=1, value=1, step=1))

if st.button("📥 Ανάκτηση υπάρχουσας προσπάθειας και interactions"):
    if not token:
        st.error("Δεν βρέθηκε Web Service token.")
    else:
        try:
            count = moodle_call(
                base_url, token, "mod_scorm_get_scorm_attempt_count",
                {"scormid": scorm_id, "userid": user_id}
            )
            attempts = int(count.get("attemptscount", count.get("attempts", 0)) or 0)

            c1, c2, c3 = st.columns(3)
            c1.metric("SCORM ID", scorm_id)
            c2.metric("User ID", user_id)
            c3.metric("Καταγεγραμμένες προσπάθειες", attempts)

            if attempts == 0:
                st.warning("Για αυτόν τον χρήστη δεν βρέθηκε καταγεγραμμένη προσπάθεια.")
            else:
                scoes = moodle_call(
                    base_url, token, "mod_scorm_get_scorm_scoes",
                    {"scormid": scorm_id}
                )
                sco_list = scoes.get("scoes", []) if isinstance(scoes, dict) else []

                all_tracks = []
                all_interactions = []

                for attempt in range(1, attempts + 1):
                    for sco in sco_list:
                        sco_id = sco.get("id")
                        if not sco_id:
                            continue

                        response = moodle_call(
                            base_url, token, "mod_scorm_get_scorm_sco_tracks",
                            {
                                "scoid": int(sco_id),
                                "userid": int(user_id),
                                "attempt": attempt
                            }
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
                    st.subheader("Score / completion / status")
                    status_mask = tracks_df["element"].astype(str).str.contains(
                        r"score|lesson_status|completion_status|success_status",
                        case=False,
                        regex=True
                    )
                    status_df = tracks_df[status_mask]
                    if status_df.empty:
                        st.info("Δεν βρέθηκαν standard score/status fields.")
                    else:
                        st.dataframe(status_df, use_container_width=True, hide_index=True)

                if all_interactions:
                    interactions_df = pd.concat(all_interactions, ignore_index=True)
                    st.session_state["interactions_df"] = interactions_df
                    st.subheader("Πραγματικά SCORM interactions")
                    st.dataframe(interactions_df, use_container_width=True, hide_index=True)

                    st.success(
                        "ΕΠΙΤΥΧΙΑ — Το Moodle επιστρέφει πραγματικά δεδομένα ανά ερώτηση. "
                        "Μπορούν πλέον να περάσουν στην Ποιότητα Αξιολόγησης του INELSO."
                    )
                else:
                    st.warning(
                        "Βρέθηκε η υπάρχουσα προσπάθεια, αλλά δεν επιστράφηκαν "
                        "cmi.interactions.*. Άνοιξε τα raw tracks παρακάτω για να δούμε "
                        "πώς ακριβώς τα αποθηκεύει το συγκεκριμένο SCORM."
                    )

        except Exception as e:
            st.error(f"Σφάλμα live SCORM test: {e}")

tracks_df = st.session_state.get("tracks_df")
if isinstance(tracks_df, pd.DataFrame) and not tracks_df.empty:
    st.divider()
    with st.expander("🔬 Raw SCORM tracks"):
        st.dataframe(tracks_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Λήψη raw SCORM tracks",
        data=tracks_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="INELSO_real_SCORM_tracks.csv",
        mime="text/csv",
    )

interactions_df = st.session_state.get("interactions_df")
if isinstance(interactions_df, pd.DataFrame) and not interactions_df.empty:
    st.download_button(
        "⬇️ Λήψη SCORM interactions",
        data=interactions_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="INELSO_real_SCORM_interactions.csv",
        mime="text/csv",
    )
