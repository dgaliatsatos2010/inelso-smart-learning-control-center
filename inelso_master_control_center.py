
import ast
import re
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="INELSO SCORM Live Test", layout="wide")
st.title("INELSO — Live έλεγχος πραγματικού SCORM")
st.caption("Διαβάζει υπάρχουσες SCORM προσπάθειες από Moodle και αναγνωρίζει Storyline tracking/interactions.")

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

def parse_possible_list(value):
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

def flatten_track_items(items, out):
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        element = item.get("element") or item.get("name") or item.get("track")
        value = item.get("value")

        # Moodle wrapper row: element='tracks', value=[{element,value}, ...]
        if str(element) == "tracks":
            nested = parse_possible_list(value)
            if nested is not None:
                flatten_track_items(nested, out)
            else:
                out["tracks"] = value
        elif element is not None:
            out[str(element)] = value

def flatten_moodle_track_response(track_response):
    """
    Handles all formats observed from Moodle:
    1) {"data":[{"element":"...","value":"..."}]}
    2) {"data":{"attempt":1,"tracks":[{"element":"...","value":"..."}]}}
    3) {"tracks":[...]}
    4) {"attempt":1,"tracks":[...]}
    """
    out = {}
    if not isinstance(track_response, dict):
        return out

    # Most important case found in this Moodle:
    data = track_response.get("data")
    if isinstance(data, dict):
        if "attempt" in data:
            out["attempt"] = data.get("attempt")
        nested_tracks = parse_possible_list(data.get("tracks"))
        if nested_tracks is not None:
            flatten_track_items(nested_tracks, out)

        # Also inspect any element/value pairs nested in data
        if data.get("element") is not None:
            flatten_track_items([data], out)

    elif isinstance(data, list):
        flatten_track_items(data, out)

    # Some Moodle versions return tracks at the top level
    top_tracks = parse_possible_list(track_response.get("tracks"))
    if top_tracks is not None:
        flatten_track_items(top_tracks, out)

    # Fallback: top-level element/value
    if track_response.get("element") is not None:
        flatten_track_items([track_response], out)

    # Last fallback: only scalar keys; do not overwrite parsed values
    if not out:
        for k, v in track_response.items():
            if not isinstance(v, (dict, list)):
                out[str(k)] = v

    return out

def interaction_indexes(tracks):
    indexes = set()
    for key in tracks:
        for pattern in (
            r"cmi\.interactions\.(\d+)\.",
            r"cmi\.interactions_(\d+)\.",
        ):
            m = re.match(pattern, key)
            if m:
                indexes.add(int(m.group(1)))
                break
    return sorted(indexes)

def get_interaction_value(tracks, idx, *suffixes):
    for suffix in suffixes:
        for key in (
            f"cmi.interactions.{idx}.{suffix}",
            f"cmi.interactions_{idx}.{suffix}",
        ):
            if key in tracks:
                return tracks[key]
    return None

def extract_interactions(tracks):
    rows = []
    for idx in interaction_indexes(tracks):
        rows.append({
            "interaction_index": idx,
            "question_id": get_interaction_value(tracks, idx, "id"),
            "type": get_interaction_value(tracks, idx, "type"),
            "student_response": get_interaction_value(tracks, idx, "student_response", "learner_response"),
            "correct_response": get_interaction_value(
                tracks, idx,
                "correct_responses_0.pattern",
                "correct_responses.0.pattern"
            ),
            "result": get_interaction_value(tracks, idx, "result"),
            "weighting": get_interaction_value(tracks, idx, "weighting"),
            "latency": get_interaction_value(tracks, idx, "latency"),
            "time": get_interaction_value(tracks, idx, "time"),
        })
    return pd.DataFrame(rows)

def get_score_status_rows(tracks):
    wanted = [
        "score_raw",
        "status",
        "total_time",
        "session_time",
        "cmi.core.score.raw",
        "cmi.core.score.min",
        "cmi.core.score.max",
        "cmi.core.lesson_status",
        "cmi.core.total_time",
        "cmi.completion_status",
        "cmi.success_status",
    ]
    return [{"element": k, "value": tracks[k]} for k in wanted if k in tracks]

base_url = st.text_input("Moodle URL", value=DEFAULT_MOODLE_URL)
course_id = st.number_input("Course ID", min_value=1, value=DEFAULT_COURSE_ID, step=1)

secret_token = get_secret("MOODLE_TOKEN", "moodle_token", "WSTOKEN", "wstoken")
token = st.text_input(
    "Web service token",
    value=secret_token,
    type="password",
    help="Αν υπάρχει στα Streamlit Secrets, θα συμπληρωθεί αυτόματα.",
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
            st.session_state["course_users"] = [
                {
                    "id": u.get("id"),
                    "fullname": u.get("fullname"),
                    "username": u.get("username"),
                    "email": u.get("email"),
                }
                for u in users if isinstance(u, dict)
            ]
            st.success(
                f"Βρέθηκαν {len(found)} SCORM δραστηριότητες και "
                f"{len(st.session_state['course_users'])} χρήστες."
            )
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
    selected_scorm = st.selectbox("SCORM δραστηριότητα", list(scorm_options.keys()))
    scorm_id = int(scorm_options[selected_scorm]["scorm_id"])
else:
    scorm_id = int(st.number_input("SCORM ID", min_value=1, value=4, step=1))

if users:
    st.subheader("2. Επίλεξε χρήστη με υπάρχουσα προσπάθεια")
    user_options = {
        f"{u['fullname']} — ID {u['id']}": u
        for u in users if u.get("id")
    }
    selected_user = st.selectbox("Χρήστης", list(user_options.keys()))
    user_id = int(user_options[selected_user]["id"])
else:
    user_id = int(st.number_input("Moodle User ID", min_value=1, value=7, step=1))

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
                st.warning("Δεν βρέθηκε καταγεγραμμένη προσπάθεια.")
            else:
                scoes = moodle_call(
                    base_url, token, "mod_scorm_get_scorm_scoes",
                    {"scormid": scorm_id}
                )
                sco_list = scoes.get("scoes", []) if isinstance(scoes, dict) else []

                all_raw = []
                all_interactions = []
                all_summary = []

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
                                "attempt": attempt,
                            }
                        )

                        tracks = flatten_moodle_track_response(response)

                        for element, value in tracks.items():
                            all_raw.append({
                                "attempt": attempt,
                                "sco_id": sco_id,
                                "sco_title": sco.get("title") or sco.get("name"),
                                "element": element,
                                "value": value,
                            })

                        for item in get_score_status_rows(tracks):
                            all_summary.append({
                                "attempt": attempt,
                                "sco_id": sco_id,
                                **item,
                            })

                        inter = extract_interactions(tracks)
                        if not inter.empty:
                            inter.insert(0, "attempt", attempt)
                            inter.insert(1, "sco_id", sco_id)
                            all_interactions.append(inter)

                raw_df = pd.DataFrame(all_raw)
                summary_df = pd.DataFrame(all_summary)

                st.subheader("Score / completion / status")
                if summary_df.empty:
                    st.info("Δεν βρέθηκαν score/status fields.")
                else:
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                if all_interactions:
                    interactions_df = pd.concat(all_interactions, ignore_index=True)
                    interactions_df["is_correct"] = (
                        interactions_df["result"].astype(str).str.lower().eq("correct")
                    )
                    correct_n = int(interactions_df["is_correct"].sum())
                    total_n = len(interactions_df)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ερωτήσεις", total_n)
                    c2.metric("Σωστές", correct_n)
                    c3.metric(
                        "Επιτυχία",
                        f"{(100*correct_n/total_n):.1f}%" if total_n else "—"
                    )

                    st.subheader("Πραγματικά SCORM interactions")
                    st.dataframe(interactions_df, use_container_width=True, hide_index=True)
                    st.success(
                        "ΕΠΙΤΥΧΙΑ — Το Storyline SCORM επιστρέφει πραγματικά δεδομένα "
                        "ανά ερώτηση και μπορούν να τροφοδοτήσουν την Ποιότητα Αξιολόγησης."
                    )

                    st.download_button(
                        "⬇️ Λήψη καθαρισμένων interactions CSV",
                        data=interactions_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="INELSO_real_SCORM_interactions.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Δεν εντοπίστηκαν interactions στο συγκεκριμένο attempt.")

                with st.expander("🔬 Raw SCORM tracks"):
                    st.dataframe(raw_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Σφάλμα live SCORM test: {e}")
