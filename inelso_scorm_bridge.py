import json
import streamlit.components.v1 as components

MOODLE_ORIGIN = "https://contentsolutions.eu"

def send_scorm_result(score, lesson_status="completed", confidence=None,
                      mastery=None, service="Service 26", entity=None,
                      auto_commit=False):
    payload = {
        "type": "INELSO_SCORM_RESULT",
        "score": float(score),
        "lesson_status": lesson_status,
        "confidence": confidence,
        "mastery": mastery,
        "service": service,
        "entity": entity,
        "auto_commit": bool(auto_commit),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    components.html(
        f"""
        <script>
        const payload = {payload_json};
        window.parent.parent.postMessage(payload, "https://contentsolutions.eu");
        </script>
        """,
        height=0,
        width=0,
    )
