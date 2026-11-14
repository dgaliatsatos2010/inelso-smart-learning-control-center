import json
import streamlit.components.v1 as components

MOODLE_ORIGIN = "https://contentsolutions.eu"

def send_scorm_result(
    score=None,
    lesson_status="completed",
    confidence=None,
    mastery=None,
    service=None,
    entity=None,
    auto_commit=False,
):
    """Send a service result/event from Streamlit to the parent SCORM wrapper."""
    if score is not None:
        score = float(score)
        if not 0 <= score <= 100:
            raise ValueError("SCORM score must be between 0 and 100.")

    payload = {
        "type": "INELSO_SCORM_RESULT",
        "score": score,
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
        window.parent.parent.postMessage(payload, "{MOODLE_ORIGIN}");
        </script>
        """,
        height=0,
        width=0,
    )
