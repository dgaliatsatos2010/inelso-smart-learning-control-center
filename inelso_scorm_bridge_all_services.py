import json
import streamlit.components.v1 as components

BRIDGE_TOKEN = "INELSO_SCORM_BRIDGE_V5_2026"

def send_scorm_result(score=None, lesson_status="completed", confidence=None,
                      mastery=None, service=None, entity=None, auto_commit=False):
    if score is not None:
        score=float(score)
        if not 0 <= score <= 100:
            raise ValueError("SCORM score must be between 0 and 100.")
    payload={
        "type":"INELSO_SCORM_RESULT",
        "bridge_token":BRIDGE_TOKEN,
        "score":score,
        "lesson_status":lesson_status,
        "confidence":confidence,
        "mastery":mastery,
        "service":service,
        "entity":entity,
        "auto_commit":bool(auto_commit),
    }
    payload_json=json.dumps(payload, ensure_ascii=False)
    components.html(f"""
    <script>
    const payload = {payload_json};
    const targets=[];
    try{targets.push(window.parent);}catch(e){}
    try{targets.push(window.parent.parent);}catch(e){}
    try{targets.push(window.parent.parent.parent);}catch(e){}
    try{targets.push(window.top);}catch(e){}
    for(const t of targets){try{t.postMessage(payload,"*");}catch(e){}}
    </script>
    """, height=0, width=0)
