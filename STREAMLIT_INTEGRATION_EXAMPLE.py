import streamlit as st
from inelso_scorm_bridge import send_scorm_result

# Replace these with your real Service 26 variables:
current_score = 100
confidence = 0.85
mastery = "Κατάκτηση"
learner_name = "Demo Student 0201"

if st.button("Αποστολή τρέχοντος αποτελέσματος στο SCORM"):
    status = "completed" if current_score >= 70 else "incomplete"
    send_scorm_result(
        score=current_score,
        lesson_status=status,
        confidence=confidence,
        mastery=mastery,
        service="Service 26",
        entity=learner_name,
        auto_commit=False,
    )
    st.success("Το αποτέλεσμα στάλθηκε στον SCORM wrapper.")
