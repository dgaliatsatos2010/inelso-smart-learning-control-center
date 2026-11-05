INELSO SMART LEARNING CONTROL CENTER v4

Major upgrades
- Executive Command Center
- Service health scores
- Upgraded Service 21 assessment intelligence
- Upgraded Service 26 adaptive practice intelligence
- Functional Service 31 controlled synthetic demo
- Cross-Service Intelligence / evidence chain
- Unified action queue
- Presentation Mode for Moodle/customer demo
- Explicit provenance labels: LIVE-capable / CONTROLLED / SYNTHETIC
- Master Report

DEPLOY
1. In GitHub repository inelso-smart-learning-control-center
2. Replace existing inelso_master_control_center.py with this file
3. Keep repository filename exactly:
   inelso_master_control_center.py
4. requirements.txt:
   streamlit>=1.40
   pandas>=2.0
   numpy>=1.26
5. Commit and wait for Streamlit redeploy.
6. Moodle embed URL remains:
   https://inelso-control-center.streamlit.app/?embed=true

IMPORTANT
Service 31 values in this build are synthetic demo values and are explicitly labelled as such.
