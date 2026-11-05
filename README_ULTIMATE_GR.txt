INELSO ΚΕΝΤΡΙΚΟ ΣΥΣΤΗΜΑ ΕΞΥΠΝΗΣ ΜΑΘΗΣΗΣ — ULTIMATE ΕΚΔΟΣΗ

ΝΕΕΣ ΔΥΝΑΤΟΤΗΤΕΣ

1. INELSO Analytics Lab
- Αυτόματη διάγνωση dataset
- Δείκτης ποιότητας δεδομένων
- Συσχετίσεις
- Classification
  * Logistic Regression
  * Random Forest
- Regression
  * Linear Regression
  * Random Forest Regression
- Clustering
  * K-means
  * Silhouette score
- Feature importance
- Σύγκριση ομάδων
- Αυτόματη αναφορά

2. Προσομοιωτής What-if
- Αλλαγή confidence threshold
- Αλλαγή score threshold
- Άμεση επανεκτίμηση READY / RETRY / ESCALATE
- Καμία αλλαγή σε Moodle ή επίσημους βαθμούς

3. Κέντρο Εισαγωγής Δεδομένων
- CSV
- XLSX
- XLS
- JSON
- XML
- Moodle exports
- Καθαρισμός
- Data-quality report
- Service 21/26 mapping

4. Παραμένουν
- Service 21
- Service 26
- Service 31
- Cross-service intelligence
- Action lifecycle
- Methodology & provenance
- Readiness checks
- Presentation mode
- Master report

REQUIREMENTS
streamlit>=1.40
pandas>=2.0
numpy>=1.26
openpyxl>=3.1
xlrd>=2.0
scikit-learn>=1.4

DEPLOY
1. Μετονομάστε το αρχείο σε:
   inelso_master_control_center.py
2. Αντικαταστήστε το υπάρχον Python αρχείο στο GitHub.
3. Αντικαταστήστε το requirements.txt με requirements_ULTIMATE.txt.
4. Commit.
5. Το υπάρχον Streamlit URL και Moodle iframe παραμένουν ίδια.

ΣΗΜΑΝΤΙΚΟ
Οι ML αναλύσεις είναι γενικής χρήσης exploratory analyses.
Η σωστή επιστημονική χρήση εξαρτάται από το ερευνητικό ερώτημα, το sampling, το leakage control,
τη χρονική δομή, την εξωτερική επικύρωση και άλλες μεθοδολογικές απαιτήσεις που δεν μπορούν να
συναχθούν αυτόματα μόνο από ένα ανεβασμένο dataset.
