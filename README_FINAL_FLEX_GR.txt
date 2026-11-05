INELSO ΚΕΝΤΡΙΚΟ ΣΥΣΤΗΜΑ — ΤΕΛΙΚΗ ΕΥΕΛΙΚΤΗ ΕΚΔΟΣΗ

ΝΕΑ ΔΥΝΑΤΟΤΗΤΑ: ΕΡΓΑΣΤΗΡΙΟ ΔΕΔΟΜΕΝΩΝ

Υποστηριζόμενα αρχεία:
- CSV
- TSV
- XLSX
- XLS
- ODS
- JSON
- XML

Το εργαστήριο προσφέρει:
- αυτόματη φόρτωση και αναγνώριση δομής
- έλεγχο ποιότητας δεδομένων
- missing values / duplicates
- καθαρισμό και μετατροπή αριθμητικών στηλών
- περιγραφική στατιστική
- συχνότητες κατηγορικών μεταβλητών
- φίλτρα
- group-by aggregations
- Pearson / Spearman correlations
- εξαγωγή καθαρισμένων δεδομένων
- εξαγωγή αναφοράς ποιότητας

ΣΥΝΔΕΣΗ ΜΕ SERVICE 21
Ο χρήστης μπορεί να χαρτογραφήσει στήλες:
- ερώτηση
- facility
- discrimination
- discriminative efficiency

Το Service 21 επανυπολογίζεται με το ανεβασμένο αρχείο.

ΣΥΝΔΕΣΗ ΜΕ SERVICE 26
Ο χρήστης μπορεί να χαρτογραφήσει:
- μαθητή
- προσπάθεια
- score/grade
- confidence
- RC-01 ... RC-05

Το Service 26 επανυπολογίζεται και το Service 31 ανανεώνεται από τα νέα signals.

MOODLE DATA
Τα Moodle exports σε CSV/Excel μπορούν να φορτωθούν κανονικά.
Η εφαρμογή προσπαθεί να προτείνει αυτόματα πιθανές αντιστοιχίσεις στηλών,
αλλά ο χρήστης επιβεβαιώνει τη χαρτογράφηση πριν τροφοδοτήσει Service 21 ή 26.

DEPLOY
1. Μετονομάστε:
   inelso_master_control_center_FINAL_FLEX_GR.py
   σε:
   inelso_master_control_center.py
2. Αντικαταστήστε το υπάρχον αρχείο στο GitHub.
3. Αντικαταστήστε το requirements.txt με το requirements_FINAL_FLEX.txt.
4. Commit.
5. Το ίδιο Streamlit URL και Moodle iframe παραμένουν.

ΣΗΜΑΝΤΙΚΟ
- Οι αλλαγές/καθαρισμοί γίνονται στη μνήμη της Streamlit συνεδρίας.
- Το αρχικό αρχείο δεν τροποποιείται.
- Η εφαρμογή δεν γράφει αυτόματα πίσω στο Moodle.
- Για μόνιμη write-back λειτουργία απαιτείται Moodle API/plugin/backend.
