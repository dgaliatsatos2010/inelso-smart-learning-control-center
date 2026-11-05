INELSO ΚΕΝΤΡΙΚΟ ΣΥΣΤΗΜΑ ΕΞΥΠΝΗΣ ΜΑΘΗΣΗΣ — ULTIMATE PRO WRITE

ΝΕΑ ΕΝΟΤΗΤΑ
«Ελεγχόμενη Εγγραφή Moodle»

Υποστηριζόμενες write ενέργειες στην πρώτη έκδοση:
1. Αποστολή προσωπικού μηνύματος
   Moodle function:
   core_message_send_instant_messages

2. Ενημέρωση βαθμού — προχωρημένο
   Moodle function:
   core_grade_update_grades

ΑΣΦΑΛΕΙΑ
- Καμία write ενέργεια δεν εκτελείται αυτόματα.
- Υπάρχει προεπισκόπηση.
- Απαιτούνται ρητές επιβεβαιώσεις.
- Η αλλαγή βαθμού απαιτεί επιπλέον τη φράση:
  ΑΛΛΑΓΗ ΒΑΘΜΟΥ
- Κάθε προσπάθεια εγγραφής καταγράφεται σε audit log.
- Το token δεν πρέπει να μπει στον πηγαίο κώδικα ή σε δημόσιο GitHub.
- Για παραγωγή προτείνεται ξεχωριστό read token και write token.
- Το write service account πρέπει να έχει μόνο τις απολύτως απαραίτητες capabilities/functions.

MOODLE FUNCTIONS
Το External Service / token πρέπει να έχει ρητά ενεργοποιημένες τις functions που θα χρησιμοποιηθούν.

Για αποστολή μηνυμάτων:
- core_message_send_instant_messages
- capability moodle/site:sendmessage

Για grade updates:
- core_grade_update_grades
- απαιτούνται τα κατάλληλα grade capabilities και σωστά course/activity/component IDs.

ΣΗΜΑΝΤΙΚΟ
Η core_grade_update_grades είναι χαμηλού επιπέδου grade API.
Δοκιμή πρώτα σε test course. Το ακριβές component/activity mapping πρέπει να επαληθεύεται στο συγκεκριμένο Moodle.

DEPLOY
1. Μετονομάστε:
   inelso_master_control_center_ULTIMATE_PRO_WRITE_GR.py
   σε:
   inelso_master_control_center.py
2. Αντικαταστήστε το αρχείο στο GitHub.
3. Το requirements.txt παραμένει όπως στο ULTIMATE PRO.
4. Commit.
5. Το Streamlit URL και Moodle iframe παραμένουν ίδια.
