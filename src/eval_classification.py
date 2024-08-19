import sys
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load the JSON data
with open(sys.argv[1]) as f:
    data = json.load(f)

# Extract predictions and ground truth
y_pred = data['pred']
y_true = data['gt']

# Calculate overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate precision, recall, and F1 score for each class
unique_labels = sorted(list(set(y_true + y_pred)))
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=unique_labels)

# Print the precision, recall, and F1 score for each CWE code
for cwe_code, p, r, f in zip(unique_labels, precision, recall, f1):
    print(f'CWE Code: {cwe_code}')
    print(f'  Precision: {p:.4f}')
    print(f'  Recall: {r:.4f}')
    print(f'  F1 Score: {f:.4f}')

# Optional: If you want to see a classification report
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, labels=unique_labels)
print('\nClassification Report:')
print(report)
