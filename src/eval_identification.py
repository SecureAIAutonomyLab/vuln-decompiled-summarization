import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Extract pred and gt arrays
def extract_arrays(data):
    pred = data.get("pred", [])
    gt = data.get("gt", [])
    return pred, gt

# Calculate metrics
def calculate_metrics(pred, gt):
    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred, pos_label="YES")
    recall = recall_score(gt, pred, pos_label="YES")
    f1 = f1_score(gt, pred, pos_label="YES")
    return accuracy, precision, recall, f1

# Calculate accuracy for positive and negative ground truth records
def calculate_accuracy_by_label(pred, gt, label):
    filtered_pred = [p for p, g in zip(pred, gt) if g == label]
    filtered_gt = [g for g in gt if g == label]
    if not filtered_gt:
        return 0.0  # Avoid division by zero if there are no records with the given label
    accuracy = accuracy_score(filtered_gt, filtered_pred)
    return accuracy

# Main function
def main(file_path):
    data = load_json(file_path)
    pred, gt = extract_arrays(data)
    
    if not pred or not gt or len(pred) != len(gt):
        raise ValueError("Pred and GT arrays must be non-empty and of the same length.")
    
    accuracy, precision, recall, f1 = calculate_metrics(pred, gt)
    
    accuracy_yes = calculate_accuracy_by_label(pred, gt, "YES")
    accuracy_no = calculate_accuracy_by_label(pred, gt, "NO")
    
    print(f"Overall Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy for 'YES' ground truth: {accuracy_yes:.2f}")
    print(f"Accuracy for 'NO' ground truth: {accuracy_no:.2f}")

# Example usage
if __name__ == "__main__":
    file_path = '../../Experiments_results/results_x86codellama_identification.json'
    main(file_path)
