"""
Rule-Based Credit Risk Classification System
Code assisted by Chat-Based Generative AI
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Load datasets ---
train_df = pd.read_csv("data/training_data.csv")
test_df = pd.read_csv("data/test_data.csv")

# --- Normalize column names ---
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# --- Derive thresholds from training data ---
credit_q25 = train_df["Credit amount"].quantile(0.25)
credit_q75 = train_df["Credit amount"].quantile(0.75)
dur_q25 = train_df["Duration"].quantile(0.25)
dur_q75 = train_df["Duration"].quantile(0.75)
age_q25 = train_df["Age"].quantile(0.25)
age_q75 = train_df["Age"].quantile(0.75)

print(f"Training thresholds: Credit ({credit_q25:.0f}-{credit_q75:.0f}), "
      f"Duration ({dur_q25:.0f}-{dur_q75:.0f}), Age ({age_q25:.0f}-{age_q75:.0f})\n")

def safe_str(x):
    return str(x).strip().lower() if pd.notnull(x) else ""


# --- Identify recurring high-risk patterns ---
def identify_high_risk_patterns(df, min_count=5, risk_ratio=0.6):
    """
    Identify (Purpose, Savings, Checking) combinations with mostly 'bad' outcomes.
    Returns a dictionary of patterns and their penalty scores.
    """
    patterns = {}
    grouped = df.groupby(["Purpose", "Saving accounts", "Checking account"])
    for keys, group in grouped:
        if len(group) < min_count:
            continue
        bad_ratio = (group["Risk"].astype(str).str.lower() == "bad").mean()
        if bad_ratio >= risk_ratio:
            patterns[keys] = +0.5  # consistent +0.5 penalty for risky patterns
    return patterns


# --- Discover risky behavioral combinations ---
high_risk_patterns = identify_high_risk_patterns(train_df)
print(f"Identified {len(high_risk_patterns)} high-risk behavioral patterns:")
for k, v in high_risk_patterns.items():
    print(f"  {k}: +{v}")

# --- Rule-Based Classifier ---
def classify_risk(row):
    score = 0
    savings = safe_str(row.get('Saving accounts', ''))
    checking = safe_str(row.get('Checking account', ''))
    housing = safe_str(row.get('Housing', ''))
    purpose = safe_str(row.get('Purpose', ''))
    credit = float(row.get('Credit amount', 0))
    duration = float(row.get('Duration', 0))
    age = float(row.get('Age', 0))

    # --- Financial condition rules ---
    if (savings in ['none', 'little', '']) and (checking in ['none', 'little', '']):
        score += 1.5
    else:
        if savings in ['none', 'little', '']: score += 1
        elif savings in ['moderate', 'quite rich']: score -= 0.5
        elif savings == 'rich': score -= 1.5

        if checking in ['none', 'little', '']: score += 1
        elif checking == 'moderate': score -= 0.5
        elif checking == 'rich': score -= 1.5

    # --- Credit and duration thresholds ---
    if credit > credit_q75: score += 1.5
    elif credit < credit_q25: score -= 0.5

    if duration > dur_q75: score += 1.5
    elif duration < dur_q25: score -= 0.5

    # --- Demographic & housing factors ---
    if housing == 'own': score -= 0.5
    elif housing == 'rent': score += 0.5

    if age < age_q25: score -= 0.5
    elif age > age_q75: score += 0.5

    # --- Purpose-specific risk ---
    if any(word in purpose for word in ['radio', 'tv', 'vacation', 'car']):
        score += 0.5
    if 'education' in purpose:
        score += 0.3
    if 'business' in purpose or 'repairs' in purpose:
        score -= 0.5

    # --- Pattern Consistency Adjustment ---
    key = (purpose, savings, checking)
    if key in high_risk_patterns:
        score += high_risk_patterns[key]

    # --- Final classification ---
    if score >= 1.8:
        return 'bad'
    elif score <= -1:
        return 'good'
    else:
        return 'good' if (credit < credit_q75 and duration < dur_q75) else 'bad'

# --- Apply classifier ---
train_df["Predicted_Risk"] = train_df.apply(classify_risk, axis=1)
test_df["Predicted_Risk"] = test_df.apply(classify_risk, axis=1)

# --- Evaluate only on labeled training data ---
y_true = train_df["Risk"].astype(str).str.strip().str.lower()
y_pred = train_df["Predicted_Risk"].astype(str).str.strip().str.lower()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label='bad')
recall = recall_score(y_true, y_pred, pos_label='bad')
f1 = f1_score(y_true, y_pred, pos_label='bad')
metrics = {'Precision': precision, 'Recall': recall, 'F1-Score': f1}

print(f"\nRule-Based System Accuracy: {accuracy:.4f}\n")

print("Classification Report:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred, labels=["bad", "good"])
print("Confusion Matrix:\n", cm, "\n")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bad", "good"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Rule-Based Credit Risk Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='Blues')
plt.ylim(0, 1)
plt.title('Performance Metrics for Bad Risk Classification')
plt.ylabel('Score')
plt.show()

# --- Combine and export ---
test_df["Risk"] = "Unknown"
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.to_csv("data/predicted_data.csv", index=False)

print("\nCombined dataset saved to: predicted_data.csv")
