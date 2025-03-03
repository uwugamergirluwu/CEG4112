# Import required libraries
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess dataset
# Load 'emotion' dataset from HuggingFace
dataset = load_dataset("emotion", split="train")

# Convert pandas DataFrame for easier handling
df = pd.DataFrame(dataset)

# Extract features (text) and labels (emotion)
X = df["text"]
y = df["label"]

# Split into training and test sets (80% training, 20% test) to avoid train-test leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training the model
# Initialize LogisticRegression for multi-class classification
model = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)

# Train the model using fit()
model.fit(X_train_tfidf, y_train)

# Evaluating the model
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")

# Detailed classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=dataset.features["label"].names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.features["label"].names, yticklabels=dataset.features["label"].names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
plt.savefig("confusion_matrix.png")