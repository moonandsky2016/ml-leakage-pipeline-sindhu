# -------------------------------
# Task 1 — Leakage Example (Wrong Approach)
# -------------------------------

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# ❌ WRONG: Scaling before split (causes data leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split after scaling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Task 1 — Leakage Approach")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy :", accuracy_score(y_test, y_test_pred))


# -------------------------------
# Task 2 — Correct Pipeline Approach
# -------------------------------

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

# Proper split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("\nTask 2 — Pipeline + Cross Validation")
print("Mean Accuracy:", np.mean(cv_scores))
print("Std Dev      :", np.std(cv_scores))


# -------------------------------
# Task 3 — Decision Tree Depth
# -------------------------------

from sklearn.tree import DecisionTreeClassifier

depths = [1, 5, 20]

print("\nTask 3 — Decision Tree Results")

results = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    results.append((depth, train_acc, test_acc))
    print(f"Depth={depth} | Train={train_acc:.3f} | Test={test_acc:.3f}")