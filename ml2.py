import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

#plotting
plt.figure(figsize=(3,4))
sns.countplot(x=y,palette="Set2")
plt.title("target class distribution")
plt.xlabel("target class")
plt.ylabel("count")
plt.show()

#correlation matrix visulaization
plt.figure(figsize=(12,8))
df=pd.DataFrame(X,columns=data.feature_names)
corr_matrix=df.corr()
sns.heatmap(corr_matrix,cmap="coolwarm")
plt.title("correlation matrix")
plt.show()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare data for MultinomialNB(non-negative)
X_shifted = X_scaled - X_scaled.min(axis=0)  # shift to non-negative

# Prepare data for BernoulliNB by binarizing features (threshold at 0)
X_binary = (X_scaled > 0).astype(int)

# Split dataset
X_train_g, X_test_g, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # GaussianNB
X_train_m, X_test_m, _, _ = train_test_split(X_shifted, y, test_size=0.2, random_state=42)           # MultinomialNB
X_train_b, X_test_b, _, _ = train_test_split(X_binary, y, test_size=0.2, random_state=42)            # BernoulliNB

# Initialize models
model_gaussian = GaussianNB()
model_multinomial = MultinomialNB()
model_bernoulli = BernoulliNB()

# Train models
model_gaussian.fit(X_train_g, y_train)
model_multinomial.fit(X_train_m, y_train)
model_bernoulli.fit(X_train_b, y_train)

# Predict
y_pred_gaussian = model_gaussian.predict(X_test_g)
y_pred_multinomial = model_multinomial.predict(X_test_m)
y_pred_bernoulli = model_bernoulli.predict(X_test_b)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"---- {model_name} ----")
    print("Confusion Matrix:")
    cm=confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=data.target_names,yticklabels=data.target_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

# Evaluate all
evaluate_model(y_test, y_pred_gaussian, "GaussianNB")
evaluate_model(y_test, y_pred_multinomial, "MultinomialNB")
evaluate_model(y_test, y_pred_bernoulli, "BernoulliNB")




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
data = load_digits()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train model
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# Plot decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt, filled=True, feature_names=data.feature_names if hasattr(data, 'feature_names') else None)
plt.show()

# Make predictions and evaluate
y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

# Load dataset
data = load_digits()
X, y = data.data, data.target

# Show class distribution
sns.countplot(x=y, palette="Set3")
plt.title("Digit Class Distribution")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)  # limiting depth for readable visualization
rf_model = RandomForestClassifier(random_state=42)
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)
boosting_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42)

# Stacking model
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train models
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
bagging_model.fit(X_train, y_train)
boosting_model.fit(X_train, y_train)
stacking_model.fit(X_train, y_train)

# 🎯 Visualize the Decision Tree (limited depth for clarity)
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True)
plt.title("Decision Tree Visualization (Depth=4)")
plt.show()

# Predictions
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
bagging_pred = bagging_model.predict(X_test)
boosting_pred = boosting_model.predict(X_test)
stacking_pred = stacking_model.predict(X_test)

# Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score:{f1:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate all models
evaluate_model(y_test, dt_pred, "Decision Tree")
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, bagging_pred, "Bagging")
evaluate_model(y_test, boosting_pred, "Boosting")
evaluate_model(y_test, stacking_pred, "Stacking")





import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

plt.figure(figsize=(6,4))
sns.countplot(x=y,palette="Set3")
plt.title("Digit Class Distribution")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define kernels to test
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

# Train and evaluate each kernel
for kernel in kernels:
    print(f"\n--- Kernel: {kernel.upper()} ---")
    model = SVC(kernel=kernel, gamma=0.001, C=100.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[kernel] = acc  # Store accuracy for plotting

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix ({kernel} kernel)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# 📊 Compare Kernel Accuracies
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("SVM Accuracy Comparison Across Kernels")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)  # Zoom in on the useful range
plt.xlabel("Kernel Type")
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import dendrogram, linkage

# Create standard datasets
blobs, _ = make_blobs(n_samples=300, centers=4, random_state=42)
moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

scaler = StandardScaler()
blobs = scaler.fit_transform(blobs)
moons = scaler.fit_transform(moons)

# Initialize clustering models
kmeans = KMeans(n_clusters=4, random_state=42)
agglo = AgglomerativeClustering(n_clusters=4)
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Fit and predict
kmeans_labels_blobs = kmeans.fit_predict(blobs)
agglo_labels_blobs = agglo.fit_predict(blobs)
dbscan_labels_blobs = dbscan.fit_predict(blobs)
kmeans_labels_moons = kmeans.fit_predict(moons)
agglo_labels_moons = agglo.fit_predict(moons)
dbscan_labels_moons = dbscan.fit_predict(moons)

# Plot clustering results
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
ax[0, 0].scatter(blobs[:, 0], blobs[:, 1], c=kmeans_labels_blobs)
ax[0, 0].set_title('k-Means on Blobs')
ax[1, 0].scatter(blobs[:, 0], blobs[:, 1], c=agglo_labels_blobs)
ax[1, 0].set_title('Agglomerative Clustering on Blobs')
ax[2, 0].scatter(blobs[:, 0], blobs[:, 1], c=dbscan_labels_blobs)
ax[2, 0].set_title('DBSCAN on Blobs')
ax[0, 1].scatter(moons[:, 0], moons[:, 1], c=kmeans_labels_moons)
ax[0, 1].set_title('k-Means on Moons')
ax[1, 1].scatter(moons[:, 0], moons[:, 1], c=agglo_labels_moons)
ax[1, 1].set_title('Agglomerative Clustering on Moons')
ax[2, 1].scatter(moons[:, 0], moons[:, 1], c=dbscan_labels_moons)
ax[2, 1].set_title('DBSCAN on Moons')
plt.tight_layout()
plt.show()
