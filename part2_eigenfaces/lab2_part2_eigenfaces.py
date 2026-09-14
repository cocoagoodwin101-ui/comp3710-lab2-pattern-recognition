import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# ---- Load data ----
print("Fetching LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# ---- Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_components = 150

# ---- Center data ----
mean = np.mean(X_train, axis=0)
X_train = X_train - mean
X_test = X_test - mean

# PCA finds directions of maximum variance. Without mean-centering, the 
# first component would be dominated by the average face's brightness/shape rather 
# than the meaningful variation between different faces.

# ---- PCA via SVD ----
U, S, V = np.linalg.svd(X_train, full_matrices=False) 
# SVD factorises X = U*S*V^T directly, and the rows of V^T mathematically
# are the principal components, without ever forming the potentially large 
# and less numerically stable covariance matrix explicitly.
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

X_transformed = np.dot(X_train, components.T)
print("X_transformed shape:", X_transformed.shape)

X_test_transformed = np.dot(X_test, components.T)
print("X_test_transformed shape:", X_test_transformed.shape)

# ---- Plot eigenfaces gallery ----
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.savefig("eigenfaces_gallery.png")
print("Saved eigenfaces_gallery.png")

# ---- Compactness plot ----
explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = np.cumsum(explained_variance_ratio)
print("ratio_cumsum shape:", ratio_cumsum.shape)

eigenvalueCount = np.arange(n_components)
plt.figure()
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.savefig("compactness.png")
print("Saved compactness.png")

# ---- Random Forest classification ----
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train)

predictions = estimator.predict(X_test_transformed)
correct = predictions == y_test
total_test = len(X_test_transformed)

print("Total Testing", total_test)
print("Total Correct:", np.sum(correct))
print("Accuracy:", np.sum(correct) / total_test)
print(classification_report(y_test, predictions, target_names=target_names))

print("\nDone.")

#swapped every plt.show() from the lab sheet for plt.savefig(...) 
# — this is a necessary change, not optional. plt.show() tries to 
# open an interactive window, but a Slurm batch job has no display 
# attached to it, so it would either hang or silently do nothing. 
# Saving to a PNG file is better