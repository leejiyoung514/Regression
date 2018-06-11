from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import mglearn
from matplotlib import pyplot as plt

# X, y=make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
#
# mlp=MLPClassifier(\
#     solver="lbfgs", random_state=0).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
# mglearn.discrete_scatter(X_train[:,0], X_train[:,1],y_train)
# plt.xlabel("특성0")
# plt.ylabel("특성1")
# plt.show()


# X, y=make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
#
# mlp=MLPClassifier(\
#     solver="lbfgs", random_state=0,\
#     hidden_layer_sizes=[10]).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
# mglearn.discrete_scatter(X_train[:,0], X_train[:,1],y_train)
# plt.xlabel("특성0")
# plt.ylabel("특성1")
# plt.show()


X, y=make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)

mlp=MLPClassifier(\
    solver="lbfgs", random_state=0,\
    hidden_layer_sizes=[10,10]).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:,0], X_train[:,1],y_train)
plt.xlabel("특성0")
plt.ylabel("특성1")
plt.show()
