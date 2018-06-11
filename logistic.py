from matplotlib import pyplot as plt
import mglearn
import pandas as pd
from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.rcParams['figure.dpi']=300
X,y=mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["클래스 0","클래스 1"], loc=4)
plt.xlabel("첫번째 특성")
plt.ylabel("두번째 특성")

print("X.shape: {}".format(X.shape))

fig, axes = plt.subplots(1, 1,figsize=(5,3))
model=LogisticRegression()

clf=model.fit(X,y)

mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                ax=axes, alpha=0.7)

mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=axes)
axes.set_title("로지스틱 회귀분석")
axes.set_xlabel("특성0")
axes.set_ylabel("특성1")
axes.legend()
plt.show()