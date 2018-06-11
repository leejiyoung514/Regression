import mglearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

boston=load_boston()
print("데이터의 형태: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()
print(X.shape)
print(y)
print("="*50)
X_train, X_test, y_train, y_test=\
    train_test_split(X,y,test_size=0.3, random_state=0)
lr=LinearRegression().fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
print("="*50)
#릿지 회귀
ridge=Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train,y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
print("="*50)
ridge=Ridge(alpha=10).fit(X_train,y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
print("="*50)
ridge=Ridge(alpha=0.1).fit(X_train,y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
