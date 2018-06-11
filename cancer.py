from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.neural_network import MLPClassifier

cancer=load_breast_cancer()
print("cancer.key():{}".format(cancer.keys()))
print("데이터의 형태:{}".format(cancer.data.shape))
print("클래스별 샘플 갯수:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

X_train, X_test, y_train, y_test=train_test_split(
    cancer.data, cancer.target, random_state=0)
mlp=MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
print("훈련 세트 정확도: {:.2f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(mlp.score(X_test,y_test)))