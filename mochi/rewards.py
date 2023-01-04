from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.y = train_y
        self.clf = LogisticRegression(random_state=0).fit(self.X, self.y)
        print("score: {}".format(self.clf.score(self.X, self.y)))

    def predict(self, pred_x):
        pred_y = self.clf.predict_proba(pred_x)
        return pred_y


class LinearModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.y = train_y
        self.clf = LinearRegression().fit(self.X, self.y)
        print("score: {}".format(self.clf.score(self.X, self.y)))

    def predict(self, pred_x):
        pred_y = self.clf.predict(pred_x)
        return pred_y


class LassoModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.y = train_y
        reg = LassoCV(alphas=np.logspace(-6, 6, 13))
        reg.fit(self.X, self.y)
        # print(reg.alpha_)
        self.clf = Ridge(alpha=reg.alpha_).fit(self.X, self.y)
        print("score: {}".format(self.clf.score(self.X, self.y)))

    def predict(self, pred_x):
        pred_y = self.clf.predict(pred_x)
        return pred_y


class RidgeModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.y = train_y
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        # reg = RidgeCV(alphas=np.logspace(-6, 6, 13))
        # reg.fit(self.X, self.y)
        # print(reg.alpha_)
        # print(reg.best_score_)
        # self.clf = Ridge(alpha=100., solver="lbfgs", positive=True).fit(self.X, self.y)
        self.clf = Ridge(alpha=100., solver="saga", random_state=1024).fit(self.X, self.y)
        # print("score: {}".format(self.clf.score(self.X, self.y)))
        # self.clf = make_pipeline(StandardScaler(), Ridge(alpha=reg.alpha_))
        # self.clf.fit(self.X, self.y)

    def predict(self, pred_x):
        print("x before scaling: {}".format(pred_x))
        pred_x = self.scaler.transform(pred_x)
        print("x after scaling: {}".format(pred_x))
        pred_y = self.clf.predict(pred_x)
        # print("x: {}".format(pred_x))
        print("intercept: {}, coef: {}".format(self.clf.intercept_, list(self.clf.coef_)))
        return pred_y


class SVRModel():
    def __init__(self, train_X, train_y):
        self.type = self.__class__.__name__
        self.X = train_X
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y = train_y
        self.svr = SVR(kernel="poly", C=0.1, epsilon=0.1)
        # self.svr = make_pipeline(StandardScaler(), SVR(kernel="linear", C=.1, epsilon=0.2))
        # self.svr = make_pipeline(StandardScaler(), SVR(kernel="poly", C=0.1, epsilon=0.1))
        # self.svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=0.1, epsilon=0.1))
        self.svr.fit(self.X, self.y)

    def predict(self, pred_x):
        pred_x = self.scaler.fit_transform(pred_x)
        pred_y = self.svr.predict(pred_x)
        return pred_y
