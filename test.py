class Test:
    def __init__(self, x):
        self.x = x
    
    def f(self):
        self.x = self.x - 1


# from sklearn.datasets import load_diabetes
# from sklearn.feature_selection import f_regression, mutual_info_regression
# from sklearn.feature_selection import chi2

# diabetes = load_diabetes()
# X = diabetes["data"]
# y = diabetes["target"]
# f,p = f_regression(X, y)
# print(f)

# import matplotlib.pyplot as plt
# plt.scatter(X[:,4], y)
# plt.savefig("4.png")
# plt.show()


# 2 3 6 8 10