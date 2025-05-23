import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
x=2+np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)
y=y.flatten()
def gradient_descent(x,y,alpha,iterations):
    cost_h=[]
    m,n=x.shape
    theta=np.zeros(n)
    for i in range(iterations):
        predictions=x.dot(theta)
        error=predictions-y
        gradient=(1/m)*x.T.dot(error)
        theta-=alpha*gradient
        cost=(1/(2*m)*np.sum(error**2))
        cost_h.append(cost)
    return theta,cost_h
def least(x,y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

def polynomial(x,y,d):
    ploy=PolynomialFeatures(d)
    x_p=ploy.fit_transform(x)
    return np.linalg.inv(x_p.T.dot(X).dot(x.T).dot(y))
x_b=np.c_[np.ones((x.shape[0],1)),x]
# theta,cost=gradient_descent(x_b,y,0.1,1000)
# plt.scatter(x,y,label="data")
# plt.plot(x,x_b.dot(theta),color="red")
# plt.show()
# plt.plot(cost)
# plt.show()
# print(theta)
theta=polynomial(x_b,y,3)
print(theta)




import numpy as np
import matplotlib.pyplot as plt

def least_squares(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# === Generate synthetic data ===
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

# # Add bias term to X
# X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (100, 2)

# === Fit using least squares ===
theta = least_squares(X_b, y)

# === Plot data and regression line ===
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data")
plt.plot(X, X_b.dot(theta), color="red", label="Least Squares Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Least Squares Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# === Print result ===
print("Learned parameters (theta):", theta.flatten())




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(0)
x=2*np.random.rand(100,1)
y=4+3*x+2*x**2+np.random.randn(100,1)
poly=PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly.fit_transform(x)

# === Add bias column ===
X_poly_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]

# === Solve using Normal Equation ===
theta = np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y)
x1=np.linspace(0,2,100).reshape(100,1)
x1_poly=poly.transform(x1)
x1_b=np.c_[np.ones((x1_poly.shape[0],1)),x1_poly]
y1=x1_b.dot(theta)
plt.scatter(x,y)
plt.plot(x1,y1,color="red")
plt.show()

print(theta)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def lasso_regression(X, y, alpha=1.0):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)

    # Plotting
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, lasso.predict(X), color='red', label='Lasso Regression Line')
    plt.title(f"Lasso Regression (alpha={alpha})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    return lasso.coef_
# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

X = X.reshape(-1, 1)

# Run Lasso regression with visualization
lasso_regression(X, y, alpha=0.1)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def ridge_regression(X, y, alpha=1.0):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)

    # Plotting
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, ridge.predict(X), color='green', label='Ridge Regression Line')
    plt.title(f'Ridge Regression (alpha={alpha})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    return ridge.coef_
# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

X = X.reshape(-1, 1)

# Run Ridge regression with visualization
ridge_regression(X, y, alpha=0.5)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,classification_report
data=load_breast_cancer()
x,y=data.data,data.target
sns.countplot(x=y,palette="Set2")
plt.show()
df=pd.DataFrame(x,columns=data.feature_names)
c=df.corr()
sns.heatmap(c,cmap="coolwarm")
plt.show()
scaler=StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
sns.heatmap(confusion_matrix(y_test,y_pred),cmap="Blues",annot=True,xticklabels=data.target_names,yticklabels=data.target_names)
plt.show()
print(accuracy_score(y_pred,y_test))

