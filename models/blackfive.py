import numpy as np 

 

data = np.array([501.7, 503.2 , 529.4 , 553.8 , 568.7 , 584.1 , 

    608.0 , 626.1 , 655.8 , 682.0 , 717.5 ])

 

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression#导入线性回归模型

from sklearn.preprocessing import PolynomialFeatures# 导入多项式回归模型

x = np.arange(10).reshape(10,1)

y = data[:10].reshape(10,1)

 

plt.plot(x, y, 'g.',markersize =20)

plt.title('BlackFive')

plt.xlabel('x')

plt.ylabel('y')

plt.axis([0, 12, 0, 800])

plt.grid(True)

plt.legend()

 

 

polynomial = PolynomialFeatures(degree = 2)

x_transformed = polynomial.fit_transform(x)

 

poly_linear_model = LinearRegression()

poly_linear_model.fit(x_transformed, y)

 

xx = np.linspace(0, 12, 100)

xx_transformed = polynomial.fit_transform(xx.reshape(xx.shape[0], 1))

yy = poly_linear_model.predict(xx_transformed)

plt.plot(xx, yy,label="$y = ax^2 + bx + c$")

plt.legend()


polynomial2 = PolynomialFeatures(degree = 3)

x_transformed = polynomial2.fit_transform(x)

 

poly_linear_model2 = LinearRegression()

poly_linear_model2.fit(x_transformed, y)

 

xx = np.linspace(0, 12, 100)

xx_transformed = polynomial2.fit_transform(xx.reshape(xx.shape[0], 1))

yy = poly_linear_model2.predict(xx_transformed)

plt.plot(xx, yy,label="$y = ax^3 + bx + c$")

plt.legend()


polynomial4 = PolynomialFeatures(degree = 4)

x_transformed = polynomial4.fit_transform(x)

 

poly_linear_model4 = LinearRegression()

poly_linear_model4.fit(x_transformed, y)

 

xx = np.linspace(0, 12, 100)

xx_transformed = polynomial4.fit_transform(xx.reshape(xx.shape[0], 1))

yy = poly_linear_model4.predict(xx_transformed)

plt.plot(xx, yy,label="$y = ax^4 + bx + c$")

plt.legend()


 

x_test_cubic = polynomial.transform(x)
print('二次线性回归 r-squared', poly_linear_model.score(x_test_cubic, y))
 
aa=polynomial.fit_transform([[10]])

bb = poly_linear_model.predict(aa)

print(bb, data[-1])

x_test_cubic = polynomial2.transform(x)
print('三次线性回归 r-squared', poly_linear_model2.score(x_test_cubic, y))

aa=polynomial2.fit_transform([[10]])

bb = poly_linear_model2.predict(aa)

print(bb, data[-1])

x_test_cubic = polynomial4.transform(x)
print('三次线性回归 r-squared', poly_linear_model4.score(x_test_cubic, y))

aa=polynomial4.fit_transform([[10]])

bb = poly_linear_model4.predict(aa)

print(bb, data[-1])


plt.savefig("blackfive.jpg")