#! /usr/bin/env python3

# 引入基本计算将要使用的库
import numpy as np
import numpy.linalg as alg
import json

# 从数据文件中读取需要的数据
with open('LeastSquare.json', 'r') as dataFile:
    jsonRaw = dataFile.read()

jsonData = json.loads(jsonRaw)

# 检验数据有效性
if jsonData['size'] > 10000 or jsonData['order'] > 15:
    print("规模过大。中止。")
    quit()

if len(jsonData['x']) != jsonData['size'] or len(jsonData['y']) != jsonData['size']:
    print("输入数据格式错误。中止。")
    quit()

# 设定基本变量
p = jsonData['order']
X = jsonData['x']
Y = jsonData['y']
size = jsonData['size']

# 构造方程组并求解
constantTerms = []
equationSet = []

for k in range(p + 1):
    constantTerms.append(sum( [ X[i]**k * Y[i] for i in range(size) ] ))
    equation = []
    for t in range(p + 1):
        equation.append(sum( [ X[i]**(k+t) for i in range(size) ] ))
    equationSet.append(equation)

left = np.array(equationSet)
right = np.array(constantTerms)

coeff = alg.solve(left, right)

# 构造输出的多项式结果
polynomialOutput = "y = "

for invp in range(p):
    if p - invp >= 2:
        polynomialOutput += "{0} x^{1} + ".format(coeff[p - invp], p - invp)
    else:
        polynomialOutput += "{0} x + ".format(coeff[p - invp])

polynomialOutput += "{0}".format(coeff[0])

print(polynomialOutput)

# 绘制输出图像
import matplotlib.pyplot as plt

def eval(x):
    return sum( [ coeff[i] * x**i for i in range(p + 1)] )

#  计算坐标轴的范围
minX = min(X)
maxX = max(X)
minY = min(Y)
maxY = max(Y)

limX = (maxX - minX) / 20.0
limY = (maxY - minY) / 20.0

plt.scatter(X, Y)
plt.xlim((minX - limX, maxX + limX))
plt.ylim((minY - limY, maxY + limY))
plt.xlabel(jsonData['varX'])
plt.ylabel(jsonData['varY'])

#  绘制拟合曲线
apprx = np.linspace(minX - limX, maxX + limX, int((maxX - minX + 2*limX)*20))
appry = [eval(t) for t in apprx]

plt.plot(apprx, appry, color='red', linewidth=0.3)

#  保存图形
plt.savefig('LeastSquareResult.png', format='png')
