Y = 100
from random import uniform

ys = []
for i in range(100):
	y = Y*uniform(0.8, 1.2)
	print(Y, y)
	ys.append(y)
	Y = Y - 0.1 * y

print(ys)
print('-------------------------------------------')
print(0.1*sum(ys))
print(0.1*sum(ys[:-1]) + ys[-1])
