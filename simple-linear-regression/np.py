import numpy as np

x = np.array([
137.97,
104.50,
100.00,
124.32,
79.20,
99.00,
124.00,
114.00,
106.69,
138.05,
53.75,
46.91,
68.00,
63.02,
81.26,
86.21
])

y = np.array([
145.00,
110.00,
93.00,
116.00,
65.32,
104.00,
118.00,
91.00,
62.00,
133.00,
51.00,
45.00,
78.50,
69.65,
75.69,
95.30
])

meanx = np.mean(x)
meany = np.mean(y)
n = len(x)
sumxy = np.sum(x*y)
sumxx = np.sum(x*x)

w = (sumxy-n*meanx*meany)/(sumxx-n*meanx*meanx)
b = meany - w*meanx

print("w=",w)
print("b=",b)

t = np.array([
128.15,
45.00,
141.43,
106.27,
99.00,
53.84,
85.36,
70.00
])

r = w*t+b
print(t, "\r\n", r)
