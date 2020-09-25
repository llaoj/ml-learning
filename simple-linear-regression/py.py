x = [
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
]
y = [
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
]

meanx = sum(x)/len(x)
meany = sum(y)/len(y)

n = len(x)

sumxy = 0
sumxx = 0
for i in range(0, n):
    sumxy += x[i]*y[i]
    sumxx += x[i]*x[i]

w = (sumxy - n*meanx*meany)/(sumxx - n*meanx*meanx)
b = meany - w*meanx

print("w=",w)
print("b=",b)

t = [
128.15,
45.00,
141.43,
106.27,
99.00,
53.84,
85.36,
70.00
]

for i in range(0,len(t)):
    y = round(w*t[i] + b, 2)
    print(t[i], "\t", y)