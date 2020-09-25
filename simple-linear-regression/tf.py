import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.constant([
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
y = tf.constant([
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

meanx = tf.math.reduce_mean(x)
meany = tf.math.reduce_mean(y)
n = tf.size(x,out_type=tf.float32)
sumxy = tf.math.reduce_sum(tf.math.multiply(x,y))
sumxx = tf.math.reduce_sum(tf.math.multiply(x,x))

w = (sumxy - n*meanx*meany)/(sumxx - n*meanx*meanx)
b = meany - w*meanx

print("w=",w)
print("b=",b)

x_test = tf.constant([
128.15,
45.00,
141.43,
106.27,
99.00,
53.84,
85.36,
70.00
])

y_predict = w*x_test + b
print(x_test.numpy())
print(y_predict.numpy())

plt.scatter(x,y,label='simple')
plt.plot(x_test,y_predict,'o-y',linewidth=.5,label='linear regression')
plt.legend()
plt.show()