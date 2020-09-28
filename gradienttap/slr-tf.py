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

w = tf.Variable(0.)
b = tf.Variable(0.)
eta = 0.0001
mse = []

for i in range(0,1000):
    #自动求导
    with tf.GradientTape() as g:
        l = tf.math.reduce_mean(tf.math.square(y-w*x-b))
    dl_dw, dl_db = g.gradient(l,[w,b])
    
    mse.append(l)
    w.assign_sub(eta*dl_dw)
    b.assign_sub(eta*dl_db)

    if i % 100 == 0:
        print('loss=',l.numpy(),' w=',w.numpy(), ' b=',b.numpy())

print('w=',w.numpy(), ' b=',b.numpy())
predict = w*x+b
actual = 0.89*x+5.41

plt.figure()

plt.subplot(1,2,1)
plt.scatter(x,y)
plt.plot(x,predict,'o-y',linewidth=.5,label='predict')
plt.plot(x,actual,'o-r',linewidth=.5,label='actual')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()

plt.subplot(1,2,2)
plt.plot(mse)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()