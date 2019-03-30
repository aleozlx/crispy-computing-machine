import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()
tf.reset_default_graph()
img = tf.placeholder(tf.float32, shape=(None, 2), name='input_plhdr')
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', name='Intermediate'),
    tf.keras.layers.Dense(2, activation='softmax', name='Output'),
])
pred = model(img)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, './exported/my_model')
    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
