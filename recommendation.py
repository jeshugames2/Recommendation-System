import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from PyQt5 import QtWidgets

history = pd.read_excel("history.xlsx")
product = pd.read_excel("products.xlsx")
product = product.iloc[:,2:]
product= product.fillna(0)

users = history.iloc[:, 1].tolist()
products = product.iloc[:, 0].tolist()
types = list(product)[2:]

num_user = len(users)
num_products = len(products)
num_types = len(types)

users_products = history.iloc[:, 2:].values.tolist()
products_types = product.iloc[:, 2:].values.tolist()

users_products = tf.constant(users_products, dtype= tf.float32)
products_types = tf.constant(products_types, dtype= tf.float32)

weighted_types_matrices = [tf.expand_dims(tf.transpose(users_products)[:, i], axis = 1) * products_types for i in range(num_user)]
users_products_types = tf.stack(weighted_types_matrices)

users_products_types_sums = tf.reduce_sum(users_products_types, axis = 1)
users_products_types_totals = tf.reduce_sum(users_products_types_sums, axis = 1)

users_types = tf.stack([users_products_types_sums[i,:]/users_products_types_totals[i] for i in range(num_user)], axis = 0)

def find_user_top_type(user_index):
    types_ind = tf.nn.top_k(users_types[user_index], num_types)[1]
    return tf.gather_nd(types, tf.expand_dims(types_ind, axis = 1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    users_toptypes = {}
    for i in range(num_user):
        top_types = sess.run(find_user_top_type(i))
        users_toptypes[users[i]] = list(top_types)
        
users_ratings = [tf.map_fn(lambda x: tf.tensordot(users_types[i], x, axes = 1), products_types) for i in range(num_user)]
all_users_ratings = tf.stack(users_ratings)

all_users_ratings_new = tf.where(tf.equal(users_products, tf.zeros_like(users_products)),
                                 all_users_ratings,
                                 -np.inf * tf.ones_like(tf.cast(users_products, tf.float32)))

def find_user_top_product(user_index, num_to_recommend):
    products_ind = tf.nn.top_k(all_users_ratings_new[user_index], num_to_recommend)[1]
    return tf.gather_nd(products, tf.expand_dims(products_ind, axis = 1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    user_topproducts = {}
    num_to_recommend = tf.reduce_sum(tf.cast(tf.equal(users_products, tf.zeros_like(users_products)), dtype = tf.float32), axis = 1)
    for ind in range(num_user):
        top_products = sess.run(find_user_top_product(ind, tf.cast(num_to_recommend[ind], dtype = tf.int32)))
        user_topproducts[users[ind]] = list(top_products)
        
def toString():
    for (i,j) in user_topproducts.items():
        for x in range(len(j)):
            j[x] = str(j[x], 'utf-8')
            

    for (i,j) in users_toptypes.items():
        for x in range(len(j)):
            j[x] = str(j[x], 'utf-8')

toString()
    
class Window(QtWidgets.QWidget):
    
    def __init__(self):
        super(Window, self).__init__()
        
        self.window()
    
    def btn_click(self):
        sender = self.sender()
        if sender.text() == "Display":
            if self.te.text() not in user_topproducts.keys():
                self.la.setText("Customer not found")
            else:
                for (i,j) in user_topproducts.items():
                    if self.te.text() == i:
                        j = ", ".join(str(f) for f in j) 
                        self.lc.setText(j)
                        break
                for (i,j) in users_toptypes.items():
                    if self.te.text() == i:
                        j = ", ".join(str(f) for f in j) 
                        self.le.setText(j)
                        break
    
    def refresh(self):
        self.la.setText("Enter Customer Name:")
        self.lc.clear()
        self.le.clear()
        self.te.clear()
        
    
    def window(self):
        self.la = QtWidgets.QLabel()
        self.la.setText("Enter Customer Name:")
        self.te = QtWidgets.QLineEdit()
        self.b1 = QtWidgets.QPushButton("Display")
        self.lb = QtWidgets.QLabel()
        self.lb.setText("Products:")
        self.lc = QtWidgets.QLabel()
        self.ld = QtWidgets.QLabel()
        self.ld.setText("Types:")
        self.le = QtWidgets.QLabel()
        self.b2 = QtWidgets.QPushButton("Refresh")
            
        
        v_box = QtWidgets.QVBoxLayout()
        v_box.addWidget(self.la)
        v_box.addWidget(self.te)
        v_box.addWidget(self.b1)
        v_box.addWidget(self.b2)
        v_box.addWidget(self.lb)
        v_box.addWidget(self.lc)
        v_box.addWidget(self.ld)
        v_box.addWidget(self.le)
        
        self.setLayout(v_box)
        self.b1.clicked.connect(self.btn_click)
        self.b2.clicked.connect(self.refresh)        
        self.setWindowTitle("Recommendation System")        
        self.show()          
        
app = QtWidgets.QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())