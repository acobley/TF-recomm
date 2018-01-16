import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import socket
import sys
import dataio
import ops

np.random.seed(13575)

BATCH_SIZE = 10000
USER_NUM = 259137
ITEM_NUM = 165201
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"


def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])
def get_movies():
    df = dataio.read_movies("/tmp/movielens/ml-latest/movies.csv", sep=",")
    rows = len(df)
    return df, rows		


if __name__ == '__main__':

    with tf.Session() as sess:
       new_saver = tf.train.import_meta_graph("tfrecomm.meta",clear_devices=True)
       new_saver.restore(sess, tf.train.latest_checkpoint('./'))
       sess.run(tf.global_variables_initializer())
       print ("Get Movies Data")
       moviefile,rows = get_movies()
       graph = tf.get_default_graph()
       tf.train.write_graph(graph, '.','saved_model.pb', as_text=False)       
       infer = graph.get_tensor_by_name("svd_inference:0")
       
       user_batch = graph.get_tensor_by_name("id_user:0")
       item_batch = graph.get_tensor_by_name("id_item:0")
       movies=list(range(len(moviefile)))
       infer = graph.get_tensor_by_name("svd_inference:0")
       users=[1]
       feed_dict={user_batch: users,item_batch: movies}       
       pred_batch = sess.run(infer, feed_dict)
       sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       # Bind the socket to the port
       server_address = ('0.0.0.0', 81)
       print >>sys.stderr, 'starting up on %s port %s' % server_address
       sock.bind(server_address)
       sock.listen(1)
       while True:
           # Wait for a connection
           print >>sys.stderr, 'waiting for a connection'
           connection, client_address = sock.accept()
           try:
               print >>sys.stderr, 'connection from', client_address
               # Receive the data in small chunks and retransmit it
               while True:
                   data = connection.recv(16)
                   print >>sys.stderr, 'received "%s"' % data
                   if data:
                       del users[:]
                       try:
                           user = int(data)
                       except:
                           break
                       users.append(int(data))
                       print (users)
                       pred_batch = sess.run(infer, feed_dict={user_batch: users,item_batch: movies})
                       moviesrecomm=list(zip(movies,pred_batch))
                       smovies=sorted (moviesrecomm,key=lambda x:x[1],reverse=True)
                       topmovies= smovies[0:10]
                       print (topmovies)
                       for item in topmovies:
                           itopmovie=item[0]
                           recommendedmovie=moviefile["title"][itopmovie]
                           recommendedtags=moviefile["tags"][itopmovie]
                           #print >>sys.stderr, 'sending data back to the client'
                           connection.sendall(recommendedmovie+":"+recommendedtags+"\n")
                           #print >>sys.stderr, 'Sent data'
                   else:
                       print >>sys.stderr, 'no more data from', client_address
                       break
           finally:
               connection.close()