import time
from collections import deque
import socket
import sys
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import dataio
import ops

np.random.seed(13575)

BATCH_SIZE = 10000
USER_NUM = 259137
ITEM_NUM = 165201
DIM = 15
EPOCH_MAX = 100
DEVICE = "/gpu:0"


def clip(x):
	return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
	return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])



def get_data():
	print("Reading")
	df = dataio.read_process("/tmp/movielens/ml-latest/ratings.csv", sep=",")
	rows = len(df)
	print("iloc")
	df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
	split_index = int(rows * 0.9)
	print("split train")
	df_train = df[0:split_index]
	print("Split test")
	df_test = df[split_index:].reset_index(drop=True)
	return df_train, df_test, rows

def get_movies():
	df = dataio.read_movies("/tmp/movielens/ml-latest/movies.csv", sep=",")
	rows = len(df)
	return df, rows	


def svd(train, test,length,moviefile):
	print ("Movies file length:")
	print (len(moviefile))
	samples_per_batch = len(train) // BATCH_SIZE

	iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                        train["rate"]],
                                        batch_size=BATCH_SIZE)

	iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                        test["rate"]],
                                        batch_size=-1)

	user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
	item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
	rate_batch = tf.placeholder(tf.float32, shape=[None])

	infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
	global_step = tf.contrib.framework.get_or_create_global_step()
	_, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)
	#zeros= tf.Variable(tf.zeros([1]),name="zeros")

	init_op = tf.global_variables_initializer()
	saver=tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init_op)
		summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
		print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
		errors = deque(maxlen=samples_per_batch)
		start = time.time()
		
		for i in range(EPOCH_MAX * samples_per_batch):
			users, items, rates = next(iter_train)
			_, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
		                                                           item_batch: items,
		                                                                                                               rate_batch: rates})
			pred_batch = clip(pred_batch)
			errors.append(np.power(pred_batch - rates, 2))
			if i % samples_per_batch == 0:
				train_err = np.sqrt(np.mean(errors))
				test_err2 = np.array([])
				for users, items, rates in iter_test:
					pred_batch = sess.run(infer, feed_dict={user_batch: users,
				                                            item_batch: items})
					pred_batch = clip(pred_batch)
					test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
				end = time.time()
				test_err = np.sqrt(np.mean(test_err2))
				print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
			                                           end - start))
				train_err_summary = make_scalar_summary("training_error", train_err)
				test_err_summary = make_scalar_summary("test_error", test_err)
				summary_writer.add_summary(train_err_summary, i)
				summary_writer.add_summary(test_err_summary, i)
				start = end

		#meta_graph_def = tf.train.export_meta_graph(filename='/tmp/tfrecomm.meta')
		save_path=saver.save(sess,"tfrecomm.ckpt")
		print("Model saved in file: %s" % save_path)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		# Bind the socket to the port
		server_address = ('127.0.0.1', 81)
		print >>sys.stderr, 'starting up on %s port %s' % server_address
		sock.bind(server_address)
		sock.listen(1)

		movies=list(range(len(moviefile)))
		print (movies)
		users=[1]
		pred_batch = sess.run(infer, feed_dict={user_batch: users,item_batch: movies})

		moviesrecomm=list(zip(movies,pred_batch))
		smovies=sorted (moviesrecomm,key=lambda x:x[1],reverse=True)

		print (" Top Movies ------------------------------------------------------------")

		topmovies= smovies[0:10]
		print (topmovies)
		print

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
							#print >>sys.stderr, 'sending data back to the client'
							connection.sendall(recommendedmovie+"\n")
							#print >>sys.stderr, 'Sent data'
					else:
						print >>sys.stderr, 'no more data from', client_address
						break
			finally:
				connection.close()
if __name__ == '__main__':
	print ("Get Ratings Data")
	df_train, df_test, length = get_data()
	print ("Get Movies Data")
	df_movies,rows = get_movies()

	svd(df_train, df_test, length,df_movies)
	print("Done!")
