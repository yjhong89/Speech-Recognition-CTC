#!/usr/bin/python
# -*- coding : utf-8 -*-

from __future__ import print_function
import numpy as np
import time, os, math, shutil
import tensorflow as tf
from ctc_model import LayerNormalizedLSTM
from data_loaders import TextLoader
from ops import *


class CLM_Model():
	def __init__(self, args, sess, infer=False):
	self.args = args
	self.sess = sess
	self.data_loader = TextLoader(self.args.files_dir, self.args.seq_length, self.args.batch_size)
	self.args.vocab_size = self.data_loader.vocab_size
	# Used when generating text
	if infer:
 		print('Inference')
		args.batch_size = 1
		args.seq_length = 1
 
	if args.model == 'rnn':
		cell = tf.nn.rnn_cell.BasicRNNCell(args.state_size)
	elif args.model == 'gru':
		cell = tf.nn.rnn_cell.GRUCell(args.state_size)
	# state_is_tuple is only supported for LSTM Cell, returned state is a 2 tuple of (c, h)
	# It makes easy wrap up multiple layer cell
	elif args.model == 'lstm':
		cell = tf.nn.rnn_cell.LSTMCell(args.state_size, state_is_tuple=True)
	# Layer Normalized LSTM
	elif args.model == 'lnlstm':
		cell = LayerNormalizedLSTM(args.state_size)
	else:
		raise Exception("Model type not supported : {}".format(args.model))
   
	if args.dropout == 'y':
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=args.keep_prob)

	if args.model == 'lstm' or args.model == 'lnlstm':
		self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
	else:
		self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

	if args.dropout == 'y':
		self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=args.keep_prob)

	self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
	self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

	with tf.variable_scope('CLM'):
		self.softmax_w = tf.get_variable('weight', [args.state_size, self.args.vocab_size])
		# To add for each row [args.vocab_size,]
		self.softmax_b = tf.get_variable('bias', [self.args.vocab_size], initializer = tf.constant_initializer(0))
		# Mapping each vocabulary to embedding dimension
		# Embed each character input 
		self.embedding = tf.get_variable('embedding', [self.args.vocab_size, args.state_size])
		# rnn_inputs dimenstion has a shape of [batch_size, vocab_size, embedding_size]
		self.rnn_inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

		# rnn_outputs has shape of [batch_size, truncated_backprop_length, state_size(rnn_size)], not list type as normal rnn
		self.rnn_outputs, final_state_  = tf.nn.dynamic_rnn(self.cell, self.rnn_inputs, dtype = tf.float32)
		# Reshape to calculate softmax function
		# Has a shape of [batch_size * trunc_bpp_length, state_size]
		self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, args.state_size])
		# Has a shape of [batch_size * trunc_bpp_length]
		self.y_reshaped = tf.reshape(self.targets, [-1])

		self.logits = tf.matmul(self.rnn_outputs, self.softmax_w) + self.softmax_b
		self.prediction = tf.nn.softmax(self.logits)
		self.pred_summary = histogram_summary('Probs for next chr', self.prediction) 

		# sparse_softmax_cross_entropy`s target label does not need to be one-hot encoded
		self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y_reshaped))
		self.cost_summary = scalar_summary('clm loss', self.cost)
		self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.cost)

		vrbs = tf.trainable_variables()
		print('Trainable variables for clm model')
		for i in xrange(len(vrbs)):
			print(vrbs[i].name)
		self.saver = tf.train.Saver()

	def train(self, args):
		self.summaries = merge_summary([self.pred_summary, self.cost_summary])
		self.writer = SummaryWriter('./clm_logs', self.sess.graph)  
		total_step = 1

		self.sess.run(tf.global_variables_initializer())
		if self.args.init_from:
			if self.load():
				print('Loaded checkpoint')
			else:
				print('Load failed')
		else:
			if os.path.exists(os.path.join(self.args.checkpoint_dir, self.model_dir)):
				try : 
					shutil.rmtree(os.path.join(self.args.checkpoint_dir, self.model_dir))
				except(PermissionError) as e:
					print('[Delete Error] %s - %s' % (e.filename, e.strerror))

		for epoch_index in xrange(args.num_epoch):
			s_time = time.time()
			training_loss = 0
			training_step = 0
			self.data_loader.reset_batch_pointer()
			for tr_step in xrange(self.data_loader.num_batches):
				training_step += 1
				start_time = time.time()
				x, y = self.data_loader.next_batch()
				feed = {self.input_data : x, self.targets : y}
				training_loss_, summary_str, _ = self.sess.run([self.cost, self.summaries, self.optimizer], feed_dict = feed)
				self.writer.add_summary(summary_str, total_step)
				training_loss += training_loss_ * args.batch_size
				print("[%d/%d] in Epoch %d , Train loss = %3.3f, Time per batch = %.3f, %d steps" % (tr_step+1, self.data_loader.num_batches, epoch_index+1, training_loss, time.time() - start_time, total_step))
				total_step += 1

			training_loss /= (args.batch_size * self.data_loader.num_batches) 
			print('Epoch %d, Training loss per Epoch = %.3f, Time per Epoch = %.3f' % (epoch_index+1, training_loss, time.time() - s_time))
   
			if (np.mod(epoch_index+1, self.args.save_interval) == 0) or (epoch_index == args.num_epoch - 1):
				self.save(epoch_index+1)
    
	@property
	def model_dir(self):
		if self.args.dropout == 0: 
			return '{}model_{}layers_{}state_{}batch_clm'.format(self.args.model, self.args.num_layers, self.args.state_size, self.args.batch_size)
		else:
			return '{}model_{}layers_{}state_{}batch_dropout_clm'.format(self.args.model, self.args.num_layers, self.args.state_size, self.args.batch_size)

	def save(self, total_step):
		model_name = 'languagemodel'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
  
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=total_step)
		print('Model saved at : %s' % (checkpoint_dir))

	def load(self):
		print('Reading checkpoint')
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		model_name = 'languagemodel'
		print(checkpoint_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			print(ckpt.model_checkpoint_path)
			if ckpt_name.split('-')[0] == model_name:
				self.saver.restore(self.sess, ckpt.model_checkpoint_path)
				print('Success to read %s' % (ckpt_name))
				return True
			else:
				return False
		else:
			print('Failed to find checkpoint')
			return False
 
	def next_label_prediction(self):
		self.sess.run(tf.global_variables_initializer())
		print(self.args)
		# If it is not loaded, it just spat out results with untrained variables
		if self.load():
			print('Loaded checkpoint')
		else:
			print('Load failed')
		self.args.dropout = False
		self.decode_summary = merge_summary([self.pred_summary])
		self.decode_writer = SummaryWriter('./decode_clm_logs', self.sess.graph)
		chr_datas = TextLoader(self.args.files_dir, self.args.seq_length, self.args.batch_size)
		self.args.vocab_size = chr_datas.vocab_size
		# chars in tuple
		self.target_chrs = chr_datas.chars
		print(self.target_chrs)
		# Make default dictionary to store label probability for each prompt
		label_pred_dict = collections.defaultdict(object)
		
		for i in xrange(0, len(self.target_chrs)):
			# Getting present label
			prompt = self.target_chrs[i]
			# Change it to index
			current_char = chr_datas.vocab_to_idx[prompt]
			# To feed to input placeholder, make it to numpy array which has 2 dimension [[ ]]
			cur_char_array = np.asarray([[current_char]])
			# Remember do not feed to target placeholder 
			feed = {self.input_data : cur_char_array}
			# Getting softmax probability distribution
			preds, summary_str = self.sess.run([self.prediction, self.decode_summary], feed_dict=feed)
			self.decode_writer.add_summary(summary_str, i)
			# Get next label prediction based on prompt, size of [vocab_size,]
			p = np.squeeze(preds)
			# Make dictionary for each character`s probability based on prompt
			pred_dict = dict(zip(self.target_chrs, p))
			# Order from whom has higher probability
			pred_dict_order = sorted(pred_dict.items(), key=lambda x : -x[-1])
			pred_dict_order = dict(pred_dict_order)
			# Add prob dictionary key
			label_pred_dict[prompt] = pred_dict_order

		clm_file = os.path.join(self.args.files_dir, 'clm.pkl')
		with open(clm_file, 'w') as f:
			cPickle.dump(label_pred_dict, f)
			f.close() 
   
		return label_pred_dict
  
 
