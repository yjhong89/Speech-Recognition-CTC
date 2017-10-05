#!/usr/bin/python
# -*- coding : utf-8 -*-

from __future__ import print_function
import numpy as np
import time, os, shutil
import tensorflow as tf
from ops import *
#from data_loaders import *


class RNN_Model():
    def __init__(self, args, sess):
    	self.args = args
    	self.sess = sess
     
    	if self.args.model == 'GRU':
    		self.cell = tf.contrib.rnn.MultiRNNCell([self.GRU() for _ in range(self.args.num_layers)])
    	elif self.args.model == 'LSTM':
    		self.cell = tf.contrib.rnn.MultiRNNCell([self.LSTM() for _ in range(self.args.num_layers)])
    	else:
    		raise Exception("Model type not supported : {}".format(self.args.model))
    
    	if args.dropout is True:
    		self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.args.keep_prob)
    
    	# batch_size, max_stepsize can vary along examples
    	self.input_data = tf.placeholder(tf.float32, [None, None, self.args.num_features])
    	# Need Sparse tensor representation for ctc calculation
    	self.targets = tf.sparse_placeholder(tf.int32)
    	# [batch_size]
    	self.seq_len = tf.placeholder(tf.int32, [None])  
    	# bidirectional has tuples of forward and backward
    	self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, self.input_data, self.seq_len, dtype = tf.float32)
    	# rnn_outputs has shape of [batch_size, truncated_backprop_length, state_size(rnn_size)], not list type as normal rnn
    	self.rnn_output_fw, self.rnn_output_bw = self.rnn_outputs
    	# Reshaping rnn output to calculate and share softmax parameters along all timestep
    	self.rnn_output_fw = tf.reshape(self.rnn_output_fw, [-1, args.state_size])
    	self.rnn_output_bw = tf.reshape(self.rnn_output_bw, [-1, args.state_size])  
    	# Getting input tensor size
    	self.batch_s = tf.shape(self.input_data)[0]
    	self.step_s = tf.shape(self.input_data)[1]
    	
    	# Define softmax parameters, Need each weight parameters for forward and backward(bidirectional rnn outputs)
    	with tf.variable_scope('softmax'):
    	 	self.bias = tf.get_variable('softmax_b', [self.args.num_classes], initializer=tf.constant_initializer(0))
    	 	self.weight_fw = tf.get_variable('softmax_w_fw', [self.args.state_size, self.args.num_classes], initializer=tf.truncated_normal_initializer(stddev=0.02))
    	 	self.weight_bw = tf.get_variable('softmax_w_bw', [self.args.state_size, self.args.num_classes], initializer=tf.truncated_normal_initializer(stddev=0.02))
    	self.logits = tf.matmul(self.rnn_output_fw, self.weight_fw) + tf.matmul(self.rnn_output_bw, self.weight_bw) + self.bias
    	# Reshaping logits to original shape
    	self.logits_reshaped = tf.reshape(self.logits, [self.batch_s, -1, self.args.num_classes])
    	# Getting probabilities for each label
    	self.probability = tf.nn.softmax(self.logits_reshaped)
    	self.prob_summary = histogram_summary('Probs for each label', self.probability)
    	
    	# Time major is True, ctc function inputs supposed to have a shape of [max_step_size, batch_size, num_classes]
    	self.logits_timemajor = tf.transpose(self.logits_reshaped, [1,0,2])
    	
    	self.loss_function = tf.nn.ctc_loss(labels=self.targets, inputs=self.logits_timemajor, sequence_length=self.seq_len )
    	self.loss = tf.reduce_mean(self.loss_function)
    	self.loss_summary = scalar_summary('ctc loss', self.loss)
    	 
    	#self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
    	self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
    	vrbs = tf.trainable_variables()
    	print('Trainable variables for ctc model')
    	for i in xrange(len(vrbs)):
    		print(vrbs[i].name)
    	# Returns a list of sum(dy/dx) for each x in xs
    	grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, vrbs), self.args.maxgrad)
    	self.train_op = self.optimizer.apply_gradients(zip(grads, vrbs))
    	
    	# decoded is sparse tensor form = indices, values, shape
    	self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits_timemajor, self.seq_len)
    	# tf.edit_distances requires sparse tensor for both arguments
    	self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))
    	self.ler_summary = scalar_summary('LER', self.ler)
    	
    	
    	# Declare saver to save trained variable
    	self.saver = tf.train.Saver()
    
    def GRU(self):
    	cell = tf.contrib.rnn.GRUCell(self.args.state_size, reuse=tf.get_variable_scope().reuse)
    	if self.args.dropout:
    		cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.args.keep_prob, input_size=self.args.num_features)
    	return cell
    def LSTM(self):
    	cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.args.state_size, layer_norm=self.args.layer_norm, reuse=tf.get_variable_scope().reuse)
    	if self.args.dropout:
    		cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.args.keep_prob, input_size=self.args.num_Features)
    	return cell
    
    
    def __str__(self):
    	return 'This model has input size {}, state size {}, {} of classes, () timesteps'.format(self.args.num_features, self.args.state_size, self.args.num_classes, self.step_s)
    
    def train(self):
        # For tensorboard, merging summaries to serialize summary probobuf
        #self.summaries = merge_summary([self.prob_summary, self.loss_summary, self.ler_summary])
        # Pass merged summary to summarywirter to save at disc, need logdir and visualize graph
        #self.writer = SummaryWriter(os.path.join('./ctc_logs', self.model_dir), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
                
        # Continuing training from saved model or not
        if self.args.init_from:
        	if self.load():
        		print('Loaded checkpoint from %d epoch' % self.loaded_epoch)
        	else:
        		print('Load failed')
        else:
        	# Training new one with same arguments
        	if os.path.exists(os.path.join(self.args.checkpoint_dir, self.model_dir)):
        		try : 
        			shutil.rmtree(os.path.join(self.args.checkpoint_dir, self.model_dir+'.csv'))
        			shutil.rmtree(os.path.join(self.args.log_dir, self.model_dir))
        		except(PermissionError) as e:
        			print('[Delete Error] %s - %s' % (e.filename, e.strerror))
        
        total_step = 1 
        datamove_flag = 1
        
        for index in xrange(self.loaded_epoch, self.args.num_epoch):
            # Shuffling datas
            # shuffle index
            start_time = time.time()
            print("%d th epoch starts" % (index+1))
            train_loss = 0
            train_ler = 0
            # Newly load new data
            if datamove_flag:
            	inputs_wave = np.load(os.path.join(self.args.train_wav_dir, 'wave_1.npy'), encoding='bytes')
            	inputs_label = np.load(os.path.join(self.args.train_lbl_dir, 'tran_1.npy'), encoding='bytes')
            	print('%d wave %d target dataset loaded' % (len(inputs_wave), len(inputs_label)))
            	data_length = len(inputs_wave)
                train_index = int(data_length*0.9)
            	trainingsteps_per_epoch = train_index // self.args.batch_size    
            	datamove_flag = 0
                best_valid_loss = 1000
                best_valid_ler = 1000
                batch_wave = inputs_wave[:train_index]
                batch_label = inputs_label[:train_index]
                valid_wav_input = inputs_wave[train_index:data_length]
                valid_trg_label = inputs_label[train_index:data_length]
            
            for tr_step in xrange(trainingsteps_per_epoch):
            	s_time = time.time()
            	batch_idx = [i for i in xrange(self.args.batch_size*tr_step, (tr_step+1)*self.args.batch_size)]
            	# wav_input is numpy array
            	batch_wav = batch_wave[batch_idx]
            	# pad input array to the same length(maximum length)
            	padded_batch_wav, original_batch_length = pad_sequences(batch_wav)
            	# target_label is numpy array 
            	batch_lbl = batch_label[batch_idx]
            	# Make target to sparse tensor form to apply to ctc functions
            	sparse_batch_lbl = sparse_tensor_form(batch_lbl)
            	feed = {self.input_data: padded_batch_wav, self.targets: sparse_batch_lbl, self.seq_len: original_batch_length}
            	tr_step_loss, tr_step_ler, _ = self.sess.run([self.loss, self.ler, self.train_op], feed_dict = feed)
            	# Add summary
            	#self.writer.add_summary(summary_str, total_step)
            	train_loss += tr_step_loss*self.args.batch_size
            	train_ler += tr_step_ler*self.args.batch_size
            	print("[%d/%d] in Epoch %d, Train loss = %3.3f, Ler = %3.3f, Time per batch = %3.3f, %d steps" % (tr_step+1, trainingsteps_per_epoch, index+1, tr_step_loss, tr_step_ler, time.time()-s_time, total_step))
            	total_step += 1
            
            # Metric mean
            train_loss /= data_length
            train_ler /= data_length
            
            print('Epoch %d/%d, Training loss : %3.3f, Training LabelError : %3.3f, Time per epoch: %3.3f' % (index+1, self.args.num_epoch, train_loss, train_ler, time.time() - start_time))
            
            valid_loss = 0
            valid_ler = 0
            valid_data_length = len(valid_wav_input)
            valid_tr_step = valid_data_length // self.args.batch_size
            
            for valid_step in xrange(valid_tr_step):
				valid_batch_wav = valid_wav_input[valid_step*self.args.batch_size:(valid_step+1)*self.args.batch_size]
				valid_batch_lbl = valid_trg_label[valid_step*self.args.batch_size:(valid_step+1)*self.args.batch_size]
				padded_valid_wav, padded_valid_length = pad_sequences(valid_batch_wav)
				valid_lbl = sparse_tensor_form(valid_batch_lbl)
				
				valid_loss_, valid_ler_ = self.sess.run([self.loss, self.ler], feed_dict = {self.input_data:padded_valid_wav, self.targets:valid_lbl, self.seq_len:padded_valid_length})
				valid_loss += valid_loss_*seef.args.batch_size
				valid_ler += valid_ler_*self.args.batch_size
	
            valid_loss /= valid_data_length
            valid_ler /= valid_data_length
            print('Validation error, Loss : %3.3f, LabelError : %3.3f' % (valid_loss, valid_ler))
            if valid_loss < best_valid_loss:
				print('Validation improved from %3.4f to %3.4f' % (best_valid_loss, valid_loss))
				best_valid_loss = valid_loss 
				# Save only when validation improved
				print('Save')
				self.save(index+1)
				overfit_index = 0
            else:	
				overfit_index += 1   	
				print('Validation not improved from %3.4f at %d epochs' % (best_valid_loss, overfit_index))
			 
            self.write_log(index+1, train_loss, train_ler, valid_loss, valid_ler, start_time)
            
            if train_ler < 1e-1 and valid_ler < 0.1:
            	print('Label error rate is below 0.1 at epoch %d' % (index+1)) 
            	print('Valid error rate is below 0.2 at epoch %d' % (index+1))
            	self.save(index+1)
            	break
            
            if (overfit_index == self.args.overfit_index) or (train_ler >1e-1):	
            	partition_idx += 1
            	print('Move to %d dataset' % (partition_idx+1))
            	# To distinguish between dataset
            	self.log_file.write('\n')
            	overfit_index = 0
            	datamove_flag = 1
   
    
    @property
    def model_dir(self):
    	if self.args.dropout == 0:
    		return '{}_{}layers_{}state_{}batch_{}classes'.format(self.args.model, self.args.num_layers, self.args.state_size, self.args.batch_size, self.args.num_classes)
    	else:
    		return '{}_{}layers_{}state_{}batch_{}classes_dropout'.format(self.args.model, self.args.num_layers, self.args.state_size, self.args.batch_size, self.args.num_classes)
    
    def save(self, total_step):
    	model_name = 'RNN' 
    	# checkpoint directory
    	checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
    
    	if not os.path.exists(checkpoint_dir):
    		os.makedirs(checkpoint_dir)
    
    	self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=total_step)
    	print('Model saved at : %s' % (checkpoint_dir))
    
    def load(self):
    	print("Reading checkpoint")
    	# checkpoint directory
    	checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
    	# check model name
    	model_name = 'RNN'
    	# Restoring, the graph is exactly as it was when the variable were saved in prior run
    	# Return checkpointstate proto
    	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    	# Get checkpoint name(latest)
    	if ckpt and ckpt.model_checkpoint_path:
    		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    		self.loaded_epoch = int(ckpt_name.split('-')[1])
    		print(ckpt_name, self.loaded_epoch)
    		if ckpt_name.split('-')[0] == model_name:
    			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    			print('Success to read %s at epoch %d' % (ckpt_name, self.loaded_epoch))
    			return True
    		else:
    			return False
    	else:
    		print('Failed to find a checkpoint')
    		self.loaded_epoch = 0
    		return False
    
    def write_log(self, epoch, loss, ler, valid_loss, valid_ler, start_time):
    	print('Write logs..')
    	log_path = os.path.join(self.args.log_dir, self.model_dir+'.csv')
    	if not os.path.exists(log_path):
    		self.log_file = open(log_path, 'w')
    		self.log_file.write('Epoch\t,avg_loss\t,avg_ler\t,valid_loss\t,valid_ler\t,time\n')
    	else:
    		self.log_file = open(log_path, 'a')
    
    	self.log_file.write(str(epoch)+'\t,' + str(loss)+'\t,' + str(ler) + '\t,' + str(valid_loss) + '\t,' + str(valid_ler) + '\t,' + str(time.time()-start_time) + '\n')
    	self.log_file.flush()
    
