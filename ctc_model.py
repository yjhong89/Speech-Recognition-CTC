#!/usr/bin/python
# -*- coding : utf-8 -*-

from __future__ import print_function
import numpy as np
import time, os, math, shutil
import tensorflow as tf
from ops import *
from data_loaders import *


class CTC_Model():
	def __init__(self, args, sess):
		self.args = args
		self.sess = sess
	  # Cell unit selection
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
	 
		if args.dropout is True :
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=args.keep_prob)
	
		if args.model == 'lstm' or args.model == 'lnlstm':
			self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
		else:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
	
		if args.dropout is True:
			self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=args.keep_prob)
	
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
		self.shape = tf.shape(self.input_data)
		self.batch_s = self.shape[0]
		self.step_s = self.shape[1]
		
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
	
	def __str__(self):
		return 'This model has input size {}, state size {}, {} of classes, () timesteps'.format(self.args.num_features, self.args.state_size, self.args.num_classes, self.step_s)
	
	def train(self):
		# For tensorboard, merging summaries to serialize summary probobuf
		#self.summaries = merge_summary([self.prob_summary, self.loss_summary, self.ler_summary])
		# Pass merged summary to summarywirter to save at disc, need logdir and visualize graph
		#self.writer = SummaryWriter(os.path.join('./ctc_logs', self.model_dir), self.sess.graph)
		 
		# Counting total step
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
		best_valid_loss = 0
		overfit_index = 0
		partition_idx = 0
		datamove_flag = 1
		# Loading Validation data
		valid_wav_input = np.load(os.path.join(self.args.valid_data_dir, 'waves_0.npy'))
		valid_trg_label = np.load(os.path.join(self.args.valid_data_dir, 'trans_0.npy'))
		print('%d valid input and %d valid target set loaded' % (len(valid_wav_input), len(valid_trg_label))) 

		for index in xrange(self.loaded_epoch, self.args.num_epoch):
		   	# Shuffling datas
			# shuffle index
		   	start_time = time.time()
			print("%d th epoch starts" % (index+1))
		   	train_loss = 0
		   	train_ler = 0
		   	# Newly load new data
		   	if datamove_flag:
				print('Loading dataset')
				batch_wave = np.load(os.path.join(self.args.train_wav_dir, 'waves_{}.npy'.format(partition_idx)))
				batch_label = np.load(os.path.join(self.args.train_lbl_dir, 'trans_{}.npy'.format(partition_idx)))
				print('%d-th %d wave %d target dataset loaded' % (partition_idx+1, len(batch_wave), len(batch_label)))
				data_length = len(batch_wave)
				trainingsteps_per_epoch = data_length // self.args.batch_size    
				datamove_flag = 0
		   
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
		   	valid_tr_step = len(valid_wav_input) // 100
		
		   	for valid_step in xrange(valid_tr_step):
				valid_batch_wav = valid_wav_input[valid_step*100:(valid_step+1)*100]
				valid_batch_lbl = valid_trg_label[valid_step*100:(valid_step+1)*100]
				padded_valid_wav, padded_valid_length = pad_sequences(valid_batch_wav)
				valid_lbl = sparse_tensor_form(valid_batch_lbl)
		    
				valid_loss_, valid_ler_ = self.sess.run([self.loss, self.ler], feed_dict = {self.input_data:padded_valid_wav, self.targets:valid_lbl, self.seq_len:padded_valid_length})
				valid_loss += valid_loss_
				valid_ler += valid_ler_
			print('Validation error, Loss : %3.3f, LabelError : %3.3f' % (valid_loss / valid_tr_step, valid_ler / valid_tr_step))
		   	if index == self.loaded_epoch:
		   		best_valid_loss = (valid_loss / valid_tr_step)
		
		   	if (valid_loss / valid_tr_step) < best_valid_loss:
				print('Validation improved from %3.4f to %3.4f' % (best_valid_loss, valid_loss / valid_tr_step))
				best_valid_loss = (valid_loss / valid_tr_step)
				overfit_index = 0
			else:	
				overfit_index += 1   	
				print('Validation not improved from %3.4f at %d epochs' % (best_valid_loss, overfit_index))
		     
		   	if (np.mod(index+1, self.args.save_interval) == 0) or (index == self.args.num_epoch -1):
				print('Save')
				self.save(index+1)

			self.write_log(index+1, train_loss, train_ler, valid_loss / valid_tr_step, valid_ler / valid_tr_step, start_time)

		   	if train_ler < 1e-1 and valid_ler < 0.2:
				print('Label error rate is below 0.1 at epoch %d' % (index+1)) 
				print('Valid error rate is below 0.2 at epoch %d' % (index+1))
				self.save(index+1)
				break
		   
		   	if overfit_index == 50:	
				partition_idx += 1
				print('Move to %d dataset' % (partition_idx+1))
				# To distinguish between dataset
				self.log_file.write('\n')
				overfit_index = 0
				datamove_flag = 1
		   	if partition_idx == 50:
				partition_idx = 0
			print('%d epoch finished' % (index+1))
   

	@property
	def model_dir(self):
		if self.args.dropout == 0:
			return '{}_{}layers_{}state_{}batch_{}classes_ctc'.format(self.args.model, self.args.num_layers, self.args.state_size, self.args.batch_size, self.args.num_classes)
		else:
			return '{}_{}layers_{}state_{}batch_{}classes_dropout_ctc'.format(self.args.model, self.args.num_layers, self.args.state_size, self.args.batch_size, self.args.num_classes)
  
	def save(self, total_step):
  		model_name = 'acousticmodel' 
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
		model_name = 'acousticmodel'
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


	def acoustic_decode(self):
		# Get test wavfile/label, 100 sets
		for_test = SpeechLoader(self.args.test_data_dir, self.args.num_features, self.args.num_classes)  
		test_wav = for_test.mel_freq
		test_lbl = for_test.target_label
		# For placeholder
		test_wav_padded, test_wav_length = pad_sequences(test_wav)
		test_sparse_lbl = sparse_tensor_form(test_lbl)
		
		self.args.dropout = False
		self.sess.run(tf.global_variables_initializer())
		# Load checkpoint path
		if self.load():
			print('Loaded checkpoint')
		else:
			print('Load failed')
		  # For tensorboard
		self.decode_summary = merge_summary([self.prob_summary])
		self.decode_writer = SummaryWriter(os.path.join('./decode_ctc_logs', self.model_dir), self.sess.graph)
		#  sample_rate, audio_data = wav.read(wavfile)
		#  # [step_size, num_features]
		#  audio_input = mfcc(audio_data, samplerate=sample_rate, numcep=self.args.num_features)
		#  # Make batchsize 1 to feed input placeholder
		#  placeholder_i = audio_input[np.newaxis, :]
		#  # Normalize
		#  placeholder_input = (placeholder_i - np.mean(placeholder_i))/np.std(placeholder_i)
		#  # [batch_size,]
		#  sequence_len = np.asarray([placeholder_input.shape[1]])
		  
		ctc_file = os.path.join(self.args.files_dir, self.model_dir) 
		  
		# Do not need to feed targets
		feed = {self.input_data : test_wav_padded, self.targets:test_sparse_lbl, self.seq_len : test_wav_length}
		# char_prob is ctc probability(emitting probability), has shape of batch_size, step_size, num_classes
		# batch_size is 1 since just 1 wav file is acquired
		# Decode has a sparse tensor form(indices, values, shape)
		char_prob, decoded, summary_str, ler_ = self.sess.run([self.probability, self.decoded[0], self.decode_summary, self.ler], feed_dict = feed)
		#self.decode_writer.add_summary(summary_str)
		# Has a shape of batch_size, num_step, num_classes
		char_prob = np.asarray(char_prob)
		# To use at beam decoding
		#  with open(ctc_file, 'w') as f:
		#   cPickle.dump(char_prob, f)
		#   f.close()
		high_index = np.argmax(char_prob, axis=2)
		print(decoded)
		print('Label Error rate : %3.4f' % ler_)
		# CTC decode function returns sparse tensor(indices, values, shape)
		str_decoded = ''.join([chr(x) for x in np.asarray(decoded[1]) + SpeechLoader.FIRST_INDEX])
		# Change to label using value tensor
		if self.args.num_classes == 30:
			print('Number of classes %d' % self.args.num_classes)
			# 0:Space, 27:Apstr, 28:<EOS>, SpeechLoader.FIRSTINDEX=96(ord('a')=97), last class:blank
			# For blank, last class
			str_decoded = str_decoded.replace(chr(ord('z')+3), "'")
			# For space
			str_decoded = str_decoded.replace(chr(ord('a')-1), ' ')
			# For Apstr
			str_decoded = str_decoded.replace(chr(ord('z')+1), "'")
			# For EOS
			str_decoded = str_decoded.replace(chr(ord('z')+2), '.')
			print(str_decoded)
			#  for a in range(20):
			#   high_prob_chr = [SpeechLoader.SPACE_TOKEN if x == ord('a')-1 else SpeechLoader.APSTR_TOKEN if x == ord('z')+1 else SpeechLoader.EOS_TOKEN if x == ord('z')+2 else SpeechLoader.BLANK_TOKEN if x == ord('z')+3 else chr(x) for x in high_index[a, :] + SpeechLoader.FIRST_INDEX]
			#   print(high_prob_chr)
			#  print('Decoded: %s' %str_decoded)
 
# def beam_decode(args, alpha=2, beta=1.5, beam_width=128):
#  # Getting ctc probability and language probability dictionary
#  # checkpoint for language model
#  # Getting clm prob dictinary from lm checkpoint and label predict function
#  # Would be {'prompt1': {prob_dict for prompt1}, 'prompt2': {prob_dict for prompt2}, .....}
#  clm_file = os.path.join(args.files_dir, 'clm.pkl')  
#  ctc_file = os.path.join(args.files_dir, 'ctc.pkl')
#  # Loading ctc prob by loading pickle
#  if not os.path.exists(ctc_file):
#   print ('CTC file does not exist')
#   print(os.path.abspath(ctc_file))
#  if not os.path.exists(clm_file):
#   print('CLM file does not exists')
#   print(os.path.abspath(clm_file))
#  # Has shape of 1 * num step * num classes
#  with open(ctc_file, 'r') as f:
#   ctc_prob = cPickle.load(f)
#   f.close()
#  with open(clm_file, 'r') as f:
#   clm_dict = cPickle.load(f)
#   f.close()
# 
#  # Make ctc dimension to num step * num classes(2 dimension)
#  ctc_prob = np.squeeze(ctc_prob)
#  # Get time step T
#  T = ctc_prob.get_shape()[0].value
#  # Get number of labels
#  num_labels = ctc_prob.get_shape()[1].value
#  # float('-inf') is used to represent log0
#  init_hyp = Hypothesis(float('-inf'), float('-inf'), 0)
#  
#  # Dictionary which stores 'prefix:probability' not in beam
#  beam_old = collections.defaultdict(init_hyp)
#  # Beam cutoff, input would be prefix prob dictionary
#  prev_prob = lambda x:Hypothesis.exp_sum_log(x[1].p_nb+x[1].p_b) + beta*x[1].prefix_len
#  
#  for t in xrange(T+1):
#   # Initialize
#   if t == 0:
#    beam_cut = dict()
#    # Empty prefix, store probability structure(class Hypothesis)
#    # p_nb = 0, p_b = 1, prefix_len = 0
#    # Empty tuple
#    beam_cut[()] = Hypothesis(float('-inf'), 0, 0)
#   else:
#    print(t)
#    print('-'*10)
#    # Beam cutoff by beamwidth
#    beam_sorted = sorted(beam.items(), key=prev_prob, reverse=True)
#    # Would be of tuples so make it to dictionary
#    beam_cut = dict(beam_sorted[:beam_width])
#    beam_old = beam
#   
#   # Make dictionary for each timestep, beam <- {}
#   beam = collections.defaultdict(init_hyp)
#   # for prefix string in beam cutoff at 't-1'
#   for pfx_str, hyps in beam_cut.items():
#    new_hyp = beam[pfx_str]
#    # Get prefix length
#    pfx_len = hyps.prefix_len
#    # p_total = p_nb + p_b
#    p_tot = Hypothesis.exp_sum_log(hyps.p_b, hyps.p_nb)
#    # if string is not empty 
#    if pfx_len > 0:
#     # Get last prefix, would be character -> need to change to index
#     pfx_end = pfx_str[pfx_len-1]
#     pfx_end_idx = Hypothesis.label_to_index(pfx_end)
#     # p_nb update : p_nb(string, t) = p_nb(string, t-1) * ctc_prob(string end, t)
#     new_hyp.p_nb = ctc_prob[t-1, pfx_end_idx] + hyps.p_nb
#     # Handle repeat character collapsing
#     # y_hat prefix of y with the last label removed
#     y_hat = pfx_str[:pfx_len-1]
#     # if y_hat in beam_cut
#     if y_hat in beam_cut:
#      prev_hyp = beam_cut[y_hat]
#     else:
#      prev_hyp = beam_old[y_hat]
# 
#     # Define extension probability of prefix string y by label k at time t
#     # pr(k,y,t) = ctc_prob(k, t | x) * transision prob(clm prob) * collabse_prob
#     # here is pr(pfx_end, y_hat, t)
#     if len(y_hat) == 0:
#      extension_prob = ctc_prob[t-1, pfx_end_idx]
#     else:
#      extension_prob = ctc_prob[t-1, pfx_end_idx] + clm_dict[y_hat[-1]][pfx_end]   
#     # y_hat_pfx_end == pfx_str_end ===> handle repeat character
#     # Need pfx_len above 1 to be meaningful
#     if (pfx_len > 1) and (pfx_end == y_hat[-1]):
#      extension_prob += prev_hyp.p_b
#     else:
#      extension_prob += Hypothesis.exp_sum_log(prev_hyp.p_nb, prev_hyp.p_b)
#     # For non_blank 
#     new_hyp.p_nb = exp_sum_log(new_hyp.p_nb, extension_prob)
#     
#    # Handling blank
#    new_hyp.p_b = hyps.p_b + ctc_prob[t-1, 29]
# 
#    new_hyp.prefix_len = pfx_len
#    # Add pfx_str to beam
#    beam[pfx_str] = new_hyp
#    
#    # loop for character to extend prefix string except blank, blank is reserved as last class
#    for k in xrange(num_labels-1):
#     extension_hyp = Hypothesis(0,0,0)
#     # We except blank
#     extension_hyp.p_b = float('-inf')    
#     extension_hyp.prefix_len = len(pfx_len) + 1
#     # p_nb(y+k, t) <- pr(k, y, t)
#     extended_pfx = tuple(list(pfx_str)+[k])
#     extended_pfx_label = Hypothesis.index_to_label(k)
#     extension_hyp.p_nb = ctc_prob[t-1, k] + alpha*clm_dict[pfx_str[-1]][extended_pfx_label]
#     if pfx_len > 0:
#      pfx_end = pfx_str[pfx_len-1]
#      pfx_end_idx = Hypothesis.label_to_index(pfx_end)
#      if pfx_end_idx == k:
#       extension_hyp.p_nb += hyps.p_b
#     else:
#      extension_hyp.p_nb += p_tot
# 
#     beam[extended_pfx] = extension_hyp
# 
#  beam_final = sorted(beam.items(), key=prev_prob, reverse=True)
#  return beam_final
#     
