import numpy as np
import tensorflow as tf
from ops import *
import time

class Wavenet_Model():
    def __init__(self, args, sess):
    	self.args = args
    	self.sess = sess
    
    	self.create_model()
    
    
    def create_model(self):
        # Placeholders
        # [batch size, step, features]
        self.inputs = tf.placeholder(tf.float32, [None, None, self.args.num_features], name='wave_input')
        self.targets = tf.sparse_placeholder(tf.int32, name='target')
        self.seq_len = tf.placeholder(tf.int32, [None], name='sequence_length')
    
        skip = 0
        '''
        	Construct of a stack of dilated causal convolutional layers
        '''
        # First non-causal convolution to inputs to expand feature dimension
        h = conv1d(self.inputs, self.args.num_hidden, filter_width=self.args.filter_width, name='conv_in', normalization=self.args.layer_norm, activation=tf.nn.tanh)
        # As many as number of blocks, block means one total dilated convolution layers
        for blocks in range(self.args.num_blocks):
        	# Construction of dilation
        	for dilated in range(self.args.num_wavenet_layers):
        		# [1,2,4,8,16..]
        		rate = 2**dilated 
        		h, s = res_block(h, self.args.num_hidden, rate, self.args.causal, self.args.filter_width, normalization=self.args.layer_norm, activation=self.args.dilated_activation, name='{}block_{}layer'.format(blocks+1, dilated+1))
        		skip += s
        # Make skip connections
        with tf.variable_scope('postprocessing'):
        	# 1*1 convolution
        	skip = conv1d(tf.nn.relu(skip), self.args.num_hidden, filter_width=self.args.filter_width, activation=tf.nn.relu, normalization=self.args.layer_norm, name='conv_out1')
        	hidden = conv1d(skip, self.args.num_hidden, filter_width=self.args.filter_width, activation=tf.nn.relu, normalization=self.args.layer_norm, name='conv_out2')
        	self.logits = conv1d(hidden, self.args.num_classes, filter_width=1, activation=None, normalization=self.args.layer_norm, name='conv_out3')
        
        # To calculate ctc, consider timemajor
        self.logits_reshaped = tf.transpose(self.logits, [1,0,2])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.targets, inputs=self.logits_reshaped, sequence_length=self.seq_len))
        self.ctc_decoded, _ = tf.nn.ctc_greedy_decoder(self.logits_reshaped, self.seq_len)	
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.ctc_decoded[0], tf.int32), self.targets))
        # When use tf.contrib.layers.layer_norm(batch_norm), update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
            #self.train_op = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.loss)
        trainable_vr = tf.trainable_variables()
        for i in trainable_vr:
            print(i.name)
        optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        grad, vrbs = zip(*optimizer.compute_gradients(self.loss))
        # clip_by_global_norm returns (list_clipped, global_norm), global_norm is sum of total l2 gradient(inputs)
        # If global_norm(sum of total norm) is greater than clipnorm(maxgrad), each element of input shrunk as ratio of global norm
        # Right way of gradient clipping
        # Automatically ignore None gradient
        grads, _ = tf.clip_by_global_norm(grad, self.args.maxgrad)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(zip(grads, vrbs))	
        
        self.saver = tf.train.Saver()
    
    
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        
        if self.load():
        	print('Checkpoint loaded')
        else:
        	print('Load failed')
        
        total_step = 1
        datamove_flag = 1
        
        for index in range(0, self.args.num_epoch):
            start_time = time.time()
            print('%d th epoch start' % (index+1))
            train_loss = 0
            train_ler = 0
            # Move to new data
            if datamove_flag :
            	inputs_wave = np.load(os.path.join(self.args.train_wav_dir, 'wave_{}.npy'.format(self.data_index+1)), encoding='bytes')
            	inputs_label = np.load(os.path.join(self.args.train_lbl_dir, 'tran_{}.npy'.format(self.data_index+1)), encoding='bytes')
            	print('%d-th %d wave %d target loaded' % (self.data_index+1, len(inputs_wave), len(inputs_label)))
            	train_index = int(len(inputs_wave) * 0.9)
            	print('Train index : %d' % train_index)
            	training_step_per_epoch = train_index // self.args.batch_size
            	wave_train = inputs_wave[:train_index]
            	label_train = inputs_label[:train_index] 
            	wave_valid = inputs_wave[train_index:len(inputs_wave)]
            	label_valid = inputs_label[train_index:len(inputs_label)]
            	best_valid_loss = 1000
            	best_valid_ler = 1000
            	datamove_flag = 0
            # Permutation to get regularize effect
            perm_index = np.random.permutation(len(wave_train))
            wave_perm = wave_train[perm_index]
            label_perm = label_train[perm_index]
            
            for tr_step in range(training_step_per_epoch):
            	s_time = time.time()
            	# wave_input is numpy array
            	batch_wave = wave_perm[tr_step*self.args.batch_size : (tr_step+1)*self.args.batch_size]
            	# pad input array to the same length(maximum length)
            	padded_batch_wav, original_batch_length = pad_sequences(batch_wave)
            	# target_label is numpy array 
            	batch_label = label_perm[tr_step*self.args.batch_size : (tr_step+1)*self.args.batch_size]
            	# Make target to sparse tensor form to apply to ctc functions
            	sparse_batch_lbl = sparse_tensor_form(batch_label)
            	feed = {self.inputs: padded_batch_wav, self.targets: sparse_batch_lbl, self.seq_len: original_batch_length}
            	tr_step_loss, tr_step_ler, _ = self.sess.run([self.loss, self.ler, self.train_op], feed_dict = feed)
            	# Add summary
            	#self.writer.add_summary(summary_str, total_step)
            	train_loss += tr_step_loss*self.args.batch_size
            	train_ler += tr_step_ler*self.args.batch_size
            	print("[%d/%d] in Epoch %d, Train loss = %3.3f, Ler = %3.3f, Time per batch = %3.3f, %d steps" % (tr_step+1, training_step_per_epoch, index+1, tr_step_loss, tr_step_ler, time.time()-s_time, total_step))
            	total_step += 1
            
            train_loss /= len(wave_train)
            train_ler /= len(wave_train)
            
            print('Epoch %d/%d, Training loss : %3.3f, Training LabelError : %3.3f, Time per epoch: %3.3f' % (index+1, self.args.num_epoch, train_loss, train_ler, time.time() - start_time))
            
            valid_loss, valid_ler = self.evaluate(wave_valid, label_valid)
            self.write_log(index+1, train_loss, train_ler, valid_loss, valid_ler, start_time)				
            
            # Save only when validation improved
            if valid_ler < best_valid_ler:
            	print('Validation improved from %3.4f to %3.4f' % (best_valid_ler, valid_ler))
            	best_valid_loss = valid_loss
            	best_valid_ler = valid_ler
            	self.save(self.data_index)
            	overfit_index = 0
            else:
                overfit_index += 1
            
            if train_ler < 1e-1 and valid_ler < 0.2:
            	print('Label error rate is below 0.1 at epoch %d' % (index+1)) 
            	print('Valid error rate is below 0.2 at epoch %d' % (index+1))
            	self.save(self.data_index)
            	break
            
            if (overfit_index == self.args.overfit_index) or (train_ler < 1e-1):	
            	self.data_index += 1
            	print('Move to %d dataset' % (self.data_index+1))
            	# To distinguish between dataset
            	self.log_file.write('\n')
            	overfit_index = 0
            	datamove_flag = 1
            	best_valid_loss = 1000
               
    
    def evaluate(self, wave, label):
    	valid_loss = 0
    	valid_ler = 0
    	valid_tr_step = len(wave) // self.args.batch_size
    	for valid_step in range(0, valid_tr_step):
    		valid_batch_wav = wave[valid_step*self.args.batch_size : (valid_step+1)*self.args.batch_size]
    		valid_batch_lbl = label[valid_step*self.args.batch_size : (valid_step+1)*self.args.batch_size]
    
    		padded_valid_wav, padded_valid_length = pad_sequences(valid_batch_wav)
    		sparse_valid_lbl = sparse_tensor_form(valid_batch_lbl)
    		valid_loss_, valid_ler_ = self.sess.run([self.loss, self.ler], feed_dict={self.inputs:padded_valid_wav, self.targets:sparse_valid_lbl, self.seq_len:padded_valid_length})
    		valid_loss += valid_loss_*self.args.batch_size
    		valid_ler += valid_ler_*self.args.batch_size
    	valid_loss /= len(wave)
    	valid_ler /= len(wave) 
    	print('Validation error, Loss : %3.3f, LabelError : %3.3f' % (valid_loss, valid_ler))
    	return valid_loss, valid_ler
    			
    
    @property
    def model_dir(self):
    	if self.args.layer_norm:
    		return '{}blocks_{}layers_{}width_{}_ln'.format(self.args.num_blocks, self.args.num_wavenet_layers, self.args.filter_width, self.args.dilated_activation)
    	else:
    		return '{}blocks_{}layers_{}width_{}'.format(self.args.num_blocks, self.args.num_wavenet_layers, self.args.filter_width, self.args.dilated_activation)
    
    def save(self, data_index):
    	model_name = 'Wavenet'
    	# checkpoint directory
    	checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
    	if not os.path.exists(checkpoint_dir):
    		os.makedirs(checkpoint_dir)
    
    	self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=data_index)
    	print('Model saved at : %s' % (checkpoint_dir))
    
    def load(self):
    	# checkpoint directory
    	checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
    	# check model name
    	model_name = 'Wavenet'
    	# Restoring, the graph is exactly as it was when the variable were saved in prior run
    	# Return checkpointstate proto
    	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    	# Get checkpoint name(latest)
    	if ckpt and ckpt.model_checkpoint_path:
    		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    		self.data_index = int(ckpt_name.split('-')[1])
    		print(ckpt_name, self.data_index)
    		if ckpt_name.split('-')[0] == model_name:
    			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    			print('Success to read %s, %d dataset' % (ckpt_name, self.data_index))
    			return True
    		else:
    			return False
    	else:
    		print('Failed to find a checkpoint')
    		self.data_index = 0
    		return False
	
    def write_log(self, epoch, loss, ler, valid_loss, valid_ler, start_time):
    	print('Write logs..')
    	log_path = os.path.join(self.args.log_dir, self.model_dir+'.csv')
    	if not os.path.exists(log_path):
    		self.log_file = open(log_path, 'w')
    		self.log_file.write('Epoch,\tavg_loss,\tavg_ler,\tvalid_loss,\tvalid_ler,\ttime\n')
    	else:
    		self.log_file = open(log_path, 'a')
    
    	self.log_file.write('%d,\t%3.4f,\t%3.4f,\t%3.4f,\t%3.4f,\t%3.4f sec\n' % (epoch, loss, ler, valid_loss, valid_ler, time.time()-start_time))
    	self.log_file.flush()
