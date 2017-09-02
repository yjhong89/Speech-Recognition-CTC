#!/usr/bin/python
# -*- coding : utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse, time, os
import collections, cPickle
from ops import *
from ctc_model import CTC_Model
from decoder import DECODER
#from clm_model import CLM_Model


def main():
 	parser = argparse.ArgumentParser()
 	parser.add_argument('--train_wav_dir', type=str, default='/home/yjhong89/asr_dataset/ldc/wave', help='data directory containing audio clip')
 	parser.add_argument('--train_lbl_dir', type=str, default='/home/yjhong89/asr_dataset/ldc/trsp', help='data directory containing transcript')
 	parser.add_argument('--test_data_dir', type=str, default='/home/yjhong89/asr_dataset/ldc/testset', help='data directory containing audio clip and transcription')
	parser.add_argument('--valid_data_dir', type=str, default='/home/yjhong89/asr_dataset/ldc/validation')
 	parser.add_argument('--files_dir', type=str, default='./files')
	parser.add_argument('--log_dir', type=str, default='./logs')
 	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='To restore variables and model')
 	parser.add_argument('--state_size', type=int, default=512, help='size of RNN hidden state')
	parser.add_argument('--maxgrad', type=float, default=5.0)
 	parser.add_argument('--num_layers', type=int, default=3, help='number of layers in RNN')
 	parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'lnlstm', 'gru'])
 	parser.add_argument('--batch_size', type=int, default=2)
 	parser.add_argument('--num_epoch', type=int, default=2000)
 	parser.add_argument('--learning_rate', type=float, default=5e-4)
 	parser.add_argument('--is_train', type=str2bool, default='t')
 	parser.add_argument('--init_from', type=str2bool, default='t', help='Continue training from saved model') 
 	parser.add_argument('--save_interval', type=int, default=5)
 	parser.add_argument('--num_features', type=int, default=39)
 	parser.add_argument('--num_classes', type=int, default=29,\
		help='All lowercase letter, space, apstr, eos, blank : last class is always reserved for blank')
 	parser.add_argument('--dropout', type=str2bool, default='n', help='dropout')
 	parser.add_argument('--keep_prob', type=float, default=0.9)
 	parser.add_argument('--seq_length', type=int, default=200, help='number of steps')
 	parser.add_argument('--mode', type=int, default=0, help='0 for ctc, 1 for clm', choices=[0,1])
	parser.add_argument('--alpha', type=float, default=2.0, help='language model weight')
	parser.add_argument('--beta', type=float, default=1.5, help='insertion bonus')
	parser.add_argument('--beam_width', type=int, default=128)

 	args = parser.parse_args()
 	print(args)

 	if not os.path.exists(args.checkpoint_dir):
 	 	os.makedirs(args.checkpoint_dir)
 	if not os.path.exists(args.files_dir):
 	 	os.makedirs(args.files_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

 	run_config = tf.ConfigProto()
 	run_config.log_device_placement=False
 	run_config.gpu_options.allow_growth=True

 	with tf.Session(config=run_config) as sess:
		if args.is_train:
			print('Training')
			if args.mode == 0:
				model = CTC_Model(args, sess)
				model.train()
			else:
				model = CLM_Model(args, sess, infer=True)
				model.train()
		else:
			print('Decoding')	
			decoding = DECODER(args, sess, args.mode)
			decoding.decode()

    
def str2bool(v):
	if v.lower() in ('yes', 'y', 'true', 't', 1):
		return True
	elif v.lower() in ('no', 'n', 'false', 'f', 0):
		return False



if __name__ == '__main__':
 	main()













