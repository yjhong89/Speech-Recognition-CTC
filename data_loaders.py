#!/usr/bin/python
# -*- coding : utf-8 -*-

import numpy as np
import tensorflow as tf
import os, math
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import codecs
import collections
from six.moves import cPickle

class TextLoader():
    def __init__(self, data_path, seq_length, batch_size, encoding='utf-8'):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.encoding = encoding
        
        input_file = os.path.join(data_path, 'cantab_noeos.txt')
        vocab_file = os.path.join(data_path, 'vocab.pkl')
        tensor_file = os.path.join(data_path, 'data.npy')
        
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print('Reading text file')
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print('Load files')
            self.load_preprocessed(vocab_file, tensor_file)
        
        self.create_batches()
        self.reset_batch_pointer()
    
    # Read input.txt, get vocabulary used at least once
    def preprocess(self, input_file, vocab_file, tensor_file):
        if not os.path.exists(input_file):
            print('File does not exist')
            print(os.path.abspath(input_file))
        
        with open(input_file, 'r') as f:
            # Read all input.txt include \r, \n
            raw_data = f.read()
        # Label process
        raw_data = raw_data.lower()
        # collections.Counter returns Counter type which has dictionary form. And count the number of each element used in input argument
        counter = collections.Counter(raw_data)
        print(counter)
        # counter.items() returns a list of (element, count) pairs
        # key in sorted function do sorting operation. In this case, reverse order with count element for each pair
        # sorting element in the order of count
        count_pairs = sorted(counter.items(), key=lambda x:-x[1])
        # '*' operator decompose element, count element so we can get only vocabulary typle by zip(*count_pair)
        # [(1,'a'), (2,'b'), (3,'c')] => [(1,2,3), ('a','b','c')]
        self.chars, _ = zip(*count_pairs) 
        # Get vocabulary size
        self.vocab_size = len(self.chars)
        # Make dictionary to get index for each vocabulary used
        self.vocab_to_idx = dict(zip(self.chars, range(len(self.chars))))
        self.idx_to_vocab = dict(enumerate(self.chars))
        with open(vocab_file, 'wb') as f:
        	cPickle.dump(self.chars, f)
        # Convert each char in input.txt into index list and make it  numpy array
        self.tensor = np.array(list(map(self.vocab_to_idx.get, raw_data)))
        np.save(tensor_file, self.tensor)
    
    def load_preprocessed(self, vocab_file, tensor_file):
     	with open(vocab_file, 'rb') as f:
     		self.chars = cPickle.load(f)
     	self.vocab_size = len(self.chars) 
     	self.vocab_to_idx = dict(zip(self.chars, range(len(self.chars))))
     	self.idx_to_vocab = dict(enumerate(self.chars))
     	self.tensor = np.load(tensor_file)
     	# Number of batches
     	self.num_bathes = int(self.tensor.size / (self.batch_size * self.seq_length))
    
    def create_batches(self):
     	self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
     	if self.num_batches == 0:
      		Exception('Lower batch_size and seq_length')
    
     	# Remove redundant element in self.tensor
     	self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
     	# Character language model, predict next character given present character
     	x_data = self.tensor
     	y_data = np.copy(self.tensor)
     	y_data[:-1] = self.tensor[1:]
     	y_data[-1] = 0
     	# Reshaping batches to fit place holder [batch_size * seq_length(num_step)]
     	self.x_batches = np.split(x_data.reshape(self.batch_size, -1), self.num_batches, 1)
     	self.y_batches = np.split(y_data.reshape(self.batch_size, -1), self.num_batches, 1)
    
    def next_batch(self):
        batch_x, batch_y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return batch_x, batch_y
     
    def reset_batch_pointer(self):
     	self.pointer = 0


class SpeechLoader():
    # Define class constant
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    APSTR_TOKEN = '<apstr>'
    APSTR_INDEX = 27 
    EOS_TOKEN = '<eos>'
    EOS_INDEX = 28 
    PUNC_TOKEN = '<punc>'
    BLANK_TOKEN = '<blank>'
    # ord('a') = 97
    FIRST_INDEX = ord('a') - 1
    def __init__(self, data_path, num_features, num_classes):
        print('Data from : %s' % data_path)
        self.num_classes = num_classes
        self.num_features = num_features
        self.filelist = list()
        self.wavfilelist = list()
        self.txtfilelist = list()
        self.eachlabellist = list()
        self.mel_freq = list()
        self.target_label = list()
        self.length_check = list()
        print('Number of classes %d, First INDEX : %d' % (self.num_classes, SpeechLoader.FIRST_INDEX))
        
        
        ''' Returning each file`s absolute path
        path : directory path from root
        dir : directory list below root, if there is no directory returns empty list
        file  : file list below root, if there is no file returns empty list
        '''
        for path, dirs, files in os.walk(data_path):
        	fullpath = os.path.join(os.path.abspath(data_path), path)
        	for f in files:
        		filepath = os.path.join(fullpath, f)
        		self.filelist.append(filepath)
        
        for x in self.filelist:
        	if x[-3:] == 'wav':	
        		self.wavfilelist.append(x)
        	elif x[-3:] == 'txt':
        		self.txtfilelist.append(x)
        	else:
        		print(x)
        		raise Exception('Not wanted file type')
        self.num_files = len(self.wavfilelist)
        
        self.wavfilelist.sort()
        self.txtfilelist.sort()
        print('Number of wav files : %d, Number of txt files : %d' % (len(self.wavfilelist), len(self.txtfilelist)))
        
        # Check sequence each wav,label file
        try:
        	for a, b in zip(self.wavfilelist, self.txtfilelist):
        		wavname = a.split('-')[0]+a.split('-')[-1][:-4]
        		txtname = b.split('-')[0]+b.split('-')[-1][:-4]
        	if wavname == txtname:
        		pass
                                #print(wavname)
        except:
        	raise Exception('Files do not match')
        
        # Pasing each line to represent each label
        for each_text in self.txtfilelist:
        	f = open(each_text, 'r')
        	while True:
        		each_line = f.readline()
        		if not each_line:
        			break
        		self.eachlabellist.append(' '.join(each_line.split()))
        	f.close()
        
        self.get_num_examples(self.wavfilelist, self.eachlabellist, self.num_files, self.num_features)
        self.mel_freq = np.asarray(self.mel_freq)
        self.target_label = np.asarray(self.target_label)
        if len(self.length_check) != 0:
        	print(self.length_check)
        	print('Data prprocess not done for %d' % len(self.length_check))
        	raise Exception('input is longer than output')
        print('Data preprocess done')
    
    def get_num_examples(self, wavlists, labellists, num_examples, num_features):
        for n,(w, l) in enumerate(zip(wavlists, labellists)):
            fs, au = wav.read(w)
            # Extract Spectrum of audio inputs
            melf = mfcc(au, samplerate = fs, numcep = self.num_features, winlen=0.025, winstep=0.01, nfilt=self.num_features)
            #melf = (melf - np.mean(melf))/np.std(melf)
            self.mel_freq.append(melf)
            melf_target = self.labelprocessing(l)
            self.target_label.append(melf_target)
            if n == num_examples - 1:
            	break
            if melf.shape[0] <= len(melf_target):
            	t = w,l
            	self.length_check.append(t) 
      
     # Split transcript into each label
    def labelprocessing(self, labellist):
        # Label preprocessing
        label_prep = labellist.lower().replace('[', '')
        label_prep = label_prep.replace(']', '')
        label_prep = label_prep.replace('{', '')
        label_prep = label_prep.replace('}', '')
        label_prep = label_prep.replace('-', '')
        label_prep = label_prep.replace('noise', '')
        label_prep = label_prep.replace('vocalized','')
        label_prep = label_prep.replace('_1', '')
        label_prep = label_prep.replace('  ', ' ')
        trans = label_prep.replace(' ','  ').split(' ')
        labelset = [SpeechLoader.SPACE_TOKEN if x == '' else list(x) for x in trans]
        if self.num_classes == 30:
        	labelset.append(SpeechLoader.EOS_TOKEN)
        	# Make array of each label
        for sentence in labelset:
        	for each_idx, each_chr in enumerate(sentence):
        		if each_chr == "'":
        			sentence[each_idx] = SpeechLoader.APSTR_TOKEN
        labelarray = np.hstack(labelset)
        # Transform char into index
        # including space, apostrophe, eos
        if self.num_classes == 30:
        	train_target = np.asarray([SpeechLoader.SPACE_INDEX if x == SpeechLoader.SPACE_TOKEN else SpeechLoader.APSTR_INDEX \
        		if x == SpeechLoader.APSTR_TOKEN else SpeechLoader.EOS_INDEX if x == SpeechLoader.EOS_TOKEN \
        		else ord(x) - SpeechLoader.FIRST_INDEX for x in labelarray])
        #Including space, apostrophe
        elif self.num_classes == 29:
        	train_target = np.asarray([SpeechLoader.SPACE_INDEX if x == SpeechLoader.SPACE_TOKEN else SpeechLoader.APSTR_INDEX if x == SpeechLoader.APSTR_TOKEN else ord(x) - SpeechLoader.FIRST_INDEX for x in labelarray])
        
        return train_target
    
    def save(self, save_idx):
    	print('Save as numpy')
    	num_files = len(self.mel_freq) // save_idx
    	for save_index in range(0, num_files):
    		np.save('wave_{}.npy'.format(save_index+1), self.mel_freq[save_index*save_idx: (save_index+1)*save_idx])
    		np.save('tran_{}.npy'.format(save_index+1), self.target_label[save_index*save_idx: (save_index+1)*save_idx])
    		print('{} data complete'.format(save_index+1))
    	print('Finish')
		

if __name__ == "__main__":
	a = SpeechLoader('/home/yjhong89/trainingset', 39, 29)
	a.save(1000)
