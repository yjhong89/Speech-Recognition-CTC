import numpy as np
import tensorflow as tf
import time, os, cPickle, collections
from data_loaders import SpeechLoader
from ops import *

class DECODER():
	def __init__(self,args, sess, mode=0):
		self.args = args
		self.sess = sess
		# Get trained model and load
		self.model = CTC_Model(self.args, self.sess)
		self.sess.run(tf.global_variables_initializer())
		if self.model.load():
			pass
		else:
			raise Exception('Checkpoint not loaded')
		
		# Use language model for decoding
		if mode:
			self.clm_file = os.path.join(self.args.files_dir, 'clm.pkl')


	def decode(self):
		# Get test wavefile/label
		test_set = SpeechLoader(self.args.test_dir, self.args.num_features, self.args.num_class)
		test_wave = test_set.mel_freq
		test_lbl = test_set.target_label
		test_wave_padded, test_wave_length = pad_sequences(test_wave)
		test_sparse_lbl = sparse_tensor_form(test_lbl)
	
		feed_dict = {self.model.input_data:test_wave_padded, self.model.targets:test_sparse_lbl, self.model.seq_len:test_wave_length}
		# CTC decode function returns sparse tensor(indices, values, shape)
		char_prob, decoded, ler_ = self.sess.run([self.model.probability, self.model.decoded, self.model.ler], feed_dict=feed_dict)
		decoded_original = reverse_sparse_tensor(decoded[0])
		print('Label Error rate : %3.4f' % ler_)
		
		# Save ctc probability for beam decode
		self.ctc_file = os.path.join(self.args.files_dir, self.model.model_dir)
		with open(self.ctc_file, 'w') as f:
			cPickle.dump(char_prob, f)
			f.close

		# [batch size, number of steps, number of classes]
		char_prob = np.asarray(char_prob)
		# Get greedy index
		high_index = np.argmax(char_prob, axis=2)
	
		str_decoded = list()
		for i in xrange(len(decoded_original)):
			str_decoded.append(''.join([chr(x) for x in np.asarray(decoded_original[i]) + SpeechLoader.FIRST_INDEX]))
			if self.args.num_classes == 30:
				# 0:Space, 27:Apstr, 28:<EOS>, last class:blank
				str_decoded[i] = str_decoded[i].replace(chr(ord('z')+3), "")
				str_decoded[i] = str_decoded[i].replace(chr(ord('a')-1), ' ')
				str_decoded[i] = str_decoded[i].replace(chr(ord('z')+1), "'")
				str_decoded[i] = str_decoded[i].replace(chr(ord('z')+2), '.')
			elif self.args.num_classes == 29:
				# 0:Space, 27:Apstr, last class:blank
				str_decoded[i] = str_decoded[i].replace(chr(ord('z')+2), "")
				str_decoded[i] = str_decoded[i].replace(chr(ord('a')-1), ' ')
				str_decoded[i] = str_decoded[i].replace(chr(ord('z')+1), "'"
			print(str_decoded[i])
				

	def beam_decode(self):
		'''
			Getting ctc probability and language probabiltiy dictionary
			Getting clm probability dictionary which predict future label
		'''
		# Load files
		with open(self.ctc_file, 'r') as f:
			ctc_prob = cPickle.load(f)
			f.close()
		# {'prompt1':{prob dict for prompt1}, 'prompt2':{prob dict for pormpt2}....}
		with open(self.clm_file, 'r') as f:
			clm_dict = cPickle.load(f)
			f.close()

		prev_prob = lambda x:Hypothesis.exp_sum_log(x[1].p_nb, x[1].p_b) + self.args.beta*x[1].prefix_len

		beam_decoded = list()

		for i in xrange(len(ctc_prob)):
			# Squeeze to 2 dimension, [time stpes, classes]
			ctc_p = ctc_prob[i]
			# Get time step T
			T = ctc_p.get_shape()[0].value
			# Get number of labels
			num_labels = ctc_p.get_shape()[1].value
			# float('-inf') is used to represent log0
			# Here, we deal with log probability
			init_hyp = Hypothesis(float('-inf'), float('-inf'), 0) 	
			beam_old = collections.defaultdict(init_hyp)
				
			for t in xrange(T+1):
				# Initialize
				if t == 0:
					beam_cut = dict()
					# Empty prefix, p_b = 1, p_nb = 0, prefix len = 0
					beam_cut[()] = Hypothesis(float('-inf'), 0, 0)
				else:
					print(t)
					print('-'*10)
					# Beam cutoff by beamwidth
					beam_sorted = sorted(beam.items(), key=prev_prob, reverse=True)
					beam_cut = dict(beam_sorted[:self.args.beam_width])
				
				# Make dictionary for each time step
				beam = collections.defualtdict(init_hyp)
				# For prefix string in beam cutoff at 't-1'
				for pfx_str, hyps in beam_cut.items():
					new_hyp = beam[pfx_str]
					# Get prefix length
					pfx_len = hyps.prefix_len
					# Total prob = p_nb + p_b
					p_total = Hypothesis.exp_sum_log(hyps.p_b, hyps.p_nb)
					# If string is not empty
					if pfx_str > 0:
						# Get last prefix
						pfx_end = pfx_str[pfx_len - 1]
						pfx_end_idx = Hypothesis.label_to_index(pfx_end)
						# p_nb update : p_nb(string, t) = p_nb(string, t-1) * ctc_prob(string end, t), log makes multiplying into summation
						new_hyp.p_nb = ctc_p[t-1], pfx_end_idx] + hyps.p_nb	
			 			# Handle repeat character collapsing
						# y_hat prefix of y with the last label lamoved
						y_hat = pfx_str[:pfx_len-1]
						if y_hat in beam_cut:
							prev_hyp = beam_cut[y_hat]
						# Beam old members are not included in beam cut
						else:
							prev_hyp = beam_old[y_hat]

						'''
							Define extension probability of prefix string y by label k at time t
							pr(k,y,t) = ctc_prob[k, t|y) * transition prob(clm) * collapse prob
							collapse prob = p_b(y, t-1) if y_hat == prefix_end else p_total(y_t-1)
							Here, pr(pfx_end, y_hat, t)
						'''
						if len(y_hat) == 0:
							extension_prob = ctc_p[t-1, pfx_end_idx]
						else:
							# Log makes summation
							extension_prob = ctc_p[t-1, pfx_end_idx] + clm_dict[y_hat[-1]][pfx_end]
						# y_hat pfx_end == pfx_str_end => Repeat character
						if pfx_len > 1 and pfx_end == y_hat[-1]:
							# Repeated characters have blanks between
							extension_prob += prev_hyp.p_b
						else:
							extension_prob += Hypothesis.exp_sum_log(prev_hyp.p_nb, prev_hyp.p_b)
						new_hyp.p_nb = Hypothesis.exp_sum_log(new_hyp.p_nb, extension_prob)

					# Handling blanks
					# p_b at t = ctc_prob(blank, t) * pr(y, t-1)
					new_hyp.p_b = Hypothesis.exp_sum_log(p_total, ctc_p[t-1, self.args.num_classes-1)		
					# 't-1' to 't' but length remains same since repeated character or blank
					new_hyp.prefix_len = pfx_len
					beam[pfx_str] = new_hyp

					# Loop for character to extend prefix string except blank, blnak is reserved for last class
					for k in xrange(num_labels - 1):
						extension_hyp = Hypothesis(0,0,0)
						# Except blank, prob 0
						extension_hyp.p_b = float('-inf')
						extension_hyp.prefix_len = prefix_len + 1
						# pr(y+k, t) <- pr(k, y, t)
						# Use tuple not to be changed
						extended_pfx = tuple(list(pfx_str) + [k])
						extended_pfx_label = Hypothesis.index_to_label(k)
						extension_hyp.p_nb = ctc_p[t-1, k] + self.args.alpha*clm_dict[pfx_str[-1]][extended_pfx_label]
						if pfx_len > 0:
							pfx_end = pfx_str[pfx_len - 1]
							pfx_end_idx = Hypothesis.index_to_label(pfx_end)
							if pfx_end_idx == k:
								extension_hyp.p_nb += hyps.p_b
						else:
							extension_hyp.p_nb += p_total
						beam[extended_pfx] = extenstion_hyp
	
				
			beam_final = sorted(beam.items(), key=prev_prob, reverse=True)
			beam_decoded.append(beam_final)

		return beam_decoded
