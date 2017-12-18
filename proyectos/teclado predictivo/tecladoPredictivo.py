# Librerias externas
import tensorflow as tf
import requests

# Librerias de python
import os
import collections

# Librerias que de momento no uso
import numpy as np
import string
import re
import random

##################################################### Load & clean data

data_file = 'data/shakespeare.txt'

if not os.path.isfile(data_file):
    print('Data file not found, downloading the dataset')
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    # Decode binary into string
    s_text = shakespeare_file.decode('utf-8')
    # Drop first few descriptive paragraphs.
    s_text = s_text[7675:]
    # Remove newlines
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')
    # Write to file
    with open(data_file, 'w') as out_conn:
        out_conn.write(s_text)
else:
    with open(data_file, 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

print('Sample data:\n'+ s_text[:200])

punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()


############################################################### Build Vocabulary

def build_vocab(text, min_word_freq):
	word_counts = collections.Counter(text.split(' '))
	print ('word counts: ', len(word_counts), 'text len: ', len(text.split(' ')))
	# limit word counts to those more frequent than cutoff
	word_counts = {key: val for key, val in word_counts.items() if val > min_word_freq}
	# Create vocab --> index mapping
	words = word_counts.keys()
	vocab_to_ix_dict = {key: (ix + 1) for ix, key in enumerate(words)}
	# Add unknown key --> 0 index
	vocab_to_ix_dict['unknown'] = 0
	# Create index --> vocab mapping
	ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}
	return (ix_to_vocab_dict, vocab_to_ix_dict)

# Build Shakespeare vocabulary
min_word_freq = 5  # Trim the less frequent words off
ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert (len(ix2vocab) == len(vocab2ix))


##################################################### Convert text to word Vectors

s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)



##################################################### LSTM RNN Model


epochs           = 10  # Number of epochs to cycle through data
batch_size       = 32  # Train on this many examples at once
learning_rate    = 0.001  # Learning rate
training_seq_len = 11  # how long of a word group to consider
rnn_size         = 1024  # RNN Model size, has to equal embedding size
embedding_size   = rnn_size
eval_every       = 50  # How often to evaluate the test sentences
prime_texts      = ['thou art more', 'to be or not to', 'wherefore art thou']


lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_size)
initial_state = lstm_cell.zero_state(batch_size, tf.float32)

x_data = tf.placeholder(tf.int32, [batch_size, training_seq_len])
y_output = tf.placeholder(tf.int32, [batch_size, training_seq_len])
