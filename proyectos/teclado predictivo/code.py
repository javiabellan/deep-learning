import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time


logs_path = "log"
data_path = "data"


start_time = time.time()

def tiempo(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
writer = tf.summary.FileWriter(logs_path)

##################################################### Load & clean data

# Text file containing words for training
training_file = data_path + "/belling_the_cat.txt"

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content # Devuelve un array de palabras

training_data = read_data(training_file)
#  ['long' 'ago' ',' 'the' 'mice' ...
# print("Loaded training data:\n", training_data, "\n\n")



######################################################### Construir diccionario

# La red neuronal entiende números, no palabras.
# Por lo tanto hay que convertir las palabras de alguna forma.

# Aquí estamos utilizando la técnica del diccionario,
# donde a cada palabra le asignamos un índice.
# Lo hacemos así, porque es fácil y tenemos pocas palabras
# (se necesita una salida distinta para cada palabra)

# Otra opción, probalemente mucho mejor, es usar word emmbedings
# De esta forma las palabras se codifican a un vector de tamaño fijo
# que aporta información sobre el significado de la palabra.

def build_dataset(words):
	# Array de parejas que cuanta cuantas veces aparece cada palabra. Ejemplo: ('the', 11)
	# Además están oredeanads descententemete (primero las más frecuentes)
	# [ (',', 14), ('the', 11), ('.', 8), ('and', 7), ...
    count = collections.Counter(words).most_common()

    # El dictionary, aprovechara el orden antarior, para asignar índices,
    # el 0 para la palabra mas frecuente, y asi.
    #     dictionary:         { ',':0,  'the':1,  '.':2,  'and':3, ...
    #     reverse_dictionary: { 0:',',  1:'the',  2:'.',  3:'and', ... 
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# El dictionary         se usará para codificar las palabras de entrada de la red
# El reverse_dictionary se usará para descodificar las palabras de salida de la red


######################################################### Construir modelo

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer

# The cost is a cross entropy between label and softmax() prediction
# optimized using RMSProp at a learning rate of 0.001.
# RMSProp performs generally better than Adam and SGD for this case.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



######################################################### Entrenar modelo

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        # ENTRADA: Coger las siguientes n_input palabras y formar un vector de indices de diccionario
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        # ETIQUETA: One hot encodding la siguinte palabra
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Tiempo transcurrido: ", tiempo(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")



######################################################### Probar modelo

	# Ahora viene la parte divertida, escribe 3 palabras y genera la palabra siguinte
	# Luego tus 2 ultima palabras + la predicción, predicirán otra predicción
	# Y así generarás una historia! (de 32 predicciones de palabra)
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
