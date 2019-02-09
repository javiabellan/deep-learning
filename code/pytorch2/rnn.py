vocab_size  = 85
embed_size  = 42
hidden_size = 256



class unrolled3_RNN(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super().__init__()
		self.l_emb    = nn.Embedding(vocab_size, embed_size) # input   to input_e
		self.l_in     = nn.Linear(embed_size, hidden_size)   # input_e to hidden
		self.l_hidden = nn.Linear(hidden_size, hidden_size)  # hidden  to hidden (recurrent)
		self.l_out    = nn.Linear(hidden_size, vocab_size)   # hidden  to output
		
	def forward(self, c1, c2, c3):
		# Every input
		in1 = F.relu(self.l_in(self.l_emb(c1)))
		in2 = F.relu(self.l_in(self.l_emb(c2)))
		in3 = F.relu(self.l_in(self.l_emb(c3)))
		
		h = V(torch.zeros(in1.size()).cuda()) # Initial hidden sate: zeros
		h = F.tanh(self.l_hidden(h+in1))      # First input will be sumed by zero
		h = F.tanh(self.l_hidden(h+in2))
		h = F.tanh(self.l_hidden(h+in3))
		
		return F.log_softmax(self.l_out(h))


# Input state & hidden sate: sum (loose information)
class rolled_RNN_sum(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super().__init__()
		self.l_emb    = nn.Embedding(vocab_size, embed_size) # input   to input_e
		self.l_in     = nn.Linear(embed_size, hidden_size)   # input_e to hidden
		self.l_hidden = nn.Linear(hidden_size, hidden_size)  # hidden  to hidden
		self.l_out    = nn.Linear(hidden_size, vocab_size)   # hidden  to output
		
	def forward(self, *inputs):
		bs = inputs[0].size(0)
		h = V(torch.zeros(bs, hidden_size).cuda()) # Initial hidden sate: zeros
		for i in inputs:
			inp = F.relu(self.l_in(self.l_emb(i)))
			h   = F.tanh(self.l_hidden(h+inp))
		
		return F.log_softmax(self.l_out(h), dim=-1)


# Input state & hidden sate: concat
class rolled_RNN(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super().__init__()
		self.l_emb    = nn.Embedding(vocab_size, embed_size)           # input   to input_e
		self.l_in     = nn.Linear(embed_size+hidden_size, hidden_size) # input_e to hidden
		self.l_hidden = nn.Linear(hidden_size, hidden_size)            # hidden  to hidden
		self.l_out    = nn.Linear(hidden_size, vocab_size)             # hidden  to output
		
	def forward(self, *inputs):
		bs = inputs[0].size(0)
		h = V(torch.zeros(bs, hidden_size).cuda()) # Initial hidden sate: zeros
		for i in inputs:
			inp = torch.cat((h, self.l_emb(i)), 1)
			inp = F.relu(self.l_in(inp))
			h = F.tanh(self.l_hidden(inp))
		
		return F.log_softmax(self.l_out(h), dim=-1)



opt = optim.Adam(RNNmodel.parameters, 1e-3)



######################################################### PYTORCH nn.RNN

class rolled_RNN_pytorch(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super().__init__()
		self.l_emb = nn.Embedding(vocab_size, embed_size)
		self.rnn   = nn.RNN(embed_size, hidden_size)
		self.l_out = nn.Linear(hidden_size, vocab_size)
		
	def forward(self, *cs):
		bs = cs[0].size(0)
		h = V(torch.zeros(1, bs, hidden_size)) # 1 means foward RNN
		inp = self.l_emb(torch.stack(cs))
		outp,h = self.rnn(inp, h) # For loop
		
		return F.log_softmax(self.l_out(outp[-1]), dim=-1) # outp[-1] means last output 



# STATEFUL RNN
# Si queremos que en los siguientes minibatches
# se mantengan el hidden state asnterior,
# lo metemos en el constructor.
class rolled_RNN_pytorch(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super().__init__()
		self.l_emb = nn.Embedding(vocab_size, embed_size)
		self.rnn   = nn.RNN(embed_size, hidden_size)
		self.l_out = nn.Linear(hidden_size, vocab_size)
		self.h = V(torch.zeros(1, bs, hidden_size)) # 1 means foward RNN

		
	def forward(self, *cs):
		bs = cs[0].size(0)
		inp = self.l_emb(torch.stack(cs))
		outp, self.h = self.rnn(inp, self.h) # For loop
		
		return F.log_softmax(self.l_out(outp[-1]), dim=-1) # outp[-1] means last output 


# GRADIENT EXPLOSION
# In the for loop, we do a matrix multiplication many times.
# Making the gradient very high or ver low
# To solve tha initialize the RNN weights with the identity matrix 
# This little tweak helps a lot

m.rnn.weight_hh_l0.data.copy_(torch.eye(n_hidden))