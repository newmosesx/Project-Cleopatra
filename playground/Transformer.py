
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import re
import os
import csv
import unicodedata
import itertools
import pandas as pd

# Used to change between gpu or cpu. We use this to choose where to process and store the AI temporarily.
# It can also move other variables between devices. You will need a powerful gpus (Nvidia) if you want to use cuda. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

save_dir = os.path.join("/home/newlife/Folders/Programs/A_Project/playground/data")

PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 40

class Library:
    def __init__(self):
        self.name = "Dataset"
        self.trimmed = False
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

# We are changing the Standard from unicode (Global Standard) to ASCII which is more in line with our occidental vocabulary (American Standard).
#Thus avoiding letters like "你好，안녕하세요，こんにちは" but allowing letters from A-Z.
def unicodeToAscii(string):
    return ''.join(
        c for c in unicodedata.normalize('NFD', string)
        if unicodedata.category(c) != 'Mn'
    )

# Here we process sentences and make them easier to comprehend by removing certain symbols and characters
def normalizeString(sentence):
    AsciiSentence = unicodeToAscii(sentence.lower().strip())
    AsciiSentence = re.sub(r"([.!?])", r" ", AsciiSentence)
    AsciiSentence = re.sub(r"[^a-zA-Z.!?]+", r" ", AsciiSentence)
    AsciiSentence = re.sub(r"[^\x00-\x7F]", r"", AsciiSentence) #Unicode to Ascii function
    AsciiSentence = re.sub(r"\s+", r" ", AsciiSentence).strip()
    return AsciiSentence

# Here we remove pairs not containing anything.
def removePair(pairs):
    try:
        filtered_pairs = [[left, right] for left, right in pairs if left and right]
    except:
        filtered_pairs = [[pair[0], pair[1]] for pair in pairs if pair[0] and pair[1]]
    return filtered_pairs

# Here we load and distribute a csv dataset into pairs of 2 sentences
def readCsv(datafile):
    pairs = []
    questions = []
    responses = []
    with open(datafile, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 2:
                questions.append(row[1])
                responses.append(row[2])
    for question, response in zip(questions, responses):
        question = normalizeString(question)
        response = normalizeString(response)
        pair = [question, response]
        pairs.append(pair)
    filtered_pairs = removePair(pairs)
    return filtered_pairs

# Here we load and distribute a parquet dataset into pairs of 2 sentences
def readPanda(datafile):
    pairs = []
    dataset = pd.read_parquet(datafile)
    questions = dataset['question'].tolist()
    responses = dataset['response'].tolist()
    for question, response in zip(questions, responses):
        question = normalizeString(question)
        response = normalizeString(response)
        pair = [question, response]
        pairs.append(pair)
    filtered_pairs = removePair(pairs)
    return filtered_pairs

# Here we load and distribute a txt dataset into pairs of 2 sentences
def readTxt(datafile):
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    filtered_pairs = removePair(pairs)
    return  filtered_pairs

# Here we load and distribute a custom dataset (txt) into pairs of 2 sentences
def customData(datafile):
    pairs = []
    pair = []
    data = open(datafile, encoding='utf-8').\
        read().strip().split("\n")
    for line in data:
        split_strings = line.split("\\t")
        source = normalizeString(split_strings[0])
        target = normalizeString(split_strings[1])
        pair.append(source)
        pair.append(target)
        pairs.append(pair)
        pair = []
    filtrated_pairs = removePair(pairs)
    return filtrated_pairs

# Here we filter each pair to check that they are within a certain length.
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH - 1 and len(p[1].split(' ')) < MAX_LENGTH - 1

# Here we send each pair in pairs to the filterPair function.
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Here we get the pairs and add the words of each pair of sentences in the library.
def loadPrepareData(libra, option):
    if option == 1:
        pairs = readPanda("/content/drive/MyDrive/movie-corpus/movie-corpus/0000.parquet")
    elif option == 2:
        pairs = readCsv("/content/drive/MyDrive/movie-corpus/movie-corpus/Conversation.csv")
    elif option == 3:
        pairs = readTxt("/content/drive/MyDrive/movie-corpus/movie-corpus/formatted_movie_lines.txt")
    elif option == 4:
        pairs = customData("/mnt/chromeos/GoogleDrive/MyDrive/movie-corpus/movie-corpus/customdata.txt")
    else:
        fullPairs = []
        pairs = []
        pairs1 = readTxt("/content/drive/MyDrive/movie-corpus/movie-corpus/formatted_movie_lines.txt")
        fullPairs.append(pairs1)

        pairs2 = readCsv("/content/drive/MyDrive/movie-corpus/movie-corpus/Conversation.csv")
        fullPairs.append(pairs2)

        pairs3 = readPanda("/content/drive/MyDrive/movie-corpus/movie-corpus/0000.parquet")
        fullPairs.append(pairs3)

        pairs4 = customData("/content/drive/MyDrive/movie-corpus/movie-corpus/customdata.txt")
        fullPairs.append(pairs4)

        for p in fullPairs:
            for pair in p:
                pairs.append(pair)

    pairs = filterPairs(pairs)
    for pair in pairs:
        libra.addSentence(pair[0])
        libra.addSentence(pair[1])
    return pairs

# This function will convert words in a sentence to their number representation. While, also adding the start and end tokens.
def SentenceToNum(libra, sentence):
    return [SOS_token] + [libra.word2index[word] for word in sentence.split(' ') if word in libra.word2index] + [EOS_token]

# The paddig function will add pad tokens in the tokenized sentence to make them all equal in length.
def Padding(batch):
    padded_list = []
    for sequence in batch:
        padded_sequence = list(sequence) + [PAD_token] * ((MAX_LENGTH) - len(sequence))
        padded_list.append(padded_sequence)
    return padded_list

# The function will add 1 if there is a a token other than the pad token and if the token is the pad token add a 0.
def BinaryMask(batch):
    m = []
    for i, seq in enumerate(batch):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# The function produces the source tensor we will use for our AI to learn and the lengths tensor to identify the actual lengths of each sequence in the batch.
def inputVar(batch, libra):
    indexes_batch = [SentenceToNum(libra, sentence) for sentence in batch] # batch of tokenized sentences
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = Padding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# This function produces the target tensor, the mask tensor and the maximum length of the target
def outputVar(batch, libra):
    indexes_batch = [SentenceToNum(libra, sentence) for sentence in batch]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = Padding(indexes_batch)
    mask = BinaryMask(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# This function returns all the outputs from the inputVar and outputVar
def batch2TrainData(libra, pair_batch):
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, libra)
    output, mask, max_target_len = outputVar(output_batch, libra)
    return inp, output, lengths, mask, max_target_len

# The normalization class helps bring numbers between the range of 0 and 1
class Normalization(nn.Module):
    def __init__(self, scale: float, shift: float, epsilon: float = 1e-8):
        super(Normalization, self).__init__()
        self.scale = scale
        self.shift = shift
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x)
        deviation = torch.std(x) + self.epsilon
        x = (x - mean) / deviation
        x = x * self.scale
        x = x + self.shift
        return x

# This class is the neural network structure which makes the predictions and computes the gradients.
class DecoderNeurons(nn.Module):
    def __init__(self, embedding_size: int, vocab_size: int):
        super(DecoderNeurons, self).__init__()
        self.fc1 = nn.Linear(embedding_size * 2, embedding_size)
        self.fc2 = nn.Linear(embedding_size, vocab_size)
        self.tnh = nn.Tanh()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.tnh(self.fc1(x))
        output = self.fc2(x)
        return output

# This class is also a neural network strcuture which computes the context for the understanding of the decoder.
class EncoderNeurons(nn.Module):
    def __init__(self, embedding_size: int):
        super(EncoderNeurons, self).__init__()
        self.fc1 = nn.Linear(embedding_size * 2, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.sig(self.fc1(x))
        output = self.fc2(x)
        return output

# The attention mechanism computes the weights and importance of each token in a sentence.
class Attention(nn.Module):
    def __init__(self, embedding_size: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(embedding_size, embedding_size)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = torch.sum(hidden * energy, dim=2)
        return energy

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.general_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        attn_energies = F.softmax(attn_energies, dim=1).unsqueeze(1)
        return attn_energies

# The encoder layer is used here to make sure there are no ovefitting issues with the AI's learning.
# And to return the results of the artifical neural network.
class EncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.network = EncoderNeurons(embedding_size)
        self.norm = Normalization(0.4, 0.4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rnn_output):
        input_dropped = self.dropout(rnn_output)
        input_normalized = self.norm(input_dropped)
        output = self.network(input_normalized)
        return output

# The encoder is used to gather information about the source tensor before passing it to the encoder neurons.
# To then return the encoder neurons output.
class Encoder(nn.Module):
    def __init__(self, embedding, embedding_size: int, dropout: float, n_layers: int):
        super(Encoder, self).__init__()
        self.num_layers = n_layers
        self.encoder_layer = EncoderLayer(embedding_size, dropout)
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layers,
                             dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_tensor, lengths):
        source_embedding = self.embedding(source_tensor)
        source_embedding = self.dropout(source_embedding)
        packed = nn.utils.rnn.pack_padded_sequence(source_embedding, lengths.to('cpu'),enforce_sorted=False)
        rnn_output, hidden = self.lstm(packed)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)
        output = self.encoder_layer(rnn_output)
        return output, hidden

# The decoder layer is used here to make sure there are no ovefitting issues with the AI's learning.
# And to return the results of the artifical neural network.
class DecoderLayer(nn.Module):
    def __init__(self, embedding_size: int, vocab_size: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.norm = Normalization(0.4, 0.4)
        self.dropout = nn.Dropout(dropout)
        self.network = DecoderNeurons(embedding_size, vocab_size)

    def forward(self, concat_input):
        input_dropped = self.dropout(concat_input)
        input_normalized = self.norm(input_dropped)
        output = self.network(input_normalized)
        return output

# The decoder is used to gather information about the encoder's output and the decoder input before passing it to the decoder neurons.
# To then return the decoder neurons output probabilities.
class Decoder(nn.Module):
    def __init__(self, embedding, embedding_size: int, dropout: float, n_layers: int, vocab_size: int):
        super(Decoder, self).__init__()

        self.num_layers = n_layers

        self.attention = Attention(embedding_size)
        self.decoder_layer = DecoderLayer(embedding_size, vocab_size, dropout)

        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.embedding = embedding

    def forward(self, decoder_input, encoder_output, hidden_inf):
        inputEmbedding = self.embedding(decoder_input)
        inputEmbedding = self.dropout(inputEmbedding)
        rnn_output, hidden = self.lstm(inputEmbedding, hidden_inf)
        attn_weights = self.attention(rnn_output, encoder_output)
        context = attn_weights.bmm(encoder_output.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        network_output = self.decoder_layer(concat_input)
        output = F.softmax(network_output, dim=1)
        return output, hidden

# The maskNLLLoss function is used to computes and returns the loss and the number of elements that counts for the loss operation.
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# train functions starts the forward and backward passes and returns the loss of the AI
def train(input_variable, target_variable, decoder, encoder, clip, libra,
          encoder_optimizer, decoder_optimizer, batch_size, lengths, mask, max_length):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    print_losses = []
    n_totals = 0


    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(device)

    encoder_output, hidden = encoder(input_variable.t(), lengths)
    (hidden_state, cell_state) = hidden
    decoder_hidden = (hidden_state[:decoder.num_layers], cell_state[:decoder.num_layers])

    choice = random.random()


    if choice > 0.5:
      use_teacher_forcing = True
    else:
      use_teacher_forcing = False

    target_variable = target_variable.t()
    mask = mask.t()

    if use_teacher_forcing:
        #all_tokens = torch.zeros([0], device=device_cpu, dtype=torch.long).to(device)
        for t in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, encoder_output, decoder_hidden
            )

            #decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            #all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

            decoder_input = target_variable[t].view(1, -1)
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
        #print(1,"\n")

    elif use_teacher_forcing == False:
        #all_tokens = torch.zeros([0], device=device_cpu, dtype=torch.long).to(device)
        for t in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, encoder_output, decoder_hidden
            )

            #decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            #all_tokens = torch.cat((all_tokens, decoder_input), dim=0)


            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(device)
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
        #print(2,"\n")
    #decoded_words = [libra.index2word[token.item()] for token in all_tokens]
    #target_words = [libra.index2word[token.item()] for token in target_variable]
    #decoded_words[:] = [x for x in decoded_words if not (x == 'EOS' or x == 'SOS')]
    #target_words[:] = [x for x in target_words if not (x == 'EOS' or x == 'SOS')]
    #print('Cleopatra:', ' '.join(decoded_words))
    #print('Target:', ' '.join(target_words))

    loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# The trainIters function saves the AI model, gets the source, target, mask and lengths for training
# and prinstous the report (the loss and the current iteration)
def trainIters(model_name, libra, save_dir, n_iteration, batch_size, checkpoint, clip,
               print_every, save_every, loadFilename, vocab_size, decoder, encoder,
               decoder_optimizer, encoder_optimizer, embedding, pairs):

    training_pairs = [batch2TrainData(libra, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    start_iteration = 1
    print_loss = 0
    tries = 0

    if loadFilename:
        tries = checkpoint['time']


    print("Initializing Training...")
    print()
    for iteration in range(start_iteration, n_iteration + 1):
        training_pair = training_pairs[iteration - 1]

        input_variable, target_variable, lengths, mask, max_target_len = training_pair
        # batch / length

        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        sum_loss = train(input_variable, target_variable, decoder, encoder, clip, libra,
        encoder_optimizer, decoder_optimizer, batch_size, lengths, mask, max_target_len)
        print_loss += sum_loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0


        if (iteration % save_every == 0):
                    tries += save_every
                    directory = os.path.join(save_dir, model_name)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save({
                        'iteration': iteration,
                        'time': tries,
                        'en': encoder.state_dict(),
                        'de': decoder.state_dict(),
                        'en_opt': encoder_optimizer.state_dict(),
                        'de_opt': decoder_optimizer.state_dict(),
                        'loss': sum_loss,
                        'voc_dict': libra.__dict__,
                        'embedding': embedding.state_dict()
                    }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# The beamsearch class allows for the AI to seem to perform better through searching the best result while avoiding
# the repetition of words and allowing some randomnes in the choices instead of always having the highes most probable
# predictions.
class BeamSearch(nn.Module):
    def __init__(self, beam_width, encoder, decoder, libra, temp= 1.2, penalty = 0.6):
        super(BeamSearch, self).__init__()
        self.beam_width = beam_width
        self.libra = libra
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temp
        self.rep_penalty = penalty

    # This function checks for repetitive words and applies a penalty in its probability to avoid them being used too much. 
    def repetition(self, sequence):
        token_counts = {}
        penalty = 0
        for token in sequence:
            if token in token_counts:
                penalty += token_counts[token]
                token_counts[token] += 1
            else:
                token_counts[token] = 1
        return penalty * self.rep_penalty


    def forward(self, input_sentence):
        indexedSequence = [SentenceToNum(self.libra, input_sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexedSequence])

        paddedSequence = Padding(indexedSequence)
        sentence_tensor = torch.LongTensor(paddedSequence).t()

        decoder_input = torch.LongTensor([[SOS_token]]).t()

        encoder_output, hidden = self.encoder(sentence_tensor.to(device_cpu), lengths.to(device_cpu))
        (hidden_state, cell_state) = hidden
        decoder_hidden = (hidden_state[:self.decoder.num_layers], cell_state[:self.decoder.num_layers])

        beam = [([SOS_token], 0)] # brackets allowing us to unpack both variables. Else python will consider it as 1 variable.
        for _ in range(MAX_LENGTH):
            candidates = []
            for sequence, score in beam:
                last_token = sequence[-1]

                if last_token == EOS_token:
                   candidates.append((sequence, score))
                   continue

                decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_output, decoder_hidden)

                probabilities = decoder_output.squeeze(0) / self.temperature

                topk_probs, topk_indices = torch.topk(probabilities, self.beam_width)

                for probability, index in zip(topk_probs.tolist(), topk_indices.tolist()):
                    penalty_score = self.repetition(sequence + [index])
                    candidates.append((sequence + [index], score + probability - penalty_score))

            beam = sorted(candidates, key= lambda x: x[1], reverse=True)[:self.beam_width]
        predicted_sentence = [self.libra.index2word[index] for index in beam[0][0] if index < self.libra.num_words]
        return predicted_sentence


# The evaluateInput function simply outputs the AI's response and inputs the User's question.
def evaluateInput(searcher):
    while(1):
        try:
            input_sentence = input('User > ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = normalizeString(input_sentence)
            output_words = searcher(input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'SOS')]
            print('Cleopatra:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")

# The library instance
libra = Library()

# The current task the AI will perform along with its name, if to start a pre trained model or no and the path of the pre trained model.
task = "test"
model_name = 'Cleopatra_model#v1.3.1'
checkpoint=None
start_model = "yes"
loadFilename = None if start_model == "no" else "/home/newlife/Folders/Programs/A_Project/playground/data/Cleopatra_model#v1.3.1/2000_checkpoint.tar"

# The clipping limit of the graident, the number of iterations allowed, 
# after how many training sessions we print a report and after how many sessions the model should be saved.
clip = 5.0
n_iteration = 2000
print_every = 1
save_every = 2000


# If there is a model we can load it if there is a path in loadfilename
if loadFilename:
    print("Set to: 'trained model'")
    checkpoint = torch.load(loadFilename, map_location=device)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    libra.__dict__ = checkpoint['voc_dict']
    embedding_sd = checkpoint['embedding']
    print("Loss: ",checkpoint["loss"])
    print("Time: ",checkpoint["time"])
else:
    print("Set to: 'new model'")

if task == "train":
    # if panda, change to 1. if csv, change to 2.  if txt, change to 3. if custom, change to 4,if all, change to any but 1234.
    pairs = loadPrepareData(libra, 4)

# The number of layers for an lstm operation, the embedding size for each word token defined
encoder_n_layers = 2
decoder_n_layers = 1
embedding_size = 50

# The dropout rate, the number of sentences allowed for a batch, 
# the learning rate at which the optimizer adjusts the neurons and the total number of words
dropout = 0
batch_size = 1
learning_rate = 0.00001
vocab_size = libra.num_words

# We now initialise the instances of the embedding, encoder and decoder
embedding = nn.Embedding(vocab_size, embedding_size)
decoder = Decoder(embedding, embedding_size, dropout, decoder_n_layers, vocab_size)
encoder = Encoder(embedding, embedding_size, dropout, encoder_n_layers)


if loadFilename:
    embedding.load_state_dict(embedding_sd)

if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

if task == "train":
    encoder.train()
    decoder.train()

    embedding = embedding.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
else:
    encoder.eval()
    decoder.eval()

    embedding = embedding.to(device_cpu)
    encoder = encoder.to(device_cpu)
    decoder = decoder.to(device_cpu)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-8)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-8)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

if task == "train":
    trainIters(model_name, libra, save_dir, n_iteration, batch_size, checkpoint, clip,
               print_every, save_every, loadFilename, vocab_size, decoder, encoder,
               decoder_optimizer, encoder_optimizer, embedding, pairs)

if task == "test":
    beam_width = 30
    searcher = BeamSearch(beam_width, encoder, decoder, libra)
    evaluateInput(searcher)