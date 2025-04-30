import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import argparse
import string
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats,merge,decode_token_basic,encode_text_token_basic
import tiktoken
import csv
import os

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--target_vocab_size', type=int, default=400,
                        help='target vocab size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    
    parser.add_argument('--max_iters', type=int, default=256,
                        help='Max Iterations of the Training Process')
    parser.add_argument('--eval_iters', type=int, default=30,
                        help='Eval Iterations of the Training Process')

    parser.add_argument('--eval_interval', type=int, default=256,
                        help='Max Iterations of the Training Process')
    
    #storing data 
    parser.add_argument('--destiny_file', type=str, default='./Data_Params/default.csv')

    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='')
    parser.add_argument('--save_file', type=str, default='./models/test.pth')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--training_file', type=str, default='./train_data/AI_Foundations4.txt')
    parser.add_argument('--training_file_tokenizer', type=str, default='./train_data/AI_Foundations4.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='AI_Foundations4',choices=['AI_Foundations4'], help='dataset')



    opt = parser.parse_args()


    return opt


def main():
    opt = parse_option()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(opt.training_file_tokenizer, 'r', encoding='utf-8') as f:
        text_token = f.read()

    ##################### Note that the Tokenizer is Trained Separately from the LLM ########################
    if(opt.tokenization_strategy == ''):
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
        data = torch.tensor(encode(text), dtype=torch.long)
    elif(opt.tokenization_strategy == 'BPE'):
        ######################################### We want to merge pairs of encodings together to summarize alphabet ##########
        # Start out with Unicode Encoding

        tokens = text_token.encode('utf-8')
        desired_vocab_size = opt.target_vocab_size

        # Make assumption that your Text Only Has Printable Characters in The English Language + PreProcessed Tokens
        num_merges = desired_vocab_size - 256 - 116
        ids = list(tokens)
        
        #Add predefined tokens to assess priority
        
        predefined_phrases = ["for example", "in addition", "as a result", "the", "in conclusion", 
        "first of all", "moreover", "nevertheless", "be", "of", "and", "that", 
        "have", "for", "it", "not", "on", "in", "with", "as", "does", "nevertheless", 
        "however", "because of", "as well as", "in the same way", "similarly", 
        "unlike", "due to", "even though", "like", "while", "based on", 
        "according to", "in terms of", "with respect to", "in order to", 
        "be able to", "as such", "with regard to", "as long as", "so that", "in case",
        "CNN", "convolution", "convolutional neural network", "machine learning", "ML", 
        "gradient descent", "stochastic gradient descent", "convolutional layer", 
        "fully connected layer", "FC", "support vector machines", "batch gradient descent", 
        "mini-batch gradient descent", "PCA", "regularization", "normalization", "explainability", 
        "interpretability", "maxpool", "pooling", "decision tree", "distance metrics", "LLM", 
        "self-attention layer", "encoder-decoder", "transformer", "embedding space", "token", 
        "principal component", "eigenvector", "multi-head attention", "kernel", "filter", 
        "sequential decision making", "reinforcement learning", "backpropagation", "artificial neural network", 
        "feed-forward", "activation function", "ReLU", "softmax", "information gain", "cross-entropy", 
        "loss function", "learning rate", "dropout rate", "query matrix", "key matrix", "query space", 
        "value matrix", "embedding vector", "weight initialization", "large language model", "cosine similarity", 
        "vector field", "multidimensional", "clustering", "k-means", "kernel tricks", "hyperplane", 
        "linear representation", "gaussian distribution", "covariance matrix", "soft clustering", 
        "partial derivative", "fine-tuning", "overfitting", "bias", "variance", "dimensionality reduction", 
        "cross-validation", "supervised learning", "feature selection", "tradeoff"]

        merges = {}
        j = 0 #Counter of iterations that don't add a token
        for i in range(num_merges):
            stats = get_stats(ids)
            pair  = max(stats, key=stats.get)
            if(pair == (-1,-1)): #Happens once there is no pairs with neccesary frequence
                j = j + 1 #Counter for iterations without adding words to vocabulary
                continue
            idx = 256 + 116 + i #New index calculation for tokens
            ids = merge(ids, pair, idx)
            merges[pair] = idx

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(len(predefined_phrases)): #Add pre-defined phrases to vocabulary
            vocab[256 + i] = predefined_phrases[i].encode('utf-8')
        for (p0,p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        vocab_size = desired_vocab_size - j #Update vocab size - the iterations without adds
        print(len(vocab))
        data = torch.tensor(encode_text_token_basic(text_token,merges), dtype=torch.long)
    elif(opt.tokenization_strategy == 'GPT2'):
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = 50257
        data = torch.tensor(enc.encode(text), dtype=torch.long)

   
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    if(opt.ckpt != ''):
        model.load_state_dict(torch.load(opt.ckpt))
    model = model.to(opt.device)


    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)

    #File for storing training data:
    
    file_path = opt.destiny_file

    file_exists = os.path.exists(file_path)
    
    for iter in range(opt.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % opt.eval_interval == 0:
            losses = estimate_loss(opt,model,train_data,val_data)
            
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            with open(opt.destiny_file, mode='a', newline='') as file:
                
                writer = csv.writer(file, delimiter=',')
                
                if not file_exists:
                    writer.writerow(["Learning rate", "Batch Size", "Block Size", "Dropout", "Train loss", "Val loss"])
                    file_exists = True
                    
                writer.writerow([opt.learning_rate, opt.batch_size, opt.block_size, opt.dropout, f"{losses['val']:.4f}",f"{losses['train']:.4f}"])

        # sample a batch of data
        xb, yb = get_batch('train',train_data,val_data,opt)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_file = opt.save_file
    torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    #tokenizer_test()
    main()