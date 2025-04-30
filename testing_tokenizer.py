import torch.nn as nn
import torch
from model import GPTLanguageModel
import argparse
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats,merge,decode_token_basic,encode_text_token_basic,cosine_sim
from rouge import Rouge
import string
import tiktoken
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    parser.add_argument('--vocab_size', type=int, default=65,
                        help='Size of blocks to process vocabulary')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout')


    # model dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='./models/test.pth')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--testing_file_prompt', type=str, default='./test_data/test_prompt_1.txt')
    parser.add_argument('--testing_file_response', type=str, default='./test_data/test_response_1.txt')
    parser.add_argument('--testing_file_answer', type=str, default='./test_data/test_answer_1.txt')
    parser.add_argument('--training_file', type=str, default='./train_data/AI_Foundations4.txt')
    parser.add_argument('--generate_token_number', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='AI_Foundations4', choices=['shakespeare','AI_Foundations4'], help='dataset')

    opt = parser.parse_args()

    return opt
def main():
    opt = parse_option()
    with open(opt.testing_file_prompt, 'r', encoding='utf-8') as f:
        text_test_prompt = f.read()

    with open(opt.testing_file_answer, 'r', encoding='utf-8') as f:
        text_test_answer= f.read()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()



    if (opt.tokenization_strategy == ''):
        # 100 Tokens
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
    elif(opt.tokenization_strategy == 'GPT2'):
        vocab_size = 50257
    elif(opt.tokenization_strategy == 'BPE'):
        tokens = text.encode('utf-8')
        desired_vocab_size = opt.vocab_size

        # Make assumption that your Text Only Has Printable Characters in The English Language
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
        j = 0
        for i in range(num_merges):
            stats = get_stats(ids)
            pair  = max(stats, key=stats.get)
            if(pair == (-1,-1)):
                j = j + 1
                continue
            idx = 256 + 116 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(len(predefined_phrases)):
            vocab[256 + i] = predefined_phrases[i].encode('utf-8')
        for (p0,p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        vocab_size = desired_vocab_size - j
        print(len(vocab))



    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)
    model.load_state_dict(torch.load(opt.ckpt))

    if (opt.tokenization_strategy == ''):
        encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
        context = torch.tensor(encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(text_test_answer)
        response = decode(model.generate(context, max_new_tokens=number_gen,block_size=opt.block_size)[0].tolist())
    elif(opt.tokenization_strategy == 'GPT2'):
        enc = tiktoken.get_encoding('gpt2')
        context = torch.tensor(enc.encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(context)
        response = enc.decode(model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist())
    elif (opt.tokenization_strategy == 'BPE'):
        
        prompt_ids = encode_text_token_basic(text_test_prompt, merges)
        groundT_ids = encode_text_token_basic(text_test_answer, merges)
        context = torch.tensor(prompt_ids, dtype=torch.long, device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(prompt_ids)
        gen_ids = model.generate(context, max_new_tokens=1000, block_size=opt.block_size)[0].tolist()
        response = decode_token_basic(gen_ids, vocab)

        
        

    with open(opt.testing_file_response, "w") as file:
        file.write(response)
    print(levenshtein_distance(text_test_answer, response))
    rouge = Rouge()
    scores = rouge.get_scores(response, text_test_answer)

    print(f"Cosine similarity: {cosine_sim(gen_ids, groundT_ids, model, device = 'cuda:0')}")
    print(scores)



if __name__ == "__main__":

    main()