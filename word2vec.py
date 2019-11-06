import torch
from random import shuffle
from collections import Counter
import argparse
import random
import math
import operator

from huffman import HuffmanCoding

def getRandomContext(corpus, C=4):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def CBOW(center_word_code, contextWords, inputMatrix, outputMatrix, mode):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(N,D))       #
# N : len(non-leaf node)
# K : centerword와 관련된 노드의 갯수 N > K
#########################################################################
    #print("center_word_activated_node_idx_lst : ", center_word_activated_node_idx_lst)
    #print("context_words_activated_node_idx_lst : ", context_words_activated_node_idx_lst)
    #get hidden layer
    sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D
    
    score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
    score_vector = torch.t(score_vector)

    p_center_word = 1.0
    for i, code in enumerate(center_word_code):
        if code == '0':
            p_center_word *= 1/(1+ torch.exp(score_vector[i]))
        elif code == '1':
            #idx = center_word_activated_node_idx_lst[i]
            #print(idx)
            p_center_word *= (1-(1/(1+ torch.exp(score_vector[i]))))
        else:
            print("what is it?", code)
            exit()
    
    loss = -torch.log(p_center_word)
    
    score_grad = score_vector - 1

    grad_out = torch.matmul(score_grad, sum_of_context_words_vector) #(K,1) * (1,D) = (K,D)
    grad_emb = torch.matmul(torch.t(score_grad), outputMatrix) #(1,K) * (K,D) = (1,D)

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))          #
#########################################################################

    return loss, grad_emb, grad_out
def NS():
    pass

#원래 iter 100000
def word2vec_trainer(corpus, nonleaf_ind, word2code, code2idx, mode, word2idx, dimension=100, learning_rate=0.025, iteration=100000):
    # Xavier initialization of weight matrices
    window_size = 4
    losses=[]
    
    #Training word2vec using SGD
    centerword, context = getRandomContext(corpus, window_size)
    
    if mode == "HS":
        
        W_emb = torch.randn(len(word2code), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(nonleaf_ind), dimension) / (dimension**0.5)  
        
        for i in range(iteration): 
            center_word_code = word2code[centerword] 
            context_words_codes = [word2code[i] for i in context]

            '''
            centerword, contextwords의 probabilty를 계산하는데 거치는 노드들의 code와 idx를 저장
            '''
            node_code=''
            center_word_activated_node_code_lst = ['']
            for char in center_word_code:
                node_code += char
                center_word_activated_node_code_lst.append(node_code)
            
            center_word_activated_node_code_lst = center_word_activated_node_code_lst[:-1]
            center_word_activated_node_idx_lst = [list(nonleaf_ind[center_word_activated_node_code])[0] for center_word_activated_node_code in center_word_activated_node_code_lst]
            
            node_code=''
            context_words_activated_node_code_lst = ['']
            for word_code in context_words_codes:
                node_code=''
                for char in word_code:
                    node_code += char
                    if node_code not in context_words_activated_node_code_lst and len(node_code) != len(word_code):
                        context_words_activated_node_code_lst.append(node_code)

            #context_words_activated_node_code_lst = context_words_activated_node_code_lst[:-1]
            Scontext_words_activated_node_idx_lst = [list(nonleaf_ind[context_words_activated_node_code])[0] for context_words_activated_node_code in context_words_activated_node_code_lst]

            #얘네의 idx는 실제 단어의 idx
            centerInd =  code2idx[word_code]

            contextInds = [code2idx[word_code] for word_code in context_words_codes]

            L, G_emb, G_out = CBOW(center_word_code, contextInds, W_emb, W_out[center_word_activated_node_idx_lst], mode)
            W_emb[contextInds] -= learning_rate*G_emb
            W_out[center_word_activated_node_idx_lst] -= learning_rate*G_out
            losses.append(L)
            
            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,))
                losses=[]
            
    elif mode == "NS":
        W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
        
        for i in range(iteration):
            #Training word2vec using SGD

            centerInd =  word2ind[centerword]
            contextInds = [word2ind[i] for i in context]
            
            L, G_emb, G_out = CBOW(center_word_code, contextInds, W_emb, W_out[center_word_activated_node_idx_lst], mode)
            W_emb[contextInds] -= learning_rate*G_emb
            W_out[center_word_activated_node_idx_lst] -= learning_rate*G_out
            losses.append(L)
        
        
        
        pass
    
    return W_emb, W_out


def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()
    
def get_size(vector):
    size = len(vector)
    sum = 0.0
    for v in vector:
        sum += math.pow(v,2)
    result = math.sqrt(sum)
    return result
    
#get vector innerproduct
def get_innerproduct(v1, v2):
    size = len(v1)
    result = 0.0
    for a,b in zip(v1,v2):
        result += a*b
    return result
    
def cosine_similarity(v1, v2, v1_size = None):
    if v1_size is not None:
        return get_innerproduct(v1,v2) / (v1_size * get_size(v2))
    else:
        return get_innerproduct(v1,v2) / (get_size(v1) * get_size(v2))
    

def find_sim_word(W_emb, w1, w2, w3, w2c, c2i):
    idx_w1 = c2i[w2c[w1]]
    idx_w2 = c2i[w2c[w2]]
    idx_w3 = c2i[w2c[w3]]
    
    vec_w1 = W_emb[idx_w1]
    vec_w2 = W_emb[idx_w2]
    vec_w3 = W_emb[idx_w3]
    #print(vec_w2.shape)
    vec_predicted = vec_w2.sub_(vec_w1).add_(vec_w3)
    #print(vec_predicted)
    vec_predicted_size = get_size(list(vec_predicted))
    sim_dict ={}
    for i in range(len(W_emb)):
        #여기서 이미 앞에 나오는 제외하도록 코드를 짜거나, 상위 몇 개 출력할 수 있도록 바꿔야함.
        if i == idx_w1 or i == idx_w2 or i == idx_w3:
            pass
        else:
            vec_i = W_emb[i]
            sim = cosine_similarity(list(vec_predicted), list(vec_i), v1_size = vec_predicted_size)
            sim_dict[i]=sim
            
#     sim_dict = sorted(sim_dict.items(), key=operator.itemgetter(1))
# #             if sim > best_sim:
# #                 best_sim = sim
# #                 best_idx = i
    
    return sim_dict
    
    

    

def main():
    
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='HS for Hierarchical Softmax, NS for Negative Sampling')
#     parser.add_argument('part', metavar='partition', type=str,
#                         help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    #part = args.part
    
    print("loading...")
    
    text = open('questions-words.txt',mode='r').readlines() #각 줄을 읽어서 리스트로 반환
    text =  text[1:]

    corpus=[]
    for t in text:
        t = t.lower()
        corpus += t.split()
    #print(corpus)
    print("tokenizing...")
    frequency = Counter(corpus) #dict
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>4:
            processed.append(word)
            
    frequency = Counter(processed)
    vocabulary = set(processed)
    
    if mode == "HS":
    
        '''
        Huffman Coding
        '''
        print("Start Huffman Coding")

        HC = HuffmanCoding()
        _,_ = HC.build(frequency) 


        print("Finish Huffman Coding")



        code2idx = {}
        idx2word = {}
        i=0
        for word,code in HC.codes.items():
            code2idx[code] = i
            idx2word[i] = word
            i += 1


        #HC.codes , word -> code
        #HC.reverse_mapping , code -> word
        #HC.nonleaf_ind , 
        #word -> code -> idx


        emb,_ = word2vec_trainer(processed, HC.nonleaf_ind, HC.codes, code2idx, mode, dimension=64, learning_rate=0.05, iteration=50000)
        
        sim_dict = find_sim_word(emb, "germany", "berlin", "france", HC.codes, code2idx) #마지막에 들어갈 단어는?
        sorted_sim_dict = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)
        #print(sorted_sim_dict)
        stop = 5
        for lst in sorted_sim_dict:
            idx = list(lst)[0]
            sim = list(lst)[1]
            print(idx2word[idx], sim)
            stop -= 1
            if stop == 0 :
                break
        
    elif mode == "NS":
        
        sum_of_freq = 0
        for word, freq in frequency.items():
            sum_of_freq += freq
        
        print(sum_of_freq) 
        
        ratio_of_freq = {}
        for word, freq in frequency.items():
            ratio_of_freq[word] = freq / sum_of_freq
        
        print(ratio_of_freq)
        
        prob_of_ns
        #print(frequency)

main()
