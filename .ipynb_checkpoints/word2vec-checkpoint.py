import torch
from random import shuffle
from collections import Counter
import argparse
import random
import math
import operator
import numpy as np

import time

from huffman import HuffmanCoding

def getRandomContext(corpus, C=4):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]
    #print(context)
    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def get_prob_word_code(word_code, score_vector):
    p_word = 1.0
    zero_lst=[]
    for i, code in enumerate(word_code):
        if code == '0':
            #p_word *= 1/(1+ torch.exp(score_vector[i]))
            p_word *= sigmoid(score_vector[i])
            zero_lst.append(i)
        elif code == '1':
            #p_word *= (1-(1/(1+ torch.exp(score_vector[i]))))
            p_word *= (1 - sigmoid(score_vector[i]))
        else:
            print("what is it?", code)
    return p_word, zero_lst

def get_activated_node(len_corpus, sampling_num, prob_table, correct_idx):
    activated_node_lst = [correct_idx]
    lotto_num = random.randint(0, len_corpus - 1)
    for i in range(sampling_num):
        while correct_idx == lotto_num :    
            lotto_num = random.randint(0, len_corpus - 1)
        activated_node_lst.append(int(prob_table[lotto_num]))
        lotto_num = random.randint(0, len_corpus - 1)
    #print(activated_node_lst)
    return activated_node_lst

def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix, update_system, feed_dict=None):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
# N : len(non-leaf node)
#########################################################################
    
    if update_system == "HS":
        
        context_word_code = feed_dict['context_word_code']
        
        #get hidden layer
        center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D
        #print(center_word_vector.size())
        #score
        score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)
        
        p_context_word,zero_lst = get_prob_word_code(context_word_code, score_vector)

        loss = -torch.log(p_context_word)
        
        #수정해야함
        score_grad = score_vector
        score_grad[zero_lst] -= 1
        
        grad_out = torch.matmul(score_grad, center_word_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(score_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
        
    elif update_system == "NS":

        activated_node_lst =  feed_dict['activated_node_lst']
        
        center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D
        
        score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)
        
        loss = 0.0
        
        for i, idx in enumerate(activated_node_lst):
            if idx == contextWord:
                context_idx = i
                loss -= np.log(sigmoid(score_vector[i]))
            else:
                loss -= np.log((1 - sigmoid(score_vector[i])))

        #get grad
        score_grad = score_vector #(K,1)
        score_grad[context_idx] -= 1

        grad_out = torch.matmul(score_grad, center_word_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(score_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
        
    elif update_system == "BS":
        #get hidden layer
        center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D

        #score
        score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)

        e = torch.exp(score_vector) 
        softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V

        loss = -torch.log(softmax[:,contextWord])

        #get grad
        softmax_grad = softmax
        softmax_grad[:,contextWord] -= 1.0

        grad_out = torch.matmul(torch.t(softmax_grad), center_word_vector) #(V,1) * (1,D) = (V,D)
        grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
        
    else:
        print("What is it?")
        exit()
    
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix, update_system, feed_dict=None):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(N,D))       #
# N : len(non-leaf node)
# K : centerword와 관련된 노드의 갯수 N > K
#########################################################################
    
    if update_system == "HS":
        
        center_word_code = feed_dict['center_word_code']
        
        sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D

        score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)

        p_center_word, zero_lst = get_prob_word_code(center_word_code, score_vector)
        
        loss = -torch.log(p_center_word)

        score_grad = score_vector
        score_grad[zero_lst] -= 1

        grad_out = torch.matmul(score_grad, sum_of_context_words_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(score_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
        
    elif update_system == "NS":
        activated_node_lst =  feed_dict['activated_node_lst']

        sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D
        
        score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)
        
        loss = 0.0
        for i, idx in enumerate(activated_node_lst):
            if idx == centerWord:
                center_idx = i
                loss -= np.log(sigmoid(score_vector[i]))
            else:
                loss -= np.log((1 - sigmoid(score_vector[i])))

        #get grad
        score_grad = score_vector #(K,1)
        score_grad[center_idx] -= 1

        grad_out = torch.matmul(score_grad, sum_of_context_words_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(score_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
    
    elif update_system == "BS":

        sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D

        score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)

        e = torch.exp(score_vector) 
        softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V
        #print(softmax.size())
        #print(softmax)

        loss = -torch.log(softmax[:,centerWord])

        #get grad
        softmax_grad = softmax
        softmax_grad[:,centerWord] -= 1.0

        grad_out = torch.matmul(torch.t(softmax_grad), sum_of_context_words_vector) #(1,V) * (1,D) = (V,D)
        grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
    
    else:
        print("What is it?")
        exit()

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))          #
#########################################################################

    return loss, grad_emb, grad_out


#원래 iter 100000
def word2vec_trainer(corpus, word2idx, mode, update_system, sub_sampling, feed_dict=None, dimension=300, learning_rate=0.01, iteration=50000):
    #print(len(corpus))
    #print(corpus)
    feed_dict2 = {}
    #print(corpus)
    
    #Only once
    if sub_sampling is True:
        print("Start SubSampling...")
        prob_of_ss = feed_dict['prob_of_ss']
        destiny = random.random()
        for word in corpus:
            #사라질운명
            if destiny < prob_of_ss[word]:
                corpus.remove(word) #맨 앞에 하나만 사라짐.
            else:
                pass
        print("Finish SubSampling...")
        
    
    if update_system == "HS":
        
        word2code = feed_dict['word2code']
        nonleaf_idx = feed_dict['non_leaf_code2idx']
        code2idx = feed_dict['code2idx']
        
        W_emb = torch.randn(len(word2idx), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(nonleaf_idx), dimension) / (dimension**0.5)
        window_size = 4
        losses=[]
        for i in range(iteration): 
            #Training word2vec using SGD
            centerword, context = getRandomContext(corpus, window_size)
            #print(context)
            center_word_code = word2code[centerword] 
            context_words_codes = [word2code[i] for i in context]

            node_code=''
            center_word_activated_node_code_lst = []
            for char in center_word_code:
                center_word_activated_node_code_lst.append(node_code)
                node_code += char
            
            #center_word_activated_node_code_lst = center_word_activated_node_code_lst[:-1]
            center_word_activated_node_idx_lst = [list(nonleaf_idx[center_word_activated_node_code])[0] for center_word_activated_node_code in center_word_activated_node_code_lst]
            
            node_code=''
            context_words_activated_node_code_lst = []
            for word_code in context_words_codes:
                context_word_activated_node_code_lst = []
                node_code=''
                for char in word_code:
                    context_word_activated_node_code_lst.append(node_code)
                    node_code += char
                context_words_activated_node_code_lst.append(context_word_activated_node_code_lst)

            #context_words_activated_node_code_lst = context_words_activated_node_code_lst[:-1]
            #각 context마다 영향을 끼진 non_leaf node 의 idx가 들어있음.
            context_words_activated_node_idx_lst = []
            for context_word_activated_node_code_lst in context_words_activated_node_code_lst:
                context_words_activated_node_idx_lst.append([list(nonleaf_idx[context_word_activated_node_code])[0] for context_word_activated_node_code in context_word_activated_node_code_lst])

            #얘네의 idx는 실제 단어의 idx
            centerInd =  word2idx[centerword]
            contextInds = [word2idx[context_word] for context_word in context]
            
            #print(contextInds)
            #print(idx2word[contextInds[0]])
            if mode == "CBOW":
                
                feed_dict2['center_word_code']= center_word_code
                
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[center_word_activated_node_idx_lst], update_system, feed_dict2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out[center_word_activated_node_idx_lst] -= learning_rate*G_out
                losses.append(L)
                    
                    
            elif mode=="SG":
                #print(contextInds)
                for i, contextInd in enumerate(contextInds):
                    
                    feed_dict2['context_word_code'] = context_words_codes[i]
                    
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[context_words_activated_node_idx_lst[i]], update_system, feed_dict2)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out[context_words_activated_node_idx_lst[i]] -= learning_rate*G_out
                    losses.append(L)
            else:
                print("Unkwnown mode : "+mode)
                exit()
            
            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,))
                losses=[]
                
            
    elif update_system == "NS":
        W_emb = torch.randn(len(word2idx), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(word2idx), dimension) / (dimension**0.5)
        window_size = 4
        losses=[]
        
        prob_table = feed_dict['prob_table']
        sum_of_pow_freq = feed_dict['sum_of_pow_freq']
        
        sampling_num = 5
        #len_corpus = len(corpus)
        
        for i in range(iteration):
            #Training word2vec using SGD
            
            centerword, context = getRandomContext(corpus, window_size)

            centerInd =  word2idx[centerword]
            contextInds = [word2idx[i] for i in context]            
            
            if mode == "CBOW":
                #except_idx = centerInd
                
                activated_node_lst = get_activated_node(sum_of_pow_freq, sampling_num, prob_table, centerInd)
                activated_node_lst.append(centerInd)
                
                feed_dict2['activated_node_lst']=activated_node_lst
                
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[activated_node_lst], update_system, feed_dict2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out[activated_node_lst] -= learning_rate*G_out
                losses.append(L)
                
            elif mode=="SG":
                for contextInd in contextInds:
                    #except_idx = contextInd
                    
                    activated_node_lst = get_activated_node(sum_of_pow_freq, sampling_num, prob_table, contextInd)
                    activated_node_lst.append(contextInd)
                    
                    feed_dict2['activated_node_lst'] = activated_node_lst
                    
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[activated_node_lst], update_system, feed_dict2)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out[activated_node_lst] -= learning_rate*G_out
                    losses.append(L)
                    
            else:
                print("Unkwnown mode : "+mode)
                exit()
            
            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,))
                losses=[]
    #기존과 동일
    elif update_system == "BS":
        W_emb = torch.randn(len(word2idx), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(word2idx), dimension) / (dimension**0.5)  
        window_size = 5

        losses=[]
        for i in range(iteration):
            #Training word2vec using SGD
            centerword, context = getRandomContext(corpus, window_size)
            centerInd =  word2idx[centerword]
            contextInds = [word2idx[i] for i in context]

            if mode=="CBOW":
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out, update_system)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out -= learning_rate*G_out
                losses.append(L)

            elif mode=="SG":
                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out, update_system)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out -= learning_rate*G_out
                    losses.append(L)
            else:
                print("Unkwnown mode : "+mode)
                exit()

            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,))
                losses=[]
    
    else:
        print("What is it?")
        exit()


    return W_emb, W_out


# def sim(testword, word2ind, ind2word, matrix):
#     length = (matrix*matrix).sum(1)**0.5
#     wi = word2ind[testword]
#     inputVector = matrix[wi].reshape(1,-1)/length[wi]
#     sim = (inputVector@matrix.t())[0]/length
#     values, indices = sim.squeeze().topk(5)
    
#     print()
#     print("===============================================")
#     print("The most similar words to \"" + testword + "\"")
#     for ind, val in zip(indices,values):
#         print(ind2word[ind.item()]+":%.3f"%(val,))
#     print("===============================================")
#     print()
    
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
            
    return sim_dict
    


def main():
    
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('update_system', metavar='update_system', type=str,
                        help='"HS for Hierarchical Softmax, NS for Negative Sampling, BS for Basic Softmax')
    parser.add_argument('sub_sampling', metavar='sub_sampling', type=bool,
                        help='true for sub_sampling or false for not')
    args = parser.parse_args()
    part = args.part
    mode = args.mode
    update_system = args.update_system
    sub_sampling = args.sub_sampling
    
    print("loading...")
    
#     text = open('questions-words.txt',mode='r').readlines() #각 줄을 읽어서 리스트로 반환
#     text =  text[1:]

#     corpus=[]
#     for t in text:
#         t = t.lower()
#         corpus += t.split()

    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()
        
    
    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus) #dict
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>4:
            processed.append(word)
            
    frequency = Counter(processed)
    vocabulary = set(processed)
    
    #Assign an index number to a word
    word2idx = {}
    word2idx[" "]=0
    i = 1
    for word in vocabulary:
        word2idx[word] = i
        i+=1
    idx2word = {}
    for k,v in word2idx.items():
        idx2word[v]=k
    
    feed_dict = {} #update_system에 따라 필요한 것 전달하는 용도
    
#     '''
#     Sub_Sampling or Not
#     '''
#     if sub_sampling == True:
#         sum_of_freq = 0
#         for word, freq in frequency.items():
#             sum_of_freq += freq
        
#         #print(sum_of_freq) 
        
#         ratio_of_freq = {}
#         for word, freq in frequency.items():
#             ratio_of_freq[word] = freq / sum_of_freq
        
#         #print(ratio_of_freq) #f(w_i)
        
#         #t = 0.00005 #corpus의 크기가 작으므로 t를 키워야 전체 값이 작아지면서 단어들이 적게 사라짐. 너무 많이 사라지면 학습이 힘듬.
#         t = 0.0005
#         prob_of_ss = {}
#         for word, freq in frequency.items():
#             prob_of_ss[word] = 1- (math.pow(t / ratio_of_freq[word],0.5)) #P(w_i) , discard frequent words with prob

#         feed_dict['prob_of_ss'] = prob_of_ss
        
    
#     '''
#     For Hierarchical Softmax
#     '''
#     if update_system == "HS":
#         '''
#         Huffman Coding
#         '''
#         print("Start Huffman Coding")

#         HC = HuffmanCoding()
#         _,_ = HC.build(frequency) 

#         print("Finish Huffman Coding")

#         code2idx = {}
        
#         for word,code in HC.codes.items():
#             code2idx[code] = word2idx[word]
        
#         word2code = HC.codes
#         code2word = HC.reverse_mapping
        
#         #word 관련
#         feed_dict['word2code']= word2code
#         feed_dict['code2word']= code2word
#         feed_dict['code2idx']= code2idx
#         feed_dict['idx2word']= idx2word
        
#         #non_leaf 관련
#         feed_dict['non_leaf_code2idx'] = HC.nonleaf_ind
        
#     '''
#     For Negative Sampling
#     '''
#     if update_system == "NS":        
#         pow_of_freq = {}
#         for word, freq in frequency.items():
#             pow_of_freq[word] = math.pow(freq,0.75)
        
#         sum_of_pow_freq = 0
#         for freq in pow_of_freq.values():
#             sum_of_pow_freq += freq
#         #테이블 방법
#         prob_table = np.zeros(int(sum_of_pow_freq))
#         idx = 0
#         for word, freq in pow_of_freq.items():
#             freq = int(freq)
#             prob_table[idx:idx+freq] = word2idx[word]
#             idx = idx+freq 
#         #print(prob_table)
        
#         feed_dict['prob_table'] = prob_table
#         feed_dict['sum_of_pow_freq']= int(sum_of_pow_freq) 
        
#     '''
#     For Basic Softmax
#     '''
#     if update_system == "BS":
#         pass
    
#     emb,_ = word2vec_trainer(processed, word2idx, mode, update_system, sub_sampling, feed_dict, dimension=64, learning_rate=0.05, iteration=50000)
    
    
    m = ["CBOW", "SG"]
    u = ["HS","NS","BS"]
    s = [True, False]
    
    
    for mode in m:
        for update_system in u:
            for sub_sampling in s:
                start = time.time()
                print("mode: ", mode)
                print("update_system: ",update_system)
                print("sub_sampling: ",sub_sampling)
                '''
                Sub_Sampling or Not
                '''
                if sub_sampling == True:
                    sum_of_freq = 0
                    for word, freq in frequency.items():
                        sum_of_freq += freq

                    #print(sum_of_freq) 

                    ratio_of_freq = {}
                    for word, freq in frequency.items():
                        ratio_of_freq[word] = freq / sum_of_freq

                    #print(ratio_of_freq) #f(w_i)

                    #t = 0.00005 #corpus의 크기가 작으므로 t를 키워야 전체 값이 작아지면서 단어들이 적게 사라짐. 너무 많이 사라지면 학습이 힘듬.
                    t = 0.0005
                    prob_of_ss = {}
                    for word, freq in frequency.items():
                        prob_of_ss[word] = 1- (math.pow(t / ratio_of_freq[word],0.5)) #P(w_i) , discard frequent words with prob

                    feed_dict['prob_of_ss'] = prob_of_ss


                '''
                For Hierarchical Softmax
                '''
                if update_system == "HS":
                    '''
                    Huffman Coding
                    '''
                    print("Start Huffman Coding")

                    HC = HuffmanCoding()
                    _,_ = HC.build(frequency) 

                    print("Finish Huffman Coding")

                    code2idx = {}

                    for word,code in HC.codes.items():
                        code2idx[code] = word2idx[word]

                    word2code = HC.codes
                    code2word = HC.reverse_mapping

                    #word 관련
                    feed_dict['word2code']= word2code
                    feed_dict['code2word']= code2word
                    feed_dict['code2idx']= code2idx
                    feed_dict['idx2word']= idx2word

                    #non_leaf 관련
                    feed_dict['non_leaf_code2idx'] = HC.nonleaf_ind

                '''
                For Negative Sampling
                '''
                if update_system == "NS":        
                    pow_of_freq = {}
                    for word, freq in frequency.items():
                        pow_of_freq[word] = math.pow(freq,0.75)

                    sum_of_pow_freq = 0
                    for freq in pow_of_freq.values():
                        sum_of_pow_freq += freq
                    #테이블 방법
                    prob_table = np.zeros(int(sum_of_pow_freq))
                    idx = 0
                    for word, freq in pow_of_freq.items():
                        freq = int(freq)
                        prob_table[idx:idx+freq] = word2idx[word]
                        idx = idx+freq 
                    #print(prob_table)

                    feed_dict['prob_table'] = prob_table
                    feed_dict['sum_of_pow_freq']= int(sum_of_pow_freq) 

                '''
                For Basic Softmax
                '''
                if update_system == "BS":
                    pass

                emb,_ = word2vec_trainer(processed, word2idx, mode, update_system, sub_sampling, feed_dict, dimension=200, learning_rate=0.05, iteration=100000)
                
                consume_time = time.time() - start
#                 print("dimension: ",dimension)
#                 print("lr: ", learning_rate)
#                 print("iteration: ",iteration)
                print("consum_time: ", consume_time)
    print("dimension: ",dimension)
    print("lr: ", learning_rate)
    print("iteration: ",iteration)

    '''
    Test for Training
    '''
    exit()
    sim_dict = find_sim_word(emb, "germany", "berlin", "france", word2code, code2idx) #마지막에 들어갈 단어는?
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
main()
