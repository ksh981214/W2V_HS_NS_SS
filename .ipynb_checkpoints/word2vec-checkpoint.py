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

def getRandomContext(corpus, C=5):
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
    

def get_prob_word_code(word_code, sig):
    p_word = 1.0
    zero_lst=[]

    for i, code in enumerate(word_code):
        if code == '0':
            p_word *= sig[i]
            zero_lst.append(i)
        elif code == '1':
            p_word *= (1 - sig[i])
        else:
            print("what is it?", code)
    return p_word, zero_lst

def get_activated_node(len_corpus, sampling_num, prob_table, correct_idx):
    activated_node_lst = [correct_idx]
    lotto_num = random.randint(0, len_corpus - 1)
    for i in range(sampling_num):
        while lotto_num in activated_node_lst:    
            lotto_num = random.randint(0, len_corpus - 1)
        activated_node_lst.append(int(prob_table[lotto_num]))
        lotto_num = random.randint(0, len_corpus - 1)
    return activated_node_lst

def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix, update_system, feed_dict=None):    
    if update_system == "HS":
        context_word_code = feed_dict['context_word_code']
        
        #get hidden layer
        center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D
        score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)
        
        e = np.exp(score_vector)
        sig_vector = e/(e+1)    

        p_context_word,zero_lst = get_prob_word_code(context_word_code, sig_vector)
        #print("In SG Func, Get Prob")
        loss = -np.log(p_context_word)
        
        sig_grad = sig_vector
        sig_grad[zero_lst] -= 1
        
        grad_out = torch.matmul(sig_grad, center_word_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(sig_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
        
    elif update_system == "NS":
        activated_node_lst =  feed_dict['activated_node_lst']
        
        center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D
        
        score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)

        e = np.exp(score_vector)
        sig_vector = e/(e+1)    
        
        loss = 0.0
        
        for i, idx in enumerate(activated_node_lst):
            if idx == contextWord:
                context_idx = i
                loss -= np.log(sig_vector[i])
            else:
                loss -= np.log(1 - sig_vector[i])

        #get grad
        sig_grad = sig_vector #(K,1)
        sig_grad[context_idx] -= 1

        grad_out = torch.matmul(sig_grad, center_word_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(sig_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
        
    elif update_system == "BS":
        #get hidden layer
        center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D

        #score
        score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)

        e = np.exp(score_vector) 
        softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V

        loss = -np.log(softmax[:,contextWord])

        #get grad
        softmax_grad = softmax
        softmax_grad[:,contextWord] -= 1.0

        grad_out = torch.matmul(torch.t(softmax_grad), center_word_vector) #(V,1) * (1,D) = (V,D)
        grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
        
    else:
        print("What is it?")
        exit()
    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix, update_system, feed_dict=None):
    if update_system == "HS":
        center_word_code = feed_dict['center_word_code']
        
        sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D

        score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)
        e = np.exp(score_vector)

        sig_vector = e/(e+1)        

        p_center_word, zero_lst = get_prob_word_code(center_word_code, sig_vector)
        
        loss = -np.log(p_center_word)

        sig_grad = sig_vector
        sig_grad[zero_lst] -= 1

        grad_out = torch.matmul(sig_grad, sum_of_context_words_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(sig_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
        grad_emb /= 5
        
    elif update_system == "NS":
        activated_node_lst =  feed_dict['activated_node_lst']

        sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D
        
        score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
        score_vector = torch.t(score_vector) # (K,1)
        
        e = np.exp(score_vector)
        sig_vector = e/(e+1)
        
        loss = 0.0
        for i, idx in enumerate(activated_node_lst):
            if idx == centerWord:
                center_idx = i
                loss -= np.log(sig_vector[i])
            else:
                loss -= np.log(1 - sig_vector[i])

        sig_grad = sig_vector #(K,1)
        sig_grad[center_idx] -= 1

        grad_out = torch.matmul(sig_vector, sum_of_context_words_vector) #(K,1) * (1,D) = (K,D)
        grad_emb = torch.matmul(torch.t(sig_vector), outputMatrix) #(1,K) * (K,D) = (1,D)
        grad_emb /= 5

    elif update_system == "BS":

        sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D

        score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)

        e = np.exp(score_vector) 
        softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V

        loss = -np.log(softmax[:,centerWord])

        #get grad
        softmax_grad = softmax
        softmax_grad[:,centerWord] -= 1.0

        grad_out = torch.matmul(torch.t(softmax_grad), sum_of_context_words_vector) #(1,V) * (1,D) = (V,D)
        grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
        grad_emb /= 5
    
    else:
        print("What is it?")
        exit()
    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2idx, mode, update_system, sub_sampling, dimension, learning_rate, iteration, feed_dict=None):
    feed_dict2 = {}

    print("size of corpus: %d" % len(corpus))
    #Only once
    if sub_sampling is True:
        print("Start SubSampling...")
        prob_of_ss = feed_dict['prob_of_ss']
        destiny = np.random.random(size=len(corpus))

        subsampling_word = []
        for idx, word in enumerate(corpus):
            if destiny[idx] < prob_of_ss[word]:
                subsampling_word.append(idx) 
            else:
                pass
            
        corpus = list(np.delete(corpus,subsampling_word))
        print("Finish SubSampling...")
        print("size of corpus(after): %d" % len(corpus))
    
    window_size = 5
    #like 1 epoch
    iteration = len(corpus)
#     if iteration< 10000000:
#         iteration = 16000000
    if mode == "SG":
        iteration = int(iteration/4)
    if iteration < 100000:
        iteration = 100000
    
    print("iteration: {}".format(iteration))

    decay = True
    if decay:
        lr_decay = 5
        lr_decay_iter = int(iteration / lr_decay)
    losses= []
    if update_system == "HS":
        word2code = feed_dict['word2code']
        nonleaf_idx = feed_dict['non_leaf_code2idx']
        code2idx = feed_dict['code2idx']
        
        W_emb = torch.randn(len(word2idx), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(nonleaf_idx), dimension) / (dimension**0.5)
        for i in range(iteration): 
            #Training word2vec using SGD
            centerword, context = getRandomContext(corpus, window_size)

            centerInd =  word2idx[centerword]
            contextInds = [word2idx[context_word] for context_word in context]
            if mode == "CBOW":
                node_code=''
                center_word_code = word2code[centerword] 
                center_word_activated_node_code_lst = []
                for char in center_word_code:
                    center_word_activated_node_code_lst.append(node_code)
                    node_code += char
                center_word_activated_node_idx_lst = [list(nonleaf_idx[center_word_activated_node_code])[0] 
                                                      for center_word_activated_node_code 
                                                      in center_word_activated_node_code_lst]

                feed_dict2['center_word_code']= center_word_code

                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[center_word_activated_node_idx_lst], update_system, feed_dict2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out[center_word_activated_node_idx_lst] -= learning_rate*G_out

                losses.append(L.item())
      
            elif mode=="SG":
                node_code=''
                context_words_codes = [word2code[i] for i in context]
                context_words_activated_node_code_lst = []
                for word_code in context_words_codes:
                    context_word_activated_node_code_lst = []
                    node_code=''
                    for char in word_code:
                        context_word_activated_node_code_lst.append(node_code)
                        node_code += char
                    context_words_activated_node_code_lst.append(context_word_activated_node_code_lst)

                context_words_activated_node_idx_lst = []
                for context_word_activated_node_code_lst in context_words_activated_node_code_lst:
                    context_words_activated_node_idx_lst.append([list(nonleaf_idx[context_word_activated_node_code])[0] for context_word_activated_node_code in context_word_activated_node_code_lst])
                
                for idx, contextInd in enumerate(contextInds):       
                    feed_dict2['context_word_code'] = context_words_codes[idx]
                    
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[context_words_activated_node_idx_lst[idx]], update_system, feed_dict2)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out[context_words_activated_node_idx_lst[idx]] -= learning_rate*G_out
                    losses.append(L.item())
            else:
                print("Unkwnown mode : "+mode)
                exit()
            
            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("i: %d, Loss : %f, learning_rate : %f " %(i, avg_loss, learning_rate))
                losses=[]
            if decay:
                if i%lr_decay_iter==lr_decay_iter-1:
                    learning_rate /= 2

                
            
    elif update_system == "NS":
        W_emb = torch.randn(len(word2idx), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(word2idx), dimension) / (dimension**0.5)
        #losses=[]
        
        prob_table = feed_dict['prob_table']
        sum_of_pow_freq = feed_dict['sum_of_pow_freq']
        
        sampling_num = 5

        for i in range(iteration):
            centerword, context = getRandomContext(corpus, window_size)

            centerInd =  word2idx[centerword]
            contextInds = [word2idx[i] for i in context]            
            if mode == "CBOW":
                activated_node_lst = get_activated_node(sum_of_pow_freq, sampling_num, prob_table, centerInd)

                feed_dict2['activated_node_lst']=activated_node_lst
                
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[activated_node_lst], update_system, feed_dict2)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out[activated_node_lst] -= learning_rate*G_out
                losses.append(L.item()) 


            elif mode=="SG":
                for contextInd in contextInds:
                    activated_node_lst = get_activated_node(sum_of_pow_freq, sampling_num, prob_table, contextInd)
                    #activated_node_lst.append(contextInd)
                    
                    feed_dict2['activated_node_lst'] = activated_node_lst
                    
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[activated_node_lst], update_system, feed_dict2)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out[activated_node_lst] -= learning_rate*G_out
                    losses.append(L.item())
            else:
                print("Unkwnown mode : "+mode)
                exit()
            
            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("i: %d, Loss : %f, learning_rate : %f " %(i, avg_loss, learning_rate))
                losses=[]
            if decay:
                if i%lr_decay_iter==lr_decay_iter-1:
                    learning_rate /= 2
    #기존과 동일
    elif update_system == "BS":
        W_emb = torch.randn(len(word2idx), dimension) / (dimension**0.5) 
        W_out = torch.randn(len(word2idx), dimension) / (dimension**0.5)  
        #window_size = 5

        for i in range(iteration):
            #Training word2vec using SGD
            centerword, context = getRandomContext(corpus, window_size)
            centerInd =  word2idx[centerword]
            contextInds = [word2idx[i] for i in context]
            if mode=="CBOW":
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out, update_system)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out -= learning_rate*G_out
                losses.append(L.item())
            elif mode=="SG":
                #print(i)
                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out, update_system)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out -= learning_rate*G_out
                    losses.append(L.item())
            else:
                print("Unkwnown mode : "+mode)
                exit()

            if i%10000==0:
                avg_loss=sum(losses)/len(losses)
                print("i: %d, Loss : %f, learning_rate : %f " %(i, avg_loss, learning_rate))
                losses=[]
            if decay:
                if i%lr_decay_iter==lr_decay_iter-1:
                    learning_rate /= 2
    else:
        print("What is it?")
        exit()

    return W_emb, W_out

def find_sim_word(emb, w1, w2, w3,ans, word2idx, idx2word):
    idx_w1 = word2idx[w1]
    idx_w2 = word2idx[w2]
    idx_w3 = word2idx[w3]
    
    vec_w1 = emb[idx_w1]
    vec_w2 = emb[idx_w2]
    vec_w3 = emb[idx_w3]

    #vec = vec_w2.sub_(vec_w1).add_(vec_w3)
    vec = -vec_w1 + vec_w2 + vec_w3

    wor = w1 +" "+w2 +" "+w3+" "+ans
    #vec_predicted_size = get_size(list(vec_predicted))
    vec_length = torch.sum((vec*vec))**0.5

    length = (emb*emb).sum(1)**0.5
    vec_normed = vec.reshape(1, -1)/vec_length
    sim = (vec_normed@emb.t())[0]/length
    values, indices = sim.squeeze().topk(13)
    result = {}
    topk = []
    #15개를 뽑아서 w1 w2 w3 를 제외하고 topk에 삽입 후 반환. 최대 13개 최소 10개
    for i in indices:
        temp = idx2word[i.item()]
        if temp != w1 and temp != w2 and temp != w3:
            topk.append(idx2word[i.item()])

    result[wor] = topk
    return result
    
def sim_test(testword, word2idx, idx2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2idx[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(idx2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()

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

    start = time.time()
    print("loading...")

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
    
    '''
    Sub_Sampling or Not
    '''
    if sub_sampling == True:
        sum_of_freq = 0
        for word, freq in frequency.items():
            sum_of_freq += freq       

        ratio_of_freq = {}
        for word, freq in frequency.items():
            ratio_of_freq[word] = freq / sum_of_freq
        
        t = 10e-5

        prob_of_ss = {}
        for word, freq in frequency.items():
            prob_of_ss[word] = 1- (np.sqrt(t / ratio_of_freq[word])) #P(w_i) , discard frequent words with prob

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
            pow_of_freq[word] = freq ** 0.75
        
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

        feed_dict['prob_table'] = prob_table
        feed_dict['sum_of_pow_freq']= int(sum_of_pow_freq) 
        
    '''
    For Basic Softmax
    '''
    if update_system == "BS":
        pass
    dim = 300
    learning_rate = 0.01
    iteration = 100000
    emb,_ = word2vec_trainer(processed, word2idx, mode, update_system, sub_sampling, dim, learning_rate, iteration, feed_dict)
    
    consume_time = time.time() - start
    
    print("mode: ", mode)
    print("update_system: ",update_system)
    print("sub_sampling: ",sub_sampling)
    print("consume_time: ", consume_time)
    print("dim: ",dim)
    print("learning_rate: ",learning_rate)
    '''
    Test for Training
    '''
    #exit()
    
    #Print similar words
    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    for tw in testwords:
    	sim_test(tw,word2idx,idx2word,emb)
    
    #Predict the word using the relation of words
    print("Predict the word using the relation of words")
    start = time.time()
    ques = open('questions-words.txt',mode='r').readlines()
    #ques= ques[1:]
    ques_dict = {}
    cate = None
    for s in ques:
        ls = s.split()
        if len(ls) != 4:
            cate = ls[1] 
            ques_dict[cate] = []
        else:
            ques_dict[cate].append(ls)

    # num_ques = len(ques)

    correct_dict ={}
    for cate in ques_dict.keys():
        correct_dict[cate]={'top_1' : [],'top_5' : [],'top_10' : [], 'cantknow':0}
    
    # correct_top_1_result = []
    # correct_top_5_result = []
    # correct_top_10_result = []
    wrong_result = []
    cantknow = 0
    for cate in ques_dict:
        #problem = ques_dict[cate] 
        for four_w in ques_dict[cate]:
            #temp = ques[i].lower().split()
            w0 = four_w[0].lower()
            w1 = four_w[1].lower()
            w2 = four_w[2].lower()
            ans = four_w[3].lower()
                #맞춘거, 5등 안에 든거, 10등 안에 든거 맞추는 확률로 바꾸기
            if w0 in word2idx.keys() and w1 in word2idx.keys() and w2 in word2idx.keys():
                sim_dict = find_sim_word(emb, w0, w1, w2,ans, word2idx, idx2word)
                if ans == list(sim_dict.values())[0][0]:
                    correct_dict[cate]['top_1'].append(sim_dict)
                if ans in list(sim_dict.values())[0][:5]:
                    correct_dict[cate]['top_5'].append(sim_dict)
                if ans in list(sim_dict.values())[0][:10]:
                    correct_dict[cate]['top_10'].append(sim_dict)
                else:
                    wrong_result.append(sim_dict)
            else:
                correct_dict[cate]['cantknow'] += 1
                cantknow += 1
        # else:
        #     cantknow += 1
    consume_time = time.time() - start
    print("Prediction time: {}".format(consume_time))
    num_of_correct_top_1 = 0 
    num_of_correct_top_5 = 0 
    num_of_correct_top_10 = 0 
    for cate in correct_dict:
        print("In {} region's result , number of ques is {}".format(cate, len(ques_dict[cate])))
        print("correct_top_1_result: {}".format(len(correct_dict[cate]['top_1'])))
        print(correct_dict[cate]['top_1'])
        print("correct_top_5_result: {}".format(len(correct_dict[cate]['top_5'])))
        print(correct_dict[cate]['top_5'])
        print("correct_top_10_result: {}".format(len(correct_dict[cate]['top_10'])))
        print(correct_dict[cate]['top_10'])
        print("cantknow: {}".format(correct_dict[cate]['cantknow']))
        print()
        num_of_correct_top_1 += len(correct_dict[cate]['top_1'])
        num_of_correct_top_5 += len(correct_dict[cate]['top_5'])
        num_of_correct_top_10 += len(correct_dict[cate]['top_10'])
    print("num_of_correct_top_1: {}".format(num_of_correct_top_1))
    print("num_of_correct_top_5: {}".format(num_of_correct_top_5))
    print("num_of_correct_top_10: {}".format(num_of_correct_top_10))
    print("wrong_result: {}".format(len(wrong_result)))
    #print(wrong_result)
    print("cantknow: {}".format(cantknow))
    
    
main()