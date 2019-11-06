import torch
from random import shuffle
from collections import Counter
import argparse
import random

from huffman import HuffmanCoding

def getRandomContext(corpus, C=8):
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


def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    
    #get hidden layer
    center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D
    #print(center_word_vector.size())
    #score
    score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)
    
    e = torch.exp(score_vector) 
    softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V
    #print(softmax.size())
    #print(softmax)
    
    loss = -torch.log(softmax[:,contextWord])
    
    #get grad
    softmax_grad = softmax
    softmax_grad[:,contextWord] -= 1.0
    
    grad_out = torch.matmul(torch.t(softmax_grad), center_word_vector) #(V,1) * (1,D) = (V,D)
    grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
    
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix, center_word_activated_node_idx_lst, context_words_activated_node_idx_lst, center_word_code):
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
    
    #result_vector = result_vector / len(contextWords) #1,D
    #print(result_vector.size())
    #score
    score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
    score_vector = torch.t(score_vector)

    p_center_word = 1.0
    for i, code in enumerate(center_word_code):
        #print(code)
        #print(i)
        if code == '0':
            #idx = center_word_activated_node_idx_lst[i]
            #print(idx)
            p_center_word *= 1/(1+ torch.exp(score_vector[i]))
        elif code == '1':
            #idx = center_word_activated_node_idx_lst[i]
            #print(idx)
            p_center_word *= (1-(1/(1+ torch.exp(score_vector[i]))))
        
        else:
            print("what is it?", code)
            exit()
        #print(p_center_word)
    #print("p_center_word: ",p_center_word)
    #score_vector = torch.t(score_vector)

    
    loss = -torch.log(p_center_word)
    #print("loss: ",loss)
    
    score_grad = score_vector - 1
    #for idx in center_word_activated_node_idx_lst:
    #    score_grad[idx] = score_grad[idx] -1 # (K, 1)
    
    #exit()
    #score_grad[:,centerWord] -= 1.0
    #softmax_grad = softmax_grad.view(1,-1)
    
    #grad_out = torch.matmul(torch.t(softmax_grad), result_vector) #K,D
    grad_out = torch.matmul(score_grad, sum_of_context_words_vector) #(K,1) * (1,D) = (K,D)
    grad_emb = torch.matmul(torch.t(score_grad), outputMatrix) #(1,K) * (K,D) = (1,D)

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_emb, grad_out

#원래 iter 100000
def word2vec_trainer(corpus, nonleaf_ind, word2code, code2idx, mode="CBOW", dimension=100, learning_rate=0.025, iteration=2, ):
# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2code), dimension) / (dimension**0.5) 
    W_out = torch.randn(len(nonleaf_ind), dimension) / (dimension**0.5)  
    window_size = 5

    
    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        
        #print("centerword: ", centerword)
        #print("context: ", context)
        
        center_word_code = word2code[centerword] #9자리
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
        #print("centerword_activated_code: ",center_word_activated_node_code_lst)
        #print("centerword_activated_idx: ",center_word_activated_node_idx_lst)
        
            
        node_code=''
        context_words_activated_node_code_lst = ['']
        for word_code in context_words_codes:
            node_code=''
            for char in word_code:
                node_code += char
                if node_code not in context_words_activated_node_code_lst and len(node_code) != len(word_code):
                    context_words_activated_node_code_lst.append(node_code)
        
        #context_words_activated_node_code_lst = context_words_activated_node_code_lst[:-1]
        context_words_activated_node_idx_lst = [list(nonleaf_ind[context_words_activated_node_code])[0] for context_words_activated_node_code in context_words_activated_node_code_lst]
        
        #얘네의 idx는 실제 단어의 idx
        centerInd =  code2idx[word_code]
        
        contextInds = [code2idx[word_code] for word_code in context_words_codes]
        
        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out[center_word_activated_node_idx_lst] ,center_word_activated_node_idx_lst, context_words_activated_node_idx_lst, center_word_code)
            W_emb[contextInds] -= learning_rate*G_emb
            W_out[center_word_activated_node_idx_lst] -= learning_rate*G_out
            #losses.append(L.item())
            losses.append(L)

        elif mode=="SG":
            for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[context_words_activated_node_idx_lst])
                W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                W_out[context_words_activated_node_idx_lst] -= learning_rate*G_out
                losses.append(L.item())

        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
        	avg_loss=sum(losses)/len(losses)
        	print("Loss : %f" %(avg_loss,))
        	losses=[]

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


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part

	#Load and tokenize corpus
#     print("loading...")
#     if part=="part":
#         text = open('questions-words.txt',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
#     elif part=="full":
#         text = open('questions-words.txt',mode='r').readlines()[0] #Load full corpus for submission
#     else:
#         print("Unknown argument : " + part)
#         exit()
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
    #print(frequency)
    #print(vocabulary)
    print("frequ size", len(frequency))
    print("vocab size", len(vocabulary))
    '''
    Huffman Coding
    '''
    print("Start Huffman Coding")
    
    HC = HuffmanCoding()
    _,_ = HC.build(frequency) # HC.codes(word, code(leaf_node)), HC.nonleaf_ind(code, ID)
    #leaf_node의 자릿수는 9자리
    #print("Heap: ",HC.heap)
    print("codes:",HC.codes)
    print("HC.codes's len: ",len(HC.codes))
    print()
    print("nonleaf_ind: ",HC.nonleaf_ind)
    print("HC.nonleaf_ind's len: ",len(HC.nonleaf_ind))
    print()
    
    print("Finish Huffman Coding")
    
    #Assign an index number to a word
#     word2ind = {}
#     word2ind[" "]=0
#     i = 1
#     for word in frequency:
#         word2ind[word] = i
#         i+=1
#     ind2word = {}
#     for k,v in word2ind.items():
#         ind2word[v]=k
        
    wordcode2idx = {}
    i=0
    for word,code in HC.codes.items():
        wordcode2idx[code] = i
        i += 1


#     print("word2idx size", len(word2idx)) 
#     print("idx2word size", len(idx2word)) 
#     print(word2ind)
    print()
    
    #print(processed)
    #Training section
    emb,_ = word2vec_trainer(processed, HC.nonleaf_ind, HC.codes, wordcode2idx, mode=mode, dimension=64, learning_rate=0.05, iteration=50000)
    
    #Print similar words
#     testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
#     for tw in testwords:
#     	sim(tw,word2idx,idx2word,emb)

main()
