# Word2Vec with Skip-gram and CBOW with Hierarchical Softmax, Negative Sampling and Sub-sampling

- Korea University Information Retrieval(COSE 472) Assignment4

-----

If you run "word2vec.py", you can train and test your models.
- How to run
```python
python word2vec.py [partition] [mode] [update_system] [sub_sampling]
```
- mode
  - "SG" for skipgram
  - "CBOW" for CBOW
- partition
  - "part" if you want to train on a part of corpus (fast training but worse performance), 
  - "full" if you want to train on full corpus (better performance but very slow training)
- update_system 
  - "HS" for Hierarchical Softmax
  - "NS" for Negative Sampling
  - "BS" for Basic Softmax

- Examples) 
  - python word2vec.py full SG HS True
  - python word2vec.py part CBOW NS False
  
  
## Word2Vec Implementation
### 1. Basic Softmax

![bs](https://user-images.githubusercontent.com/38184045/71541363-2f646600-299b-11ea-9616-f3260ea2ade7.png)

<그림 1. Basic Softmax>

![softmax](https://user-images.githubusercontent.com/38184045/71541362-2a9fb200-299b-11ea-83ca-3515c1d81056.png)

<수식 1. Softmax>

- Assignment 3에 구현한 CBOW와 Skip-gram의 경우에는 Basic Softmax(이하 BS)의 방법으로 각 단어(V개)의 Score Vector를 계산하였다. 이 방법은 이해하기 직관적이고 간단하다. 하지만, V(vocb에 단어의 수)가 커진다면, <그림 1>에서 볼 수 있듯이 한 노드(단어)의 Score를 계산하기 위해 Softmax의 방법을 사용한다. Softmax의 값을 계산하기 위해서는 <수식 1>처럼 V개의 모든 Exponential의 값을 매 iteration마다 구해야한다. 이러한 비효율적인 점을 보완하기 위해 나온 것이 Hierarchical Softmax와 Negative Sampling이다.

### 2. Hierarchical Softmax

![hs](https://user-images.githubusercontent.com/38184045/71541498-4441f900-299d-11ea-8c0e-f9a8d8a715c0.png)

<그림 2. Hierarchical Tree>

![sigmoid](https://user-images.githubusercontent.com/38184045/71541504-6dfb2000-299d-11ea-846f-e74104681d96.png)

<그림 3, 수식 2. sigmoid function>

- Hierarchical Softmax(이하 HS)는 이름에서도 알 수 있듯 계층적으로 이뤄지는 방법이다. 기존 BS방법에서는 output으로 V개의 노드를 가진 ScoreVector를 가지고, 이 vector에서 한 개의 노드가 한 개의 단어의 점수를 나타내었다. 하지만, HS에서는 V개의 단어가 leaf node에 위치한다고 가정하고 binary tree로 나타내었을때의 non-leaf들의 node의 값을 output으로 가진다. 

- 트리의 말단에 단어의 노드들이 존재하므로, 그 단어에 대한 score를 구하기 위해서는 단순히 root node부터 leaf node까지 일정한 연산을 하며 내려가면 된다. 위에서 언급했듯 각 노드들은 일정한 값을 가지고 있다. 하지만, 기존 BS에서 연산에서 구한 Softmax값은 확률이므로 0과 1사이의 값을 가지듯 HS의 결과도 1과 0 사이의 값을 가져야한다.따라서, 트리를 내려가기 전 <수식 2>의 Sigmoid 연산을 취해서 <그림 1>에서 나타난 것처럼 왼쪽으로 내려갈 때에는 Sigmoid(x), 오른쪽으로 내려갈 땐 (1-Sigmoid(x))의 값을 내려 보내준다. <그림 3>의 그래프에서 볼 수 있듯, Sigmoid의 값은 항상 0과 1의 사이의 값을 가지므로 아무리 많이 내려가게 되어도 값은 0과 1사이를 벗어날 수 없게되고, 결국 우리는 결과값을 단어의 확률이라고 가정하고 사용하게된다.

![hs_gradient](https://user-images.githubusercontent.com/38184045/71541516-971bb080-299d-11ea-9cc0-f60cce4e47c1.png)

<그림 4. Hierarchical Softmax의 Gradient>

![hs_gradient_2](https://user-images.githubusercontent.com/38184045/71541530-c3373180-299d-11ea-81b2-df35c6fca4f1.png)

<그림 5. HS의 gradient 계산과정>

- HS의 output의 gradient 계산은 간단하다. <그림 4>처럼 score vector의 값 전부에 <수식 2>와 같은 Sigmoid를 취해준 후, <그림 5>와 같은 계산에서 볼 수 있듯이, Sigmoid(x)를 취해서 내려 보내준 값에 1을 뺀 값, (1-Sigmoid(x))을 보낸 값에는 아무런 연산을 하지않은 것이 최종 gradient가 되는 것을 알 수 있다.

- 결과적으로, HS에서는 한 iteration에서 평균 log(V)만큼의 연산만한다면 loss를 구할 수 있게된다. BS에서 한 iteration마다 V개의 노드에 대한 exponential계산을 하던 것에 비하면 훨씬 효율적이라고 말 할 수 있을 것이다. 

### 3. Negative Sampling

![ns](https://user-images.githubusercontent.com/38184045/71541549-1d37f700-299e-11ea-89b5-7b69d91f9fcc.png)

<그림 6. Negative Sampling>

- Negative Sampling(이하 NS)이란 <그림 1>의 BS방식과 모두 똑같이 진행하지만, 다른 점은 Activation Function의 종류와 계산하는 총 노드의 개수이다. NS에서는 특정 방법으로 정답 노드를 제외하고 K개의 node를 sampling한다. 이 때의 정답 노드가 아닌 Sample들을 Negative Sample이라한다. 총 (K+1)개의 노드를 전체 노드로 취급하여 loss를 계산하는 과정을 거친다. 

![ns_table](https://user-images.githubusercontent.com/38184045/71541550-1e692400-299e-11ea-9efe-e25eba8f7a6c.png)

<그림 7. NS table>

- Negative Sample을 추출하는 방법은, corpus에서의 단어의 빈도수를 사용한다. <그림 7>처럼 각 단어의 빈도수에 0.75를 제곱한 값을 사용하여 table을 하나 생성한다. 그 다음 0과 이 테이블의 길이 사이의 난수를 생성하여 해당하는 word를 negative sample로 사용하게 된다.

- 뽑힌 negative sample들과 정답 sample들을 사용한 loss, gradient계산은 <그림 6>에 나와 있듯 HS와 똑같은 방식으로 이루어진다. 결과적으로 위 방법도 모든 V개의 노드를 사용하는 것이 아니라, sampling된 개수만큼의 노드만을 사용하기 때문에 계산량이 BS에 비해 대폭 줄어들게 된다.

### 4. Sub-sampling


![ss](https://user-images.githubusercontent.com/38184045/71541581-8455ab80-299e-11ea-8a81-8627a1d386c2.png)

<그림 7. Subsampling>

![ss_prob](https://user-images.githubusercontent.com/38184045/71541582-86b80580-299e-11ea-9bd5-c3441f38e172.png)

<수식 3. Subsampling Probability>

- Subsampling(이하 SS)의 취지는 <그림 7>과 같이 a, the, it과 같이 많이 나오는 단어들은 학습하는데 도움이 안되고, 의미가 없다고 판단하여 전체 데이터(corpus)의 크기를 줄여주는 방법이다. 전체 corpus를 돌며, <수식 3>과 같은 확률로 그 단어를 corpus에서 제외시켜주게 된다. f(w)는 단어 w의 corpus에서의 상대 빈도수(해당 단어의 빈도수/모든 단어의 빈도수의 합)이다. f(w)이 값이 크다는 것은 이 단어가 corpus에 자주 등장한다는 의미이다. 그렇다면, 문맥상 안 중요할 확률이 높으므로 discard될 확률이 커지게 된다. t의 값은 상수를 사용하는데, 값이 커질수록 단어를 discard 확률이 낮아진다. 본 실험에서 t의 값은 10e-5를 사용하였다. 

