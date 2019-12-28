# Word2Vec with Skip-gram and CBOW with Hierarchical Softmax, Negative Sampling and Sub-sampling
--------------------------------------------------------------------------------------------------
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
