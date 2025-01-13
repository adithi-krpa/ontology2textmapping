from input_embeddings import input_embeddings_get
from matplotlib import pyplot 
import torch
from sklearn.decomposition import PCA
import random 
from read_input import reading 

text_input=reading('text.txt')
sentences=text_input.split('.')
word_embeddings=input_embeddings_get('text.txt')
sentence_embeddings = torch.mean(word_embeddings, dim=1)
sentence_embeddings_np = sentence_embeddings.detach().numpy()
pca=PCA(n_components=2)
decomposedwe=pca.fit_transform(sentence_embeddings_np)

pyplot.figure(figsize=(10,10))
pyplot.scatter(decomposedwe[:,0],decomposedwe[:,1])
for i,e in enumerate(sentences):
    pyplot.annotate(e, (decomposedwe[i,0],decomposedwe[i,1]))

pyplot.title("scatterplot of sentences")
pyplot.show()