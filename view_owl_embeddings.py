import rdflib 
import networkx
import numpy as np
import pykeen.pipeline
from pykeen.triples import TriplesFactory
import ampligraph
from sklearn.decomposition import PCA
from matplotlib import pyplot
import tensorflow

from owl_embeddings import owl_embeddings_get




entity_embeddings,relation_embeddings,triplets=owl_embeddings_get('aio.owl')

tripletsin_np=np.array(triplets)
print("Entity Embeddings (First 5):", entity_embeddings[:5])
print("Relation Embeddings (First 5):", relation_embeddings[:5])

pca=PCA(n_components=2)
fixedembeddings=pca.fit_transform(entity_embeddings.detach().numpy())
fixedrelationships=pca.fit_transform(relation_embeddings.detach().numpy())

pyplot.figure(figsize=(10,10))
pyplot.subplot(1,2,1)
pyplot.scatter(fixedembeddings[:, 0], fixedembeddings[:, 1])
for i, entity in enumerate(tripletsin_np[:, 0][:10]):
    pyplot.annotate(entity, (fixedembeddings[i, 0], fixedembeddings[i, 1]))

pyplot.title("entity embeddings in 2d")

pyplot.subplot(1,2,2)
pyplot.scatter(fixedrelationships[:,0],fixedrelationships[:,1])
for i, relationship in enumerate(tripletsin_np[:,0][:10]):
    pyplot.annotate(relationship,(fixedrelationships[i,0],fixedrelationships[i,1]))

pyplot.title("relationship embeddings scatterplot")
pyplot.show()