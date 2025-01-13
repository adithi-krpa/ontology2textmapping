#DIRECTLY FROM THE TRIPLETS IN OWL FILE. NO NEED FOR KNOWLEDGE GRAPHS?
#RESOURCE MANAGEMENT ISSUE. 
import rdflib 
import networkx
import numpy as np
import pykeen.pipeline
from pykeen.triples import TriplesFactory
import ampligraph
from sklearn.decomposition import PCA
from matplotlib import pyplot
import tensorflow

def owl_embeddings_get(owl_file):

    owl_file="aio.owl"
    graph1=rdflib.Graph()

    graph1.parse(owl_file)

    triplets=[]
    for subj,pred,obj in graph1:
        triplets.append([str(subj),str(pred),str(obj)])

    tripletsin_np=np.array(triplets)
    print("first 5 triplets:",triplets[:5])
    triples_factory = TriplesFactory.from_labeled_triples(tripletsin_np)
    training1,testing1,validation1=triples_factory.split([0.8,0.1,0.1])



    results=pykeen.pipeline.pipeline(
        #TransE makes pred=subj+obj. trains model, tests then validates. 
        model="TransE", #can use RotatE if imaginary part of embedding value is handled
        training=training1, # PCA cant handle img 
        testing=testing1,
        validation=validation1,
        epochs=5,  # Control number of epochs
        training_loop='slcwa',  # Define the type of training loop, slcwa: sLCWA or lcwa: LCWA
        training_kwargs=dict(  # Customize training parameters
            batch_size=128,  # Set batch size for training
            num_epochs=5,  # Optionally re-specify the number of epochs
            use_tqdm_batch=False  # Disable tqdm batch progress bars to possibly save some resources
        ),
        #evaluator_kwargs=dict(  # Evaluation customizations
            #batch_size=128,  # This is where you would try to set the evaluation batch size if supported
            #use_tqdm=False  # Disable tqdm during evaluation to reduce resource usage
        #)
        device="cpu"

    )
    entity_embeddings = results.model.entity_representations[0](indices=None)
    relation_embeddings = results.model.relation_representations[0](indices=None)
    return entity_embeddings, relation_embeddings,triplets 













