import pandas as pd
import numpy as np
from numpy.linalg import svd
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import statistics
from owl_embeddings import owl_embeddings_get
from input_embeddings import input_embeddings_get
from rotation_helpers import translate,norm_vectors

text_embeddings=input_embeddings_get('text.txt')
text_embeddings_df=pd.DataFrame(np.array(text_embeddings))
entity_embeddings,relation_embeddings,triplets=owl_embeddings_get('aio.owl')
entity_embeddings_df=pd.DataFrame(np.array(entity_embeddings))
relation_embeddings_df=pd.DataFrame(np.array(relation_embeddings))
triplets_df=pd.DataFrame(np.array(triplets))

###########
#TRANSLATE 
###########

text_embeddings_df=translate(text_embeddings_df)
entity_embeddings_df=translate(entity_embeddings_df)
relation_embeddings_df=translate(relation_embeddings_df)
triplets_df=translate(triplets_df)

###########
#NORMALISE
###########

text_embeddings_df=norm_vectors(text_embeddings_df)
entity_embeddings_df=norm_vectors(entity_embeddings_df)
relation_embeddings_df=norm_vectors(relation_embeddings_df)
triplets_df=norm_vectors(triplets_df)







