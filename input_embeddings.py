import random
import torch 
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from read_input import reading
import numpy
from matplotlib import pyplot

def input_embeddings_get(text_file):
    text_input=reading(text_file)
    #needs to recognise multiple sentences
    sentences=text_input.split('.')
    for s in sentences:
        if(len(s.strip())==0):
            sentences.remove(s)

    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("loading tokenizer and model...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("tokenizer loaded")
    model = BertModel.from_pretrained('bert-base-uncased')
    print("model loaded")


    encoding = tokenizer.batch_encode_plus( 
            sentences,# List of input texts
            padding=True,              
            truncation=True,           
            return_tensors='pt',     
            add_special_tokens=True    
        )

    input_ids = encoding['input_ids']
    print(f"Input ID: {input_ids}")
    attention_mask = encoding['attention_mask']
    print(f"Attention mask: {attention_mask}")

    with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
    word_embeddings = outputs.last_hidden_state 
    print(f"Shape of Word Embeddings: {word_embeddings.shape}")
    return word_embeddings

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


