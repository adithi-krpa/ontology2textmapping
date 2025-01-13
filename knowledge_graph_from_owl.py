###############################
# TOO SLOW FOR LARGE OWL FILE 
###################################

from cProfile import label
import rdflib
import networkx
from matplotlib import pyplot

owlfile="maxo.owl"
graph1=rdflib.Graph()

graph1.parse(owlfile)
kg1=networkx.DiGraph()

for subj,pred,obj in graph1:
    kg1.add_edge(subj,obj,label=pred)

pyplot.figure(figsize=(100,100))
ps=networkx.spring_layout(kg1)
edgel=networkx.get_edge_attributes(kg1,"preds")
networkx.draw(kg1,ps,arrows=True, with_labels=True,node_size=3000,node_color="#5083c7",font_size=10,font_color="black",edge_color="gray")
networkx.draw_networkx_edge_labels(kg1,ps,edge_labels=edgel,font_color="red")

pyplot.title("knowledge graph from owl file")
pyplot.show()
