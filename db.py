import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, "relation")

nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title("Knowledge Graph of Project Dependencies")
plt.show()
