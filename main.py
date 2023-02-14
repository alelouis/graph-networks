import torch
from classifier import Classifier
from torch_geometric.datasets import KarateClub

"""
Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." 
arXiv preprint arXiv:1806.01261 (2018).
"""

dataset = KarateClub()
s, r = dataset[0].edge_index
V = dataset[0].x
E = torch.ones((s.shape[0], 1))
Nn, n_dim = V.shape
Ne, e_dim = E.shape
u = torch.tensor([1.0])

classifier = Classifier(Ne, Nn, n_dim, e_dim)
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-2)

accuracy = lambda y_p, y: (y_p == y).sum() / len(y)

for epoch in range(31):
    optimizer.zero_grad()
    y_p = classifier(E, V, u, r, s)
    loss = torch.nn.CrossEntropyLoss()(y_p, dataset[0].y)
    acc = accuracy(y_p.argmax(dim=1), dataset[0].y)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"epoch {epoch:>3} ~ loss={loss:.2f} ~ acc={acc * 100:.1f}%")
