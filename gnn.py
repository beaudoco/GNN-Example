import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#example of a large graph in a dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)

data = dataset[0]
print(data.is_undirected())
#this time data holds labels for each node
#it also holds which nodes to train, test and validate
#we don't need to set this up we can just perform our own
#splitting of the data as done below
print(data.train_mask.sum().item())
print(data.val_mask.sum().item())
print(data.test_mask.sum().item())

#load a data set from the library of available data
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

#print the size...should be 600
print(len(dataset))

#looking at the first graph
data = dataset[0]

#we see the graph is undirected
print(data.is_undirected())
#get the keys of the data
print(data.keys)
#print the nodes in the x key
print(data['x'])

#printing the x and y key
for key, item in data:
    print(f'{key} found in data')

#randomly shuffle the dataset this isn't necessary but can be useful
dataset = dataset.shuffle()
#shuffle is the equivalent of this
# perm = torch.randperm(len(dataset))
# dataset = dataset[perm]

#split the data set for testing and training
train_dataset = dataset[:540]
test_dataset = dataset[540:]

#torch geometric offers a dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

#batch is a column vector which maps each node to its
#respective graph in the batch
for data in loader:
    #print the batch info and the graphs per batch
    #to show that it worked
    #last batch is smaller in this case
    print(data)
    print(data.num_graphs)

    #batch can be used to do things such as get the average
    #node features in the done dimension for each indvidual graph
    x = scatter_mean(data.x, data.batch, dim=0)
    print(x.size())

#loading shapenet contains 17k 3D shapes and 16 shape categories
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
#we can use things like pre-transform to perform augmentation this 
#comes from the idea of torchvision transforms
#we can perform these transforms after loading data, but this is
#a cleaner way to do it on pre-defined data
# MAKE SURE TO DO THIS BEFORE LOADING THE DATA THE FIRST TIME
# IF YOU DON'T YOU HAVE TO GO TO THE DIR AND DELETE THE DATA
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], pre_transform=T.KNNGraph(k=6))
#we can also perform random augmentations to each node
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6),
#                     transform=T.RandomJitter(0.01))
print(dataset[0])

#back to CORA to explore GCN
dataset = Planetoid(root='/tmp/Cora', name='Cora')

#define simple GCN use RELU between layers to remove linearity
#use softmax measure
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
#choose dataset and optimizer
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
#training
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    #define loss
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#evaluate training results
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')