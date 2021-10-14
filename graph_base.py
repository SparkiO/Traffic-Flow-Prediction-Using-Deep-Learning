"""GCN implementation based on https://github.com/iarai/NeurIPS2021-traffic4cast.
The helper functions used to run the model can be found on the Traffic4Cast GitHub"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv


class Graph_resnet(torch.nn.Module):
    def __init__(self, num_features,num_classes, nh=38, K=6, K_mix=2, inout_skipconn=True, depth=3, p=0.5,bn=False):


        super(Graph_resnet, self).__init__()

        self.p = 0.5
        #KipfBlock
        self.conv1 = ChebConv(96, 80, K=4)
        self.bn1 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip1
        self.sk1 = ChebConv(96, 80, K=1)

        #KipfBlock2
        self.conv2 = ChebConv(80, 80, K=4)
        self.bn2 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip2
        self.sk2 = ChebConv(80, 80, K=1)

        #KipfBlock3
        self.conv3 = ChebConv(80, 80, K=4)
        self.bn3 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip3
        self.sk3 = ChebConv(80, 80, K=1)

        #KipfBlock4
        self.conv4 = ChebConv(80, 80, K=4)
        self.bn4 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip4
        self.sk4 = ChebConv(80, 80, K=1)

        #KipfBlock5
        self.conv5 = ChebConv(80, 80, K=4)
        self.bn5 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip5
        self.sk5 = ChebConv(80, 80, K=1)

        #inout_skipconnection:
        self.conv_mix = ChebConv(176, 48, K=2)
          
    def forward(self, data, **kwargs):
        x, edge_index = data.x, data.edge_index

        #block1
        block1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        block1= F.dropout(block1, training=self.training, p=self.p)
        
        #skip1
        x = block1 + self.sk1(x, edge_index)

        #block2
        block2 = F.relu(self.bn2(self.conv2(x, edge_index)))
        block2= F.dropout(block2, training=self.training, p=self.p)
        #skip3
        x = block2 + self.sk2(x, edge_index)

        #block3
        block3 = F.relu(self.bn3(self.conv3(x, edge_index)))
        block3= F.dropout(block3, training=self.training, p=self.p)
        #skip3
        x = block3 + self.sk3(x, edge_index)

        #block4
        block4 = F.relu(self.bn4(self.conv4(x, edge_index)))
        block4= F.dropout(block4, training=self.training, p=self.p)
        #skip4
        x = block4 + self.sk4(x, edge_index)

        #block5
        block5 = F.relu(self.bn5(self.conv5(x, edge_index)))
        block5 = F.dropout(block5, training=self.training, p=self.p)
        #skip5
        x = block5 + self.sk5(x, edge_index)

        x = torch.cat((x, data.x), 1)
        x = self.conv_mix(x, edge_index)



        return x

