"""GCN implementation based on https://github.com/iarai/NeurIPS2021-traffic4cast.
The helper functions used to run the model can be found on the Traffic4Cast GitHub"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HypergraphConv


class Graph_resnet(torch.nn.Module):
    def __init__(self, num_features,num_classes, nh=38, K=6, K_mix=2, inout_skipconn=True, depth=3, p=0.5,bn=False):


        super(Graph_resnet, self).__init__()
                    
        self.p = 0.5
        #KipfBlock
        self.conv1 = HypergraphConv(96, 80)
        self.bn1 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)
        #skip1
        self.sk1 = HypergraphConv(96, 80)

        #KipfBlock2
        self.conv2 = HypergraphConv(80, 80)
        self.bn2 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip2
        self.sk2 = HypergraphConv(80, 80)

        #KipfBlock3
        self.conv3 = HypergraphConv(80, 80)
        self.bn3 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip3
        self.sk3 = HypergraphConv(80, 80)

        #KipfBlock4
        self.conv4 = HypergraphConv(80, 80)
        self.bn4 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)

        #skip4
        self.sk4 = HypergraphConv(80, 80)

        #KipfBlock5
        self.conv5 = HypergraphConv(80, 80)
        self.bn5 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)
        
        #skip5
        self.sk5 = HypergraphConv(80, 80)

        #KipfBlock6
        self.conv6 = HypergraphConv(80, 80)
        self.bn6 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)
        
        #skip6
        self.sk6 = HypergraphConv(80, 80)

        #KipfBlock7
        self.conv7 = HypergraphConv(80, 80)
        self.bn7 = torch.nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True)
        
        #skip7
        self.sk7 = HypergraphConv(80, 80)

        #inout_skipconnection
        self.conv_mix = HypergraphConv(176, 48)
          
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

        #block6
        block6 = F.relu(self.bn6(self.conv6(x, edge_index)))
        block6 = F.dropout(block6, training=self.training, p=self.p)
        #skip6
        x = block6 + self.sk6(x, edge_index)
        
        #block7
        block7 = F.relu(self.bn7(self.conv7(x, edge_index)))
        block7 = F.dropout(block7, training=self.training, p=self.p)
        #skip7
        x = block7 + self.sk7(x, edge_index)
        
        x = torch.cat((x, data.x), 1)
        x = self.conv_mix(x, edge_index)



        return x

