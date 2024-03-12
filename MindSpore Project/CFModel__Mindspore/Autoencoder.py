import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops


class PointNet(nn.Module):
    def __init__(self, feature_dim=3):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.relu= nn.LeakyReLU(alpha=0.01)

        self.pooling = nn.MaxPool1d(1024,2048,return_indices=True)
        self.indices = None

    def forward(self, x):
        x=x.permute(0, 2, 1)   # [B,N,feature_dim]->[B,feature_dim,N]
        output = self.relu(self.bn1(self.conv1(x)))  # [B,64,N]
        output = self.relu(self.bn2(self.conv2(output)))  # [B,64,N]
        output = self.relu(self.bn3(self.conv3(output)))  # [B,64,N]
        output = self.relu(self.bn4(self.conv4(output)))  # [B,128,N]
        output = self.conv5(output)  # [B,1024,N]
        output, self.indices = self.pooling(output)    # [B,1024,1]
        output = output.view(output.size(0), -1)  # [B,1024]
        return output, self.indices

class FI(nn.Module):
    def __init__(self, feature_dim=3):
        super(FI, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.pooling = nn.MaxPool1d(1024,2048)
        self.FI1_a = nn.Parameter(Tensor(1), requires_grad=True)
        self.FI1_b = nn.Parameter(Tensor(1), requires_grad=True)
        self.FI2_a = nn.Parameter(Tensor(1), requires_grad=True)
        self.FI2_b = nn.Parameter(Tensor(1), requires_grad=True)

    def forward(self, x,y_1,y_2):
        x=x.permute(0, 2, 1)   # [B,N,feature_dim]->[B,feature_dim,N]
        output = self.relu(self.conv1(x))  # [B,64,N]
        output = self.relu(self.conv2(output))  # [B,64,N]
        output = self.relu(self.conv3(output+y_1))  # [B,64,N]
        output = self.relu(self.conv4(output))  # [B,128,N]
        output = self.relu(self.conv5(output+y_2))  # [B,1024,N]
        output = self.pooling(output)    # [B,1024,1]
        output = output.view(output.size(0), -1)  # [B,1024]
        return output
    
    def first_forward(self,x,y):
        x=x.permute(0, 2, 1)   # [B,N,feature_dim]->[B,feature_dim,N]
        y=y.permute(0, 2, 1)   # [B,N,feature_dim]->[B,feature_dim,N]
        output_x = self.relu(self.conv1(x))  # [B,64,N]
        output_x_part1 = self.conv2(output_x) # [B,64,N]
        
        output_y = self.relu(self.conv1(y))  # [B,64,N]
        output_y_part1 = self.conv2(output_y)  # [B,64,N]

        output_x_y = self.FI1_a*output_x_part1+self.FI1_b*output_y_part1

        output_x = self.relu(self.conv3(output_x_y))  # [B,64,N]
        output_y = self.relu(self.conv3(output_x_y))  # [B,64,N]

        output_x_part2 = self.relu(self.conv4(output_x))  # [B,128,N]
        output_y_part2 = self.relu(self.conv4(output_y))  # [B,128,N]

        output_x_y = self.FI2_a*output_x_part2+self.FI2_b*output_y_part2

        output_x = self.relu(self.conv5(output_x_y))  # [B,1024,N]
        output_y = self.relu(self.conv5(output_x_y))  # [B,1024,N]
        output_x = self.pooling(output_x)    # [B,1024,1]
        output_y = self.pooling(output_y)    # [B,1024,1]
        output_x = output_x.view(output_x.size(0), -1)  # [B,1024]
        output_y = output_y.view(output_y.size(0), -1)  # [B,1024]

        return output_x,output_y,output_x_part1,output_x_part2

class Decoder_FC(nn.Module):
    def __init__(self, feature_dim=3):
        super(Decoder_FC, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 2048 * feature_dim)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        output = self.relu(self.bn1(self.fc1(x)))
        output = self.relu(self.bn2(self.fc2(output)))
        output = self.relu(self.bn3(self.fc3(output)))
        output = self.th(self.fc4(output))
        output = output.view(batchsize, 3, 2048).transpose(1, 2).contiguous()  # [B,N,feature_dim]
        return output

class Decoder_Unconv(nn.Module):
    def __init__(self, feature_dim=3,encoder=None):
        super(Decoder_Unconv, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.encoder=encoder
        #使用二维卷积重建
        # self.unconv1 = nn.ConvTranspose2d(512, 512, kernel_size=[2, 2], stride=[2, 2])
        # self.unconv2 = nn.ConvTranspose2d(512, 256, kernel_size=[3, 3], stride=[1, 1])
        # self.unconv3 = nn.ConvTranspose2d(256, 256, kernel_size=[4, 5], stride=[2, 3])
        # self.unconv4 = nn.ConvTranspose2d(256, 128, kernel_size=[5, 7], stride=[3, 3])
        # self.unconv5 = nn.ConvTranspose2d(128, 3, kernel_size=[1, 1], stride=[1, 1])
        #使用一维卷积重建
        self.unpooling=nn.MaxUnpool1d(2048)
        self.unconv1 = nn.ConvTranspose1d(1024, 128,1)
        self.unconv2 = nn.ConvTranspose1d(128, 64,1)
        self.unconv3 = nn.ConvTranspose1d(64, 64, 1)
        self.unconv4 = nn.ConvTranspose1d(64, 64, 1)
        self.unconv5 = nn.ConvTranspose1d(64, feature_dim, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        output = self.relu(self.fc1(x))  # [B,1024]
        #使用二维重建的reshape
        # output = output.view(batch_size, -1, 1, 2)
        #使用一维重建的reshape
        output=output.view(batch_size,1024,1)
        # output=self.unpooling(output,indices)

        output = self.relu(self.unconv1(output))
        output = self.relu(self.unconv2(output))
        output = self.relu(self.unconv3(output))
        output = self.relu(self.unconv4(output))
        output = self.unconv5(output).transpose(1, 2).contiguous()   # [B,N,feature_dim]
        return output

class Decoder_Hierarchical_FC(nn.Module):
    def __init__(self,feature_dim=3):
        super(Decoder_Hierarchical_FC, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.pc1_feat=nn.Linear(512,64*256)
        self.bn3=nn.BatchNorm1d(64*256)

        self.pc1_xyz=nn.Linear(512,64*feature_dim)


        self.pc2=nn.Conv1d(256,256,1)
        self.pc2_xyz=nn.Conv1d(256,32*3,1)

    def forward(self, x):
        batch_size=x.size(0)
        output=self.relu(self.bn1(self.fc1(x)))
        output=self.relu(self.bn2(self.fc2(output)))

        pc1_feat=self.relu(self.bn3(self.pc1_feat(output)))
        pc1_xyz=self.pc1_xyz(output)
        pc1_feat=pc1_feat.view(batch_size,256,64)
        pc1_xyz=pc1_xyz.view(batch_size,3,64)   #[B,3,64]
        pc2=self.relu(self.pc2(pc1_feat))
        pc2_xyz=self.pc2_xyz(pc2)
        pc2_xyz=pc2_xyz.view(batch_size,3,32,64) #[B, 3, 32, 64]
        pc1_xyz=pc1_xyz.unsqueeze(2)  #[B, 3, 1, 64]
        pc2_xyz=pc1_xyz+pc2_xyz
        pc_xyz=pc2_xyz.view(batch_size,2048,3)
        return pc_xyz


