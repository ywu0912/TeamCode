import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

from model.Autoencoder import FI, Decoder_FC
from operations.transform_functions import PCRNetTransform
from operations.dual import dual_quat_to_extrinsic
from .regressionBranch import  RegressionBranch

class SPCNet(nn.Module):
    def __init__(self, encoder=FI(), decoder=Decoder_FC()):
        super(SPCNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regressionBranch = RegressionBranch()

        # self.fc1 = nn.Linear(2048, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(512, 512)
        # self.bn3 = nn.BatchNorm1d(512)
        # self.fc4 = nn.Linear(512, 256)
        # self.bn4 = nn.BatchNorm1d(256)
        # # self.fc5 = nn.Linear(256, 8)
        # self.fc5 = nn.Linear(256, 8)

    def forward(self, template, source, max_iteration=2):
        # 估计的旋转矩阵
        est_R = ops.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        # 估计的平移向量
        est_t = ops.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
       
        template_feature,source_feature,template_feature_one,template_feature_two= self.encoder.first_forward(template,source)  # [B,1024]

        # print("template_feature",template_feature)
        # numpy.savetxt("heat.txt", template_feature.detach().cpu().numpy())

        retemplate = self.decoder(template_feature)  # [B,N,3]
        
        for i in range(max_iteration):
            if i!=0:
                source_feature = self.encoder(source,template_feature_one,template_feature_two)  # [B,1024]
            batch_size = source_feature.shape[0]
            self.feature_difference = source_feature - template_feature

            pose_7d = self.regressionBranch(template_feature,source_feature)
            pose_7d = PCRNetTransform.create_pose_7d(pose_7d) # [B,7]
            identity = ops.eye(3).to(source).view(1,3,3).expand(batch_size,3,3).contiguous()
            est_R_temp = PCRNetTransform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1) # [B,3,3]
            est_t_temp = PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3) # [B,1,3]
            est_t = ops.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
            est_R = ops.bmm(est_R_temp, est_R)
            source = PCRNetTransform.quaternion_transform(source, pose_7d)   
            resource  = self.decoder(source_feature)

        result = {
            'est_R': est_R,  # 估计的旋转矩阵[B,3,3]
            'est_t': est_t,  # 估计的平移向量[B,1,3]
            'est_T': PCRNetTransform.convert2transformation(est_R, est_t),   #估计的变换矩阵[B,4,4]
            'feature_difference': self.feature_difference,    #特征差异  [B,1024]
            'transformed_source': source,    #经过迭代后的源点云   [B,1024,3]
            'retemplate': retemplate,  #重建后的模版点云   [B,N,3]
            'resource': resource    #重建后的源点云  [B,N,3]
        }
        return result

if __name__ == '__main__':
    template, source = ops.rand(10, 1024, 3), ops.rand(10, 1024, 3)
    model = SPCNet(FI(), Decoder_FC())
    result = model(template,source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
    print(result['est_T'].shape)
    print(result['feature_difference'].shape)
    print(result['transformed_source'].shape)
    print(result['retemplate'].shape)
    print(result['resource'].shape)
