import open3d as o3d
import numpy as np
import torch
import math, sys
sys.path.append("..")
from open3d.open3d.geometry import PointCloud
from open3d.open3d.utility import Vector3dVector


class FGR:
    def __init__(self, threshold=0.1, max_iteration=1):
        self.threshold = threshold
        self.criteria = o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=threshold, iteration_number=max_iteration)

    # Preprocess template, source point clouds.
    def preprocess(self, template, source):
        if self.is_tensor: template, source = template.detach().cpu().numpy(), source.detach().cpu().numpy()  # Convert to ndarray if tensors.

        if len(template.shape) > 2:  # Reduce dimension to [N, 3]
            template, source = template[0], source[0]

        # Find mean of template & source.
        self.template_mean = np.mean(template, axis=0, keepdims=True)
        self.source_mean = np.mean(source, axis=0, keepdims=True)

        # Convert to open3d point clouds.
        template_ = o3d.geometry.PointCloud()
        source_ = o3d.geometry.PointCloud()

        # Subtract respective mean from each point cloud.
        template_.points = o3d.utility.Vector3dVector(template - self.template_mean)
        source_.points = o3d.utility.Vector3dVector(source - self.source_mean)
        return template_, source_

    # Postprocess on transformation matrix.
    def postprocess(self, res):
        # Way to deal with mean substraction
        # Pt = R*Ps + t 								original data (1)
        # Pt - Ptm = R'*[Ps - Psm] + t' 				mean subtracted from template and source.
        # Pt = R'*Ps + t' - R'*Psm + Ptm 				rearrange the equation (2)
        # From eq. 1 and eq. 2,
        # R = R' 	&	t = t' - R'*Psm + Ptm			(3)

        est_R = np.array(res.transformation[0:3, 0:3])  # ICP's rotation matrix (source -> template)
        t_ = np.array(res.transformation[0:3, 3]).reshape(1, -1)  # ICP's translation vector (source -> template)
        est_T = np.array(res.transformation)  # ICP's transformation matrix (source -> template)
        est_t = np.matmul(est_R, -self.source_mean.T).T + t_ + self.template_mean[
            0]  # update predicted translation according to eq. 3
        est_T[0:3, 3] = est_t
        return est_R, est_t, est_T

    # Convert result to pytorch tensors.
    @staticmethod
    def convert2tensor(result):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        result['est_R'] = torch.tensor(result['est_R']).to(device).float().view(-1, 3,
                                                                                3)  # Rotation matrix [B, 3, 3] (source -> template)
        result['est_t'] = torch.tensor(result['est_t']).to(device).float().view(-1, 1,
                                                                                3)  # Translation vector [B, 1, 3] (source -> template)
        result['est_T'] = torch.tensor(result['est_T']).to(device).float().view(-1, 4,
                                                                                4)  # Transformation matrix [B, 4, 4] (source -> template)
        return result

    # icp registration.
    def __call__(self, sources, templates):
        if sources.shape[1] == 3:
            sources = sources.permute(0, 2, 1)
        if templates.shape[1] == 3:
            templates = templates.permute(0, 2, 1)

        est_Ts = []
        for source, template in zip(sources, templates):
            self.is_tensor = torch.is_tensor(sources)
            # compute normal
            source_ = PointCloud()
            template_ = PointCloud()
            source_.points = Vector3dVector(source)
            template_.points = Vector3dVector(template)
            source_.estimate_normals()
            template_.estimate_normals()

            source_fea = o3d.registration.compute_fpfh_feature(source_, o3d.geometry.KDTreeSearchParamKNN(knn=20))
            template_fea = o3d.registration.compute_fpfh_feature(template_, o3d.geometry.KDTreeSearchParamKNN(knn=20))

            template, source = self.preprocess(template, source)
            res = o3d.registration.registration_fast_based_on_feature_matching(source,
                                    template, source_fea, template_fea, self.criteria)  # FGR registration in open3d.
            est_R, est_t, est_T = self.postprocess(res)
            est_Ts.append(est_T)

        est_Ts = torch.Tensor(est_Ts)
        # R, t
        return est_Ts[:, 0:3, 0:3], est_Ts[:, 0:3, 3]