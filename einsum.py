import torch
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable
from utils.geometry import perspective_projection

device = 'cpu'
criterion_keypoints = nn.MSELoss(reduction='none').to(device)
def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d):
    loss = criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss

R = torch.eye(3).unsqueeze(0).expand(1, -1, -1)
R[0,0,0] = 2
rotation = Variable(R, requires_grad=True)
camera_translation = Variable(torch.zeros(1, 3),requires_grad=True)

points = torch.randn(17,3)
# points = Variable(torch.randn(1,17,3), requires_grad=True)
npoints = perspective_projection(points[None], rotation, camera_translation, None, None)


# # This is OK
# npoints = torch.einsum('bij,bkj->bki', rotation, points)
# loss = keypoint_loss(points, npoints)
# import ipdb; ipdb.set_trace()
loss = keypoint_loss(points[:,:2], npoints)
loss.backward()
print(loss)
print(rotation.grad)
print(camera_translation.grad)