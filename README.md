
we should download 
https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth

and copy to root folder 

/////////////////////////////////////////////////////////////////////////////////


i used gpt and 
from here 
[https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth](https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html#monocular-3d-demo)

I read this file demo/mono_det_demo.py and try to run it without passing parameters like image address in a py file in body of it
and we are try to change it to run many images in a loop  


and based on this 
[https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html](https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html#monocular-3d-demo)



this is sample to run

python demo/mono_det_demo.py demo/data/kitti/000008.png demo/data/kitti/000008.pkl  configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth  --show --cam-type CAM2 --print-result

