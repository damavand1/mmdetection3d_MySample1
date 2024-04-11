import logging
import os

from mmengine.logging import print_log

from mmdet3d.apis import MonoDet3DInferencer

def main():
    # Directly assign values to variables
    img = 'demo/data/kitti/000008.png'
    infos = 'demo/data/kitti/000008.pkl'
    model = 'configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py'
    weights = 'pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth'
    device = 'cuda:0'
    cam_type = 'CAM2'
    pred_score_thr = 0.3
    out_dir = 'outputs'
    show = True
    wait_time = -1
    no_save_vis = False
    no_save_pred = False
    print_result = True

    inputs = dict(img=img, infos=infos)

    # Check for display device
    if os.environ.get('DISPLAY') is None and show:
        print_log('Display device not found. `--show` is forced to False',
                  logger='current', level=logging.WARNING)
        show = False

    # Initialize inferencer
    inferencer = MonoDet3DInferencer(model=model, weights=weights, device=device)

    # Call inferencer with arguments
    inferencer(
        inputs=inputs,
        #device=device,
        cam_type=cam_type,
        pred_score_thr=pred_score_thr,
        out_dir=out_dir,
        show=show,
        wait_time=wait_time,
        no_save_vis=no_save_vis,
        no_save_pred=no_save_pred,
        print_result=print_result
    )

    if out_dir != '' and not (no_save_vis and no_save_pred):
        print_log(f'results have been saved at {out_dir}', logger='current')


if __name__ == '__main__':
    main()
