#dict(type='Resize', scale=(1242, 375), keep_ratio=True),
#640*360

# conda activate openmmlab

import logging
import os
import cv2

from mmengine.logging import print_log

from mmdet3d.apis import MonoDet3DInferencer


def Initialize():
    model = 'configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py'
    weights = 'pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth'
    device = 'cuda:0'

    # Check for display device
    if os.environ.get('DISPLAY') is None and show:
        print_log('Display device not found. `--show` is forced to False',
                  logger='current', level=logging.WARNING)
        show = False

    # Initialize inferencer
    inferencer = MonoDet3DInferencer(model=model, weights=weights, device=device)

    return inferencer
    #################################################################

def Inference(inferencer, show):

    # Directly assign values to variables
    img = 'demo/data/kitti/000008.png'
    infos = 'demo/data/kitti/000008.pkl'
    cam_type = 'CAM2'
    out_dir = 'outputs'
    show = True
    wait_time = -1
    no_save_vis = False
    no_save_pred = False
    print_result = True
    #pred_score_thr = 0.3 ### this show very very buggy result
    pred_score_thr = 5 ## but it's good i found this number by try and faild

    inputs = dict(img=img, infos=infos)

    # Call inferencer with arguments
    inferencer(
        inputs=inputs,
        #device=device,
        cam_type=cam_type,
        pred_score_thr=pred_score_thr,
        out_dir=out_dir,
        #show=show,
        show=False,
        wait_time=wait_time,
        no_save_vis=no_save_vis,
        no_save_pred=no_save_pred,
        print_result=print_result
    )

    if out_dir != '' and not (no_save_vis and no_save_pred):
        print_log(f'results have been saved at {out_dir}', logger='current')


def main():
    inferencer= Initialize()
    #Inference(inferencer,True)

    # Initialize video capture object
    video_path ="/home/user1/Desktop/Pirooz testing/20240222_104259.mp4" # wideq
    cap = cv2.VideoCapture(video_path)

    # Get the original width and height of the video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new width and height while preserving aspect ratio
    target_width = 640
    target_height = int(original_height * target_width / original_width)



    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:

            frame = cv2.resize(frame, (target_width, target_height))
            
            output_path = os.path.join("demo/data/kitti", f"000008.png")
            cv2.imwrite(output_path, frame)

            Inference(inferencer,True)
            
            
            cv2.imshow("Original Frame",cv2.imread(os.path.join("outputs/vis_camera/CAM2", f"000008.png")))
            
            # Resize the frame
            #frame = cv2.resize(frame, (target_width, target_height))

            # Display the original frame
            #cv2.imshow("Original Frame", frame)

            # Resize the window
            #cv2.resizeWindow("iPilot v1", 640,480)

            # Break the loop if 'q' is pressed 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break 


if __name__ == '__main__':
    main()
