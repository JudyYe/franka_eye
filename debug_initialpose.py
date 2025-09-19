import imageio
import numpy as np
import cv2
import pickle


def load_initial_pose(initial_pose_file):
    with open(initial_pose_file, 'rb') as f:
        data = pickle.load(f)
    
    imageio.imwrite('realsense_output/initial_pose_rgb.jpg', data['rgb'])

    # show rgb, depth, and mask in different windows
    cv2.imshow('initial_pose_rgb', data['rgb'])
    cv2.imshow('initial_pose_depth', data['depth'])
    print(data['mask'].dtype, data['mask'].shape)
    cv2.imshow('initial_pose_mask', data['mask'].astype(np.uint8)*255)

    print(data['K'], data['rgb'].shape, data['current_pose'], data['depth'].shape)
    print(data['depth'].min(), data['depth'].max())

    # vis depth as colormap
    cv2.imshow('initial_pose_depth_colormap', cv2.applyColorMap((data['depth']*255).astype(np.uint8), cv2.COLORMAP_JET))
    cv2.waitKey(0)
    return data

if __name__ == '__main__':
    initial_pose_file = 'realsense_output/initial_pose/initial_pose.pkl'
    data = load_initial_pose(initial_pose_file)
    print(data)