# use grounded sam 2 to find the mask, instead of user selection
# monitor pose score, if it's below a threshold, re initialize pose


import os
import os.path as osp
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
import logging
import torch
import nvdiffrast.torch as dr
from FoundationPose.estimater import FoundationPose
from FoundationPose.learning.training.predict_score import ScorePredictor
from FoundationPose.learning.training.predict_pose_refine import PoseRefinePredictor
from FoundationPose.Utils import draw_posed_3d_box, draw_xyz_axis
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from torchvision import transforms

import sys
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-SAM-2'))
print(f"Adding {path_to_add} to sys.path")
sys.path.insert(0, path_to_add)
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

class Text2MaskPredictor:
    """Text-to-mask predictor using GroundingDINO + SAM2."""
    
    def __init__(self, device='cuda:0', text_prompt='yellow bottle', 
                 box_threshold=0.35, text_threshold=0.25):
        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
    
        # Initialize SAM2
        
        # Store functions as instance variables for later use
        self.predict_fn = predict
        self.box_convert = box_convert
        
        # Build SAM2 model
        sam2_model = build_sam2_hf('facebook/sam2-hiera-large', device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # Build GroundingDINO model
        grounding_config = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounding_checkpoint = "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        
        self.grounding_model = load_model(
            model_config_path=grounding_config,
            model_checkpoint_path=grounding_checkpoint,
            device=device
        )
        
        # Setup autocast for better performance
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logging.info("Text2MaskPredictor initialized successfully")
        self.initialized = True

    def predict(self, image, text_prompt=None):
        """
        Predict mask from text prompt using GroundingDINO + SAM2.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            text_prompt (str): Text description of object to segment
            
        Returns:
            np.ndarray: Boolean mask of the detected object
        """
        if not self.initialized:
            logging.error("Text2MaskPredictor not initialized")
            return None
            
        if text_prompt is None:
            text_prompt = self.text_prompt
            
        # Convert BGR to RGB for GroundingDINO
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Load image for GroundingDINO
        image_source, image_tensor = self.load_image_from_array(rgb_image)
        
        # Set image for SAM2
        self.sam2_predictor.set_image(image_source)
        
        # Predict bounding boxes using GroundingDINO
        boxes, confidences, labels = self.predict_fn(
            model=self.grounding_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            logging.warning(f"No objects detected for prompt: {text_prompt}")
            return None
        
        # Convert boxes to SAM2 format
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = self.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # Enable autocast for better performance
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # Predict masks using SAM2
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        
        # Convert to numpy and select best mask
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Select the mask with highest confidence
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx].astype(bool)
        
        logging.info(f"Generated mask for '{text_prompt}' with confidence {scores[best_mask_idx]:.3f}")
        return best_mask
        

    @staticmethod
    def load_image_from_array(image_array):
        """Load image from numpy array for GroundingDINO compatibility."""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(pil_image)
        
        return image_array, image_tensor

def setup_foundation_pose(mesh_file, debug=True, debug_dir='./realsense_debug'):
    """
    Initialize FoundationPose with the given mesh file.
    
    Args:
        mesh_file (str): Path to the object mesh file (.obj)
        debug (bool): Enable debug mode
        debug_dir (str): Directory for debug outputs
    
    Returns:
        FoundationPose: Initialized pose estimator
        dict: Object metadata (bbox, to_origin, etc.)
    """
    # Load mesh
    mesh = trimesh.load(mesh_file)
    
    # Create debug directory
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f'{debug_dir}/poses', exist_ok=True)
    os.makedirs(f'{debug_dir}/visualizations', exist_ok=True)
    
    # Initialize FoundationPose components
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    # Create FoundationPose estimator
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx
    )
    
    # Get object metadata
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    
    metadata = {
        'mesh': mesh,
        'bbox': bbox,
        'to_origin': to_origin,
        'extents': extents
    }
    
    logging.info("FoundationPose initialization completed")
    return estimator, metadata

def get_camera_intrinsics(intr):
    """
    Convert RealSense intrinsics to camera matrix K.
    
    Args:
        intr: RealSense intrinsics object
    
    Returns:
        np.ndarray: 3x3 camera matrix K
    """
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    return K

class InteractiveMaskSelector:
    """Interactive mask selection using mouse clicks and SAM2."""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.sam2_predictor = None
        self.current_image = None
        self.current_mask = None
        self.click_points = []
        self.click_labels = []  # 1 for positive, 0 for negative
        self.window_name = "Interactive Mask Selection"
        
    def setup_sam2(self):
        """Initialize SAM2 image predictor."""
        sam2_model = build_sam2_hf('facebook/sam2-hiera-large', device=self.device)

        predictor = SAM2ImagePredictor(sam2_model)        
        self.sam2_predictor = predictor
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for mask selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - positive point
            self.click_points.append([x, y])
            self.click_labels.append(1)
            logging.info(f"Added positive point at ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - negative point
            self.click_points.append([x, y])
            self.click_labels.append(0)
            logging.info(f"Added negative point at ({x}, {y})")
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click - clear all points
            self.click_points = []
            self.click_labels = []
            logging.info("Cleared all points")
    
    def generate_mask_from_clicks(self, image):
        """Generate mask using SAM2 based on user clicks."""
        if not self.sam2_predictor or len(self.click_points) == 0:
            return None
        
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Prepare points and labels for SAM2
            points = np.array(self.click_points)
            labels = np.array(self.click_labels)
            
            # Use SAM2 image predictor to generate mask
            # Set the image first
            self.sam2_predictor.set_image(rgb_image)
            
            # Predict mask from points
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
            print('mask size:', masks.shape, type(masks), masks.max())
            
            return masks[0]  # Return the first (and only) mask
            
        except Exception as e:
            logging.error(f"Failed to generate mask from clicks: {e}")
            return None
    
    def visualize_clicks_and_mask(self, image, mask=None):
        """Visualize current clicks and mask on the image."""
        vis_image = image.copy()
        
        # Draw click points
        for i, (point, label) in enumerate(zip(self.click_points, self.click_labels)):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, red for negative
            cv2.circle(vis_image, tuple(point), 5, color, -1)
            cv2.putText(vis_image, f"{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Overlay mask if available
        if mask is not None:
            # Create colored mask overlay
            mask_colored = np.zeros_like(vis_image)
            mask_colored[mask.astype(bool)] = [0, 255, 255]  # Yellow for mask
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)
        
        # Add instructions
        cv2.putText(vis_image, "Left click: positive point", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, "Right click: negative point", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, "Middle click: clear points", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, "Press 'g' to generate mask", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, "Press 'y' to confirm, 'n' to retry", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def interactive_mask_selection(self, image):
        """Main interactive mask selection loop."""
        self.current_image = image.copy()
        self.click_points = []
        self.click_labels = []
        self.current_mask = None
        
        # Setup OpenCV window and mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        logging.info("Starting interactive mask selection...")
        logging.info("Instructions: Left click for positive points, right click for negative points")
        logging.info("Press 'g' to generate mask, 'y' to confirm, 'n' to retry, 'q' to quit")
        
        while True:
            # Visualize current state
            vis_image = self.visualize_clicks_and_mask(self.current_image, self.current_mask)
            cv2.imshow(self.window_name, vis_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('g'):
                # Generate mask
                if len(self.click_points) > 0:
                    mask = self.generate_mask_from_clicks(self.current_image)
                    if mask is not None:
                        self.current_mask = mask
                        logging.info("Mask generated successfully")
                    else:
                        logging.warning("Failed to generate mask")
                else:
                    logging.warning("No click points available. Please click on the object first.")
            
            elif key == ord('y'):
                # Confirm and return mask
                if self.current_mask is not None:
                    cv2.destroyWindow(self.window_name)
                    logging.info("Mask confirmed by user")
                    return self.current_mask.astype(bool)
                else:
                    logging.warning("No mask available. Please generate a mask first.")
            
            elif key == ord('n'):
                # Retry - clear everything
                self.click_points = []
                self.click_labels = []
                self.current_mask = None
                logging.info("Cleared all points and mask. Please try again.")
            
            elif key == ord('q'):
                # Quit
                cv2.destroyWindow(self.window_name)
                logging.info("Interactive mask selection cancelled by user")
                return None
        
        cv2.destroyWindow(self.window_name)
        return None

def pose_initialization_waiting_stage(estimator, metadata, K, color, depth, mask_selector, mode='text'):
    """
    Waiting stage for pose initialization with configurable mask selection mode.
    
    Args:
        estimator: FoundationPose estimator
        metadata: Object metadata (bbox, to_origin, etc.)
        K: Camera intrinsics matrix
        color: Color image
        depth: Depth image
        mask_selector: Text2MaskPredictor or InteractiveMaskSelector instance
        mode: 'text' for text-based prediction, 'interactive' for manual selection
    
    Returns:
        tuple: (success, pose, mask) - success flag, initial pose, and object mask
    """
    logging.info(f"Starting pose initialization waiting stage in {mode} mode...")
    
    # Step 1: Present current image to user
    if mode == 'text':
        logging.info("Presenting current image for text-based mask prediction...")
        temp_img = color.copy()
        cv2.putText(temp_img, "Generating mask from text prompt...", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Current Image - Generating Mask", temp_img)
        cv2.waitKey(1000)  # Show for 1 second
        
        # Step 2: Generate mask using text prompt
        logging.info("Starting text-based mask prediction...")
        object_mask = mask_selector.predict(color)
        
        if object_mask is None:
            logging.warning("Text-based mask prediction failed")
            return False, None, None
            
    elif mode == 'interactive':
        logging.info("Presenting current image for interactive mask selection...")
        temp_img = color.copy()
        cv2.putText(temp_img, "Starting interactive mask selection...", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Current Image - Click to Select Object", temp_img)
        cv2.waitKey(1000)  # Show for 1 second
        
        # Step 2: Interactive mask selection using SAM2
        logging.info("Starting interactive mask selection...")
        object_mask = mask_selector.interactive_mask_selection(color)
        
        if object_mask is None:
            logging.warning("Interactive mask selection cancelled by user")
            return False, None, None
    else:
        logging.error(f"Unknown mode: {mode}. Must be 'text' or 'interactive'")
        return False, None, None
    
    # Step 3: Initialize pose with the selected mask
    logging.info("Initializing pose with selected mask...")
    try:
        # Convert BGR to RGB for FoundationPose
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        # Initialize pose
        initial_pose = estimator.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=object_mask,
            iteration=5
        )
        
        if initial_pose is None:
            logging.error("Pose initialization failed")
            return False, None, object_mask
        
        logging.info("Pose initialization successful")
        
        # Step 4: Show pose visualization for user confirmation
        vis_img = draw_pose_visualization(
            color, initial_pose, K, metadata['bbox'], metadata['to_origin']
        )        
        # save all, current_pose, RGBD, K, bbox, mask
        fname = osp.join(args.output_dir, 'initial_pose', 'initial_pose.pkl')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump({'current_pose': initial_pose, 'rgb': rgb, 'depth': depth, 'K': K, 'bbox': metadata['bbox'], 'mask': object_mask}, f)
        # save vis_img 
        cv2.imwrite(osp.join(args.output_dir, 'initial_pose', 'initial_pose.jpg'), vis_img)
        print(f"Saved initial pose to {fname}")

        # Add confirmation text
        cv2.putText(vis_img, "Initial Pose - Press 'y' to confirm, 'n' to retry", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show pose center
        center_3d = initial_pose[:3, 3]
        cv2.putText(vis_img, f"Position: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}]", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Pose Initialization - Confirm or Retry", vis_img)
        
        # Wait for user confirmation
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                cv2.destroyWindow("Pose Initialization - Confirm or Retry")
                logging.info("Pose initialization confirmed by user")
                return True, initial_pose, object_mask
            elif key == ord('n'):
                cv2.destroyWindow("Pose Initialization - Confirm or Retry")
                logging.info("Pose initialization rejected by user - retrying...")
                return False, None, object_mask
            elif key == 27:  # ESC
                cv2.destroyWindow("Pose Initialization - Confirm or Retry")
                logging.info("Pose initialization cancelled by user")
                return False, None, object_mask
        
    except Exception as e:
        logging.error(f"Error during pose initialization: {e}")
        return False, None, object_mask

def draw_pose_visualization(color, pose, K, bbox, to_origin, scale=0.1):
    """
    Draw pose visualization on the color image using FoundationPose visualization functions.
    
    Args:
        color (np.ndarray): Color image
        pose (np.ndarray): 4x4 pose matrix
        K (np.ndarray): Camera intrinsics
        bbox (np.ndarray): Object bounding box
        to_origin (np.ndarray): Transform to origin
        scale (float): Scale for visualization
    
    Returns:
        np.ndarray: Image with pose visualization
    """
    vis_img = color.copy()
    
    # Transform pose to centered mesh coordinates
    center_pose = pose @ np.linalg.inv(to_origin)
    
    # Use FoundationPose visualization functions
    # Draw 3D bounding box
    vis_img = draw_posed_3d_box(K, img=vis_img, ob_in_cam=center_pose, bbox=bbox)
    
    # Draw coordinate axes
    vis_img = draw_xyz_axis(vis_img, ob_in_cam=center_pose, scale=scale, K=K, 
                           thickness=3, transparency=0, is_input_rgb=True)
    
    return vis_img

def stream_pose_estimation(mesh_file, output_dir='./realsense_output', 
                          est_refine_iter=5, track_refine_iter=2, 
                          save_poses=True, save_visualizations=True, 
                          show_preview=True, device='cuda:0', text_prompt='yellow bottle', 
                          mode='text'):
    """
    Main function to stream RGBD from RealSense and estimate object poses.
    
    Args:
        mesh_file (str): Path to object mesh file
        output_dir (str): Directory to save outputs
        est_refine_iter (int): Iterations for initial pose estimation
        track_refine_iter (int): Iterations for pose tracking
        save_poses (bool): Whether to save pose files
        save_visualizations (bool): Whether to save visualization images
        show_preview (bool): Whether to show real-time preview
        device (str): Device for SAM2 (cuda:0, cpu, etc.)
        text_prompt (str): Text prompt for object detection (text mode only)
        mode (str): Mask selection mode - 'text' for automatic, 'interactive' for manual
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize FoundationPose
    estimator, metadata = setup_foundation_pose(mesh_file, debug=2, debug_dir=output_dir)
    
    # Initialize mask selector based on mode
    if mode == 'text':
        mask_selector = Text2MaskPredictor(device=device, text_prompt=text_prompt)
        if not mask_selector.initialized:
            logging.error("Failed to initialize Text2MaskPredictor. Cannot proceed without mask prediction.")
            return
    elif mode == 'interactive':
        mask_selector = InteractiveMaskSelector(device=device)
        if not mask_selector.setup_sam2():
            logging.error("Failed to initialize SAM2 for interactive mode. Falling back to text mode.")
            mode = 'text'
            mask_selector = Text2MaskPredictor(device=device, text_prompt=text_prompt)
            if not mask_selector.initialized:
                logging.error("Failed to initialize Text2MaskPredictor. Cannot proceed.")
                return
    else:
        logging.error(f"Unknown mode: {mode}. Must be 'text' or 'interactive'")
        return
    
    # Setup RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = get_camera_intrinsics(intr)
    
    print(f"Camera intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, ppx={intr.ppx:.2f}, ppy={intr.ppy:.2f}")
    print(f"Depth scale: {depth_scale}")
    
    # Tracking variables
    frame_count = 0
    pose_initialized = False
    current_pose = None
    
    try:
        while True:
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            
            # Convert BGR to RGB for FoundationPose
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            
            if not pose_initialized:
                # Show waiting message and wait for user to press key to start initialization
                vis_img = color.copy()
                cv2.putText(vis_img, "Press 'i' to initialize pose", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(vis_img, "Press ESC to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(vis_img, f"Frame: {frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if show_preview:
                    cv2.imshow("RealSense Pose Estimation", vis_img)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('i'):
                        # User pressed 'i' - start pose initialization waiting stage
                        logging.info(f"User triggered pose initialization (frame {frame_count})")
                        
                        if mask_selector is not None:
                            # Use mask selection based on mode
                            print('depth minmax', depth.min(), depth.max())
                            
                            success, initial_pose, selected_mask = pose_initialization_waiting_stage(
                                estimator, metadata, K, color, depth.astype(np.float32) * depth_scale, mask_selector, mode
                            )
                            
                            if success and initial_pose is not None:
                                current_pose = initial_pose
                                pose_initialized = True
                                logging.info("Pose initialization completed successfully")
                            else:
                                logging.warning("Pose initialization failed or cancelled")
                                # Continue to next frame to retry
                                continue
                        else:
                            raise ValueError("Mask selector is not initialized")
                            
                    elif key == 27:  # ESC to quit
                        break
                else:
                    # If no preview, just wait a bit and continue
                    import time
                    time.sleep(0.1)
            else:
                # Pose tracking
                try:
                    current_pose = estimator.track_one(
                        rgb=rgb, 
                        depth=depth, 
                        K=K, 
                        iteration=track_refine_iter
                    )
                except Exception as e:
                    logging.warning(f"Pose tracking failed: {e}")
                    # Reset to initialization mode
                    pose_initialized = False
                    current_pose = None
            
            # Save pose if available
            if current_pose is not None and save_poses:
                pose_file = os.path.join(output_dir, 'poses', f'pose_{frame_count:06d}.txt')
                np.savetxt(pose_file, current_pose.reshape(4, 4))
            
            # Create visualization
            vis_img = color.copy()
            if current_pose is not None:
                vis_img = draw_pose_visualization(
                    vis_img, current_pose, K, metadata['bbox'], metadata['to_origin']
                )
                
                # Add pose info text
                center_3d = current_pose[:3, 3]
                cv2.putText(vis_img, f"Pos: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}]", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(vis_img, f"Frame: {frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(vis_img, "Tracking Active", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(vis_img, "Press 'i' to initialize pose", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_img, "Press ESC to quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save visualization if requested
            if save_visualizations and current_pose is not None:
                vis_file = os.path.join(output_dir, 'visualizations', f'vis_{frame_count:06d}.png')
                cv2.imwrite(vis_file, vis_img)
            
            # Show preview
            if show_preview:
                cv2.imshow("RealSense Pose Estimation", vis_img)
                if cv2.waitKey(1) == 27:  # ESC to quit
                    break
            
            frame_count += 1
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        pipe.stop()
        if show_preview:
            cv2.destroyAllWindows()
        logging.info(f"Streaming completed. Processed {frame_count} frames.")
        logging.info(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='RealSense FoundationPose Streaming')
    parser.add_argument('--mesh_file', type=str, 
                       default='./FoundationPose/demo_data/mustard0/mesh/textured_simple.obj',
                       help='Path to object mesh file')
    parser.add_argument('--output_dir', type=str, default='./realsense_output',
                       help='Output directory for poses and visualizations')
    parser.add_argument('--est_refine_iter', type=int, default=5,
                       help='Iterations for initial pose estimation')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                       help='Iterations for pose tracking')
    parser.add_argument('--no_save_poses', action='store_true',
                       help='Disable saving pose files')
    parser.add_argument('--no_save_vis', action='store_true',
                       help='Disable saving visualization images')
    parser.add_argument('--no_preview', action='store_true',
                       help='Disable real-time preview')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for SAM2 (cuda:0, cpu, etc.)')
    parser.add_argument('--text_prompt', type=str, default='yellow bottle',
                       help='Text prompt for object detection and segmentation')
    parser.add_argument('--mode', type=str, default='text', choices=['text', 'interactive'],
                       help='Mask selection mode: text (automatic) or interactive (manual)')
    
    args = parser.parse_args()
    
    stream_pose_estimation(
        mesh_file=args.mesh_file,
        output_dir=args.output_dir,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        save_poses=not args.no_save_poses,
        save_visualizations=not args.no_save_vis,
        show_preview=not args.no_preview,
        device=args.device,
        text_prompt=args.text_prompt,
        mode=args.mode
    )
