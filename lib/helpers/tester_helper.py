import os
import tqdm
import time
import shutil

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from lib.datasets.utils import compute_3d_box_cam, draw_projected_box3d

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path), f'{checkpoint_path}'
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)
        # Save plotted results
        #self.save_results_plot(results)
        self.save_results_plot_3d(results)


    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()
            
    def save_results_plot(self, results):
        """
        Save plots of detection results on images.
        Args:
            results: Dictionary of detections, keyed by image ID.
        """

        output_dir = os.path.join(self.output_dir, 'outputs', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        for img_id, detections in results.items():
            # Fetch the original image
            img = self.dataloader.dataset.get_image(img_id)
            if img is None:
                continue

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)

            for det in detections:
                class_name = self.class_name[int(det[0])]
                x_min, y_min, x_max, y_max = det[2:6]  # Extract 2D bounding box coordinates
                score = det[-1]  # Extract the detection score

                # Add bounding box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                # Add label
                label = f"{class_name} ({score:.2f})"
                ax.text(x_min, y_min - 5, label, color='yellow', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            plt.axis('off')
            output_path = os.path.join(output_dir, '{:06d}.png'.format(img_id))
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
    def save_results_plot_3d(self, results):
        """
        Save plots of detection results with 3D bounding boxes on images.
        Args:
            results: Dictionary of detections, keyed by image ID.
        """
        output_dir = os.path.join(self.output_dir, 'outputs', 'plots_3d')
        os.makedirs(output_dir, exist_ok=True)

        for img_id, detections in results.items():
            # Fetch the original image
            img = self.dataloader.dataset.get_image(img_id)
            if img is None:
                continue

            # Fetch the calibration data for the current image
            calib = self.dataloader.dataset.get_calib(img_id)  # Use the get_calib function with the image ID
            if calib is None:
                continue  # Skip if calibration is unavailable

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            
            # Set limits based on image size to prevent excessive canvas scaling
            width, height = img.size  # Get the width and height of the image
            ax.set_xlim(0, width)  
            ax.set_ylim(height, 0)  # Flip the y-axis for correct orientation
            
            for det in detections:
                class_name = self.class_name[int(det[0])]
                # Assuming det contains 3D bounding box [x_min, y_min, x_max, y_max, z_min, z_max]
                '''
                0- cls_id (int): The class ID of the detected object (e.g., car, pedestrian, etc.).
                1- alpha (float): Observation angle of the object in radians, relative to the camera's centerline.
                2- x_min (float): Left coordinate of the 2D bounding box in image space.
                3- y_min (float): Top coordinate of the 2D bounding box in image space.
                4- x_max (float): Right coordinate of the 2D bounding box in image space.
                5- y_max (float): Bottom coordinate of the 2D bounding box in image space.
                6- h (float): Height of the object in meters (3D dimensions).
                7- w (float): Width of the object in meters (3D dimensions).
                8- l (float): Length of the object in meters (3D dimensions).
                9- x (float): X-coordinate of the object's center in 3D space (left-right position).
                10- y (float): Y-coordinate of the object's center in 3D space (vertical position).
                11- z (float): Z-coordinate of the object's center in 3D space (depth from the camera).
                12- ry (float): Rotation angle of the object around the vertical axis (Y-axis) in radians.
                - score (float): Confidence score of the detection.       - x (float): X-coordinate of the object's center in 3D space (left-right position).
                - y (float): Y-coordinate of the object's center in 3D space (vertical position).
                - z (float): Z-coordinate of the object's center in 3D space (depth from the camera).
                - ry (float): Rotation angle of the object around the vertical axis (Y-axis) in radians.
                - score (float): Confidence score of the detection.
                '''
                x_min, y_min, x_max, y_max = det[2:6]  # Extract 2D bounding box coordinates
                h, w, l, x, y, z, ry = det[6:13]  # Extract 3D bounding box dimensions
                score = det[-1]  # Extract the detection score

                # Project 3D box to 2D plane (simplified approach)
                if calib is not None:
                    # Assuming calib.rect_to_img() can handle 3D points (x, y, z)
                    # Get the 8 corners of the 3D box (min/max x, y, z)
                    corners_3d = compute_3d_box_cam(h, w, l, x, y, z, ry)

                    # Project the 3D corners to the 2D image plane
                    corners_2d = calib.project_rect_to_image(corners_3d.T)

                    # Get the 2D bounding box from projected corners
                    x_min_proj, y_min_proj = np.min(corners_2d, axis=0)
                    x_max_proj, y_max_proj = np.max(corners_2d, axis=0)
                    
                    # Ensure coordinates are within the image bounds
                    x_min_proj = max(0, min(x_min_proj, width - 1))
                    y_min_proj = max(0, min(y_min_proj, height - 1))
                    x_max_proj = max(0, min(x_max_proj, width - 1))
                    y_max_proj = max(0, min(y_max_proj, height - 1))

                    # Add the 3D bounding box projection to the plot
                    rect = patches.Rectangle((x_min_proj, y_min_proj), x_max_proj - x_min_proj, y_max_proj - y_min_proj,
                                            linewidth=2, edgecolor='blue', facecolor='none', linestyle='dotted')
                    ax.add_patch(rect)
                    
                    draw_projected_box3d(img, corners_2d, color=(255, 0, 255), thickness=1)

                    # Add label for 3D detection
                    label = f"{class_name} ({score:.2f})"
                    ax.text(x_min_proj, y_min_proj - 5, label, color='cyan', fontsize=10,
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            output_path = os.path.join(output_dir, '{:06d}_3d.png'.format(img_id))
            img.save(output_path.replace(".png", "_3D.png"))
            plt.axis('off')
            plt.tight_layout(pad=0.5)  # Minimize excessive whitespace
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)


    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result
