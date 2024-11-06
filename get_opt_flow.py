import numpy as np
import torch
from submodules.third_party.raft import load_RAFT
import cv2 
import os
from torch.nn import functional as F

class OccMask(torch.nn.Module):
    def __init__(self, th=1):
        super(OccMask, self).__init__()
        self.th = th
        self.base_coord = None

    def init_grid(self, shape, device):
        H, W = shape
        hh, ww = torch.meshgrid(torch.arange(
            H).float(), torch.arange(W).float())
        coord = torch.zeros([1, H, W, 2])
        coord[0, ..., 0] = ww
        coord[0, ..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    @torch.no_grad()
    def get_oob_mask(self, base_coord, flow_1_2):
        target_range = base_coord + flow_1_2.permute([0, 2, 3, 1])
        oob_mask = (target_range[..., 0] < 0) | (target_range[..., 0] > self.W-1) | (
            target_range[..., 1] < 0) | (target_range[..., 1] > self.H-1)
        return ~oob_mask[:, None, ...]

    @torch.no_grad()
    def get_flow_inconsistency_tensor(self, base_coord, flow_1_2, flow_2_1):
        B, C, H, W = flow_1_2.shape
        sample_grids = base_coord + flow_1_2.permute([0, 2, 3, 1])
        sample_grids[..., 0] = sample_grids[..., 0] / (W - 1) / 2 -1
        sample_grids[..., 1] = sample_grids[..., 1] /(H - 1) / 2 -1
        # sample_grids -= 1
        sampled_flow = F.grid_sample(
            flow_2_1, sample_grids, align_corners=True)
        return torch.abs((sampled_flow+flow_1_2).sum(1, keepdim=True))

    def forward(self, flow_1_2, flow_2_1):
        B, _, H, W = flow_1_2.shape
        if self.base_coord is None:
            self.init_grid([H, W], device=flow_1_2.device)
        base_coord = self.base_coord.expand([B, -1, -1, -1])
        oob_mask = self.get_oob_mask(base_coord, flow_1_2)
        flow_inconsistency_tensor = self.get_flow_inconsistency_tensor(
            base_coord, flow_1_2, flow_2_1)
        valid_flow_mask = flow_inconsistency_tensor < self.th
        return valid_flow_mask*oob_mask

def save_optical_flow(flow, output_folder,name, mask=None):
    # Remove batch dimension and convert to numpy
    flow_np = flow.squeeze(0).cpu().numpy()  # Shape now (2, H, W)
    # Convert flow to HSV for visualization
    flow_hsv = np.zeros((flow_np.shape[1], flow_np.shape[2], 3), dtype=np.uint8)
    magnitude, angle = cv2.cartToPolar(flow_np[0], flow_np[1])
    flow_hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
    flow_hsv[..., 1] = 255  # Saturation is set to max
    flow_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

    # Convert HSV to RGB for saving as image
    flow_rgb = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)
    if mask is not None:
        print("masking is the key----------====================")
        flow_hsv = flow_hsv * mask.permute(1,2,0).cpu().numpy()

    # Save the optical flow image
    output_path = os.path.join(output_folder, name)
    cv2.imwrite(output_path, flow_rgb)
    print(f"Saved flow image at {output_path}")
    return flow_hsv

def save_mask(mask, output_folder, name):
    # Convert the mask to CPU and numpy format
    mask_np = mask.squeeze(0).cpu().numpy()  # Shape should now be (H, W)

    # Scale mask to 8-bit format (0 and 255) for saving
    mask_image = (mask_np * 255).astype(np.uint8)  # Converts True to 255 and False to 0

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the mask image
    output_path = os.path.join(output_folder, name)
    cv2.imwrite(output_path, mask_image)
    print(f"Saved mask image at {output_path}")

def get_flow(pair_imgs,sintel_ckpt=True): #TODO: test with gt flow
        print('precomputing flow...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_valid_flow_mask = OccMask(th=3.0)
        # pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]

        flow_net = load_RAFT() if sintel_ckpt else load_RAFT("submodules/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        flow_net = flow_net.to(device)
        flow_net.eval()
        output_folder = "optical_flow"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with torch.no_grad():
            # chunk_size = 12
            flow_ij = []
            flow_ji = []
            num_pairs = len(pair_imgs)
            for i in range(2):
                print(i)
                imgs_ij = [torch.tensor(pair_imgs[i][0]).float().to(device),
                        torch.tensor(pair_imgs[i][1]).float().to(device)]
                flow = flow_net(imgs_ij[0] * 255, imgs_ij[1] * 255, iters=20, test_mode=True)[1]
                flow_ij.append(flow)
                save_optical_flow(flow,output_folder,f"optical_flow{i}_1.png")

                flow1 = flow_net(imgs_ij[1] * 255, imgs_ij[0] * 255, iters=20, test_mode=True)[1]

                flow_ji.append(flow1)
                save_optical_flow(flow1,output_folder,f"optical_flow{i}_2.png")
                

            flow_ij = torch.cat(flow_ij, dim=0)
            flow_ji = torch.cat(flow_ji, dim=0)
            valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
            valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
            print("mask is", valid_mask_i[0].shape, valid_mask_j[0].shape)
            for i in range(2):
                save_optical_flow(flow_ij[i],output_folder,f"mask flow_i{i}.png", valid_mask_i[i])
                # output_folder = "optical_flow_masks"
                save_mask(valid_mask_i[i], output_folder, f"valid_mask_i{i}.png")
                save_mask(valid_mask_j[i], output_folder, f"valid_mask_j{i}.png")
                save_optical_flow(flow_ji[i],output_folder,f"mask flow_j{i}.png", valid_mask_j[i])
            

            # cv2.imwrite("imagemask.png", (valid_mask_i[0] * torch.ones_like(valid_mask_i[0])*255).cpu().numpy())
            
            # mask_image = valid_mask_i[0].squeeze(0).cpu().numpy()  # Shape becomes (800, 800)

            # # Optionally, convert the mask to 8-bit if it’s not already in that format
            # mask_image_8bit = (mask_image * 255).astype(np.uint8)  # Scale if mask is binary (0, 1)

            # # Save the mask image
            # cv2.imwrite("mask_image.png", mask_image_8bit)
            

            
        print('flow precomputed')
        # delete the flow net
        if flow_net is not None: del flow_net
        # return flow_ij, flow_ji, valid_mask_i, valid_mask_j

        
