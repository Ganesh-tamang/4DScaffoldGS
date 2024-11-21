### We detach the variables related to t_1 in calculation of GaussianFlow such that the gradient backward
### only works for variables at t_2 while keeping variables at t_1 unchanged because
### variables at t_1 have been updated at t_1 - 1 with the same logic.

### This can accelerate the training process since less variables needed to be updated. BTW, not detach
#### variables at t_1 will not decrase the performance but slow down the training.
import torch
from get_opt_flow import save_optical_flow


def optical_flow_loss(render_t_1, render_t_2, index, optical_flow=None):

    # Gaussian parameters at t_1
    proj_2D_t_1 = render_t_1["proj_2D"]  # shape (199800,2)
    gs_per_pixel = render_t_1["gs_per_pixel"].long()  # shape (20,800,800)
    weight_per_gs_pixel = render_t_1["weight_per_gs_pixel"]  # shape (20,800,800)
    x_mu = render_t_1["x_mu"]  # shape (20,2,800,800)
    cov2D_inv_t_1 = render_t_1["conic_2D"].detach()  # shape (199800,3)

    # Gaussian parameters at t_2
    proj_2D_t_2 = render_t_2["proj_2D"]  # shape (199800,2)
    cov2D_inv_t_2 = render_t_2["conic_2D"]  # shape (199800,3)
    cov2D_t_2 = render_t_2["conic_2D_inv"]  # shape (199800,3)

    cov2D_t_2_mtx = torch.zeros([cov2D_t_2.shape[0], 2, 2]).cuda()
    cov2D_t_2_mtx[:, 0, 0] = cov2D_t_2[:, 0]
    cov2D_t_2_mtx[:, 0, 1] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 0] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 1] = cov2D_t_2[:, 2]

    cov2D_inv_t_1_mtx = torch.zeros([cov2D_inv_t_1.shape[0], 2, 2]).cuda()
    cov2D_inv_t_1_mtx[:, 0, 0] = cov2D_inv_t_1[:, 0]
    cov2D_inv_t_1_mtx[:, 0, 1] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 0] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 1] = cov2D_inv_t_1[:, 2]

    # B_t_2
    U_t_2 = torch.svd(cov2D_t_2_mtx)[0]
    S_t_2 = torch.svd(cov2D_t_2_mtx)[1]
    V_t_2 = torch.svd(cov2D_t_2_mtx)[2]
    B_t_2 = torch.bmm(
        torch.bmm(U_t_2, torch.diag_embed(S_t_2) ** (1 / 2)), V_t_2.transpose(1, 2)
    )

    # B_t_1 ^(-1)
    U_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[0]
    S_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[1]
    V_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[2]
    B_inv_t_1 = torch.bmm(
        torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1) ** (1 / 2)),
        V_inv_t_1.transpose(1, 2),
    )

    # calculate B_t_2*B_inv_t_1
    B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)  # shape (19980,2,2)
    # calculate cov2D_t_2*cov2D_inv_t_1
    # cov2D_t_2cov2D_inv_t_1 = torch.zeros([cov2D_inv_t_2.shape[0],2,2]).cuda()
    # cov2D_t_2cov2D_inv_t_1[:, 0, 0] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1]
    # cov2D_t_2cov2D_inv_t_1[:, 0, 1] = cov2D_t_2[:, 0] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 2]
    # cov2D_t_2cov2D_inv_t_1[:, 1, 0] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 0] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 1]
    # cov2D_t_2cov2D_inv_t_1[:, 1, 1] = cov2D_t_2[:, 1] * cov2D_inv_t_1[:, 1] + cov2D_t_2[:, 2] * cov2D_inv_t_1[:, 2]

    # isotropic version of GaussianFlow
    # predicted_flow_by_gs = (proj_2D_next[gs_per_pixel] - proj_2D[gs_per_pixel].detach()) * weights.detach()

    # full formulation of GaussianFlow
    cov_multi = (
        B_t_2_B_inv_t_1[gs_per_pixel] @ x_mu.permute(0, 2, 3, 1).unsqueeze(-1).detach()
    ).squeeze()
    # print("Shape of cov_multi:", cov_multi.shape)                   # Shape after bmm operation, should be (20, 800, 800, 2) if reshaped correctly
    # print("Shape of proj_2D_t_2[gs_per_pixel]:", proj_2D_t_2[gs_per_pixel].shape)  # Shape of indexed tensor, should be (20, 800, 800, 2)
    # print("Shape of proj_2D_t_1[gs_per_pixel]:", proj_2D_t_1[gs_per_pixel].detach().shape)  # Should be (20, 800, 800, 2)
    # print("Shape of x_mu.permute(0, 2, 3, 1):", x_mu.permute(0, 2, 3, 1).detach().shape)  # Should also be (20, 800, 800, 2)
    # print("Shape of weight_per_gs_pixel:", weight_per_gs_pixel.detach().shape)  # Shape should be (20, 800, 800)

    # cov_multi = cov_multi.view(20, 800, 800, 2)

    # # Ensure proj_2D_t_1[gs_per_pixel] and proj_2D_t_2[gs_per_pixel] have shapes of (20, 800, 800, 2)
    # proj_2D_t_1_selected = proj_2D_t_1[gs_per_pixel].view(20, 800, 800, 2)
    # proj_2D_t_2_selected = proj_2D_t_2[gs_per_pixel].view(20, 800, 800, 2)

    # # Calculate predicted flow after ensuring shape compatibility
    # predicted_flow_by_gs = (
    #     cov_multi +
    #     proj_2D_t_2_selected -
    #     proj_2D_t_1_selected.detach() -
    #     x_mu.permute(0, 2, 3, 1).detach()
    # ) * weight_per_gs_pixel.detach()
    predicted_flow_by_gs = (
        cov_multi
        + proj_2D_t_2[gs_per_pixel]
        - proj_2D_t_1[gs_per_pixel].detach()
        - x_mu.permute(0, 2, 3, 1).detach()
    ) * weight_per_gs_pixel.detach().unsqueeze(-1)

    flow = predicted_flow_by_gs.sum(0).permute(2, 0, 1)
    # print("Shape of predicted_flow_by_gs:", predicted_flow_by_gs.shape, predicted_flow_by_gs.sum(0).shape)

    # predicted_flow_by_gs = (cov_multi + proj_2D_t_2[gs_per_pixel] - proj_2D_t_1[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) * weight_per_gs_pixel.detach()

    flow_thresh = 0.1
    # print("preicted flow by gs",predicted_flow_by_gs, predicted_flow_by_gs.shape)
    output_folder = "optical_flow"

    # flow supervision loss
    return save_optical_flow(
        flow.unsqueeze(0).detach(), output_folder, f"predictedflow_{index}.png"
    )
    # large_motion_msk = torch.norm(optical_flow, p=2, dim=-1) >= flow_thresh  # flow_thresh = 0.1 or other value to filter out noise, here we assume that we have already loaded pre-computed optical flow somewhere as pseudo GT
    # Lflow = torch.norm((optical_flow - predicted_flow_by_gs.sum(0))[large_motion_msk], p=2, dim=-1).mean()
    # loss = loss + 0.1 * Lflow # flow_weight could be 1, 0.1, ... whatever you want.
