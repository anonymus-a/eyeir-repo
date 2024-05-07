import numpy as np
import torch

def out2normal(out):
    t_pred_normal_n1_n2 = out
    n1 = t_pred_normal_n1_n2[:, 0, :] # [b, 224, 224]
    n2 = t_pred_normal_n1_n2[:, 1, :]
    N3 = 1 / torch.sqrt(torch.pow(n1, 2) + torch.pow(n2, 2) + 1)
    N1 = n1 * N3
    N2 = n2 * N3
    N1 = N1.unsqueeze(1)
    N2 = N2.unsqueeze(1)
    N3 = N3.unsqueeze(1)

    N = torch.cat((N1, N2, N3), 1)
    return N

def render_sphere_nm(radius, num):

    nm = []

    for i in range(num):

        centre = radius
        x_grid, y_grid = np.meshgrid(np.arange(1.,2*radius+1), np.arange(1.,2*radius+1))
        x_grid -= centre
        y_grid = centre - y_grid
        x_grid /= radius
        y_grid /= radius
        dist = 1 - (x_grid**2+y_grid**2)
        mask = dist > 0
        z_grid = np.ones_like(mask) * np.nan
        z_grid[mask] = np.sqrt(dist[mask])

        x_grid[~(mask)] = np.nan
        y_grid[~(mask)] = np.nan

        nm.append(np.stack([x_grid,y_grid,z_grid],axis=2))

    nm = np.stack(nm, axis=0)

    return nm

def lambSH_layer(nm, L_SHcoeffs, am=None, torch_style=False):  

    if torch_style:
        nm = (nm - 0.5) * 2
        norm = torch.norm(nm, p=2, dim=1, keepdim=True) + 1e-6
        nm = nm / norm
        nm = nm.permute((0, 2, 3, 1))

        if am is not None:
            am = am.permute((0, 2, 3, 1))

    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743125
    c4 = 0.886227
    c5 = 0.247708

    M_row1 = torch.stack([c1 * L_SHcoeffs[:, 8, :], c1 * L_SHcoeffs[:, 4, :], c1 * L_SHcoeffs[:, 7, :], c2 * L_SHcoeffs[:, 3, :]], dim=1)
    M_row2 = torch.stack([c1 * L_SHcoeffs[:, 4, :], -c1 * L_SHcoeffs[:, 8, :], c1 * L_SHcoeffs[:, 5, :], c2 * L_SHcoeffs[:, 1, :]], dim=1)
    M_row3 = torch.stack([c1 * L_SHcoeffs[:, 7, :], c1 * L_SHcoeffs[:, 5, :], c3 * L_SHcoeffs[:, 6, :], c2 * L_SHcoeffs[:, 2, :]], dim=1)
    M_row4 = torch.stack([c2 * L_SHcoeffs[:, 3, :], c2 * L_SHcoeffs[:, 1, :], c2 * L_SHcoeffs[:, 2, :], c4 * L_SHcoeffs[:, 0, :] - c5 * L_SHcoeffs[:, 6, :]], dim=1)

    M = torch.stack([M_row1, M_row2, M_row3, M_row4], dim=1)
    M = M.cuda()

    total_npix = nm.shape[:3]
    ones = torch.ones(total_npix).cuda()
    nm_homo = torch.cat([nm, torch.unsqueeze(ones, dim=-1)], dim=-1) 

    M = torch.unsqueeze(torch.unsqueeze(M, dim=1), dim=1) 

    nm_homo = torch.unsqueeze(torch.unsqueeze(nm_homo, dim=-1), dim=-1)
    tmp = torch.sum(nm_homo * M, dim=-3)
    E = torch.sum(tmp * nm_homo[:, :, :, :, 0, :], dim=-2)
    
    if am is not None:
        i = E * am
        return i.permute((0, 3, 1, 2))
    else:
        return E.permute((0, 3, 1, 2))

