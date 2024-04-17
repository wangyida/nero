    def _construct_nerf_ray_batch(self, imgs_info, device='cpu', is_train=True):
        imn, _, h, w = imgs_info['imgs'].shape

        i, j = torch.meshgrid(torch.linspace(0, w - 1, w),
                              torch.linspace(0, h - 1, h))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()

        K = imgs_info['Ks'][0]
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4
        if is_train:
            masks = imgs_info['masks'].reshape(imn, h * w)

        rays_d = [torch.sum(dirs[..., None, :].cpu() * poses[i, :3, :3], -1) for i in range(imn)]
        rays_d = torch.stack(rays_d, 0).reshape(imn, h * w, 3)
        rays_o = [poses[i, :3, -1].expand(rays_d[0].shape) for i in range(imn)]
        rays_o = torch.stack(rays_o, 0).reshape(imn, h * w, 3)
        rn = imn * h * w
        ray_batch = {
            # 'dirs': dirs.float().reshape(rn, 3).to(device),
            'rgbs': imgs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
            'rays_o': rays_o.float().reshape(rn, 3).to(device),
            'rays_d': rays_d.float().reshape(rn, 3).to(device),
        }
        if is_train:
            ray_batch['masks'] = masks.float().reshape(rn).to(device)
        return ray_batch, poses, rn, h, w
