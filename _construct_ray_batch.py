    def _construct_ray_batch(self, imgs_info, device='cpu'):
        imn, _, h, w = imgs_info['imgs'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.to(device)
        coords = coords.float()[None, :, :, :].repeat(imn, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(imn, h * w, 2)
        coords = torch.cat([coords + 0.5, torch.ones(imn, h * w, 1, dtype=torch.float32, device=device)],
                           2)  # imn,h*w,3
        masks = imgs_info['masks'].reshape(imn, h * w)

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        dirs = coords @ torch.inverse(imgs_info['Ks']).permute(0, 2, 1)
        imgs = imgs_info['imgs'].permute(0, 2, 3, 1).reshape(imn, h * w, 3)  # imn,h*w,3
        idxs = torch.arange(imn, dtype=torch.int64, device=device)[:, None, None].repeat(1, h * w, 1)  # imn,h*w,1
        poses = imgs_info['poses']  # imn,3,4

        rn = imn * h * w
        ray_batch = {
            'dirs': dirs.float().reshape(rn, 3).to(device),
            'rgbs': imgs.float().reshape(rn, 3).to(device),
            'idxs': idxs.long().reshape(rn, 1).to(device),
        }
        ray_batch['masks'] = masks.float().reshape(rn).to(device)
        return ray_batch, poses, rn, h, w
