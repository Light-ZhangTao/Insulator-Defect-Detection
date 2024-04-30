import torch


def run_one_image(img, model, flag, mask_image, mask_size):
    x = img.clone().detach()
    # run MAE
    _, y, mask = model(x.float(), flag, mask_size, mask_image, mask_ratio=0.75)
    y = model.unpatchify(y)  # patch->img
    y = torch.einsum('nchw->nhwc', y)  # .detach().cpu()
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask)  # .detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
    # masked image
    im_masked = x * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    return im_masked, im_paste


def run_main(data, model_mae, mask_image, mask_size):
    # interval mask
    mask1, rect1 = run_one_image(data, model_mae, 0, mask_image, mask_size)
    mask2, rect2 = run_one_image(data, model_mae, 1, mask_image, mask_size)
    rect = torch.zeros(rect1.shape, device=rect1.device)
    h = rect1.shape[1]
    w = rect1.shape[2]
    for i in range(h):
        for j in range(w):
            if (i // mask_size % 2 == 0 and j // mask_size % 2 == 0) or ((i // mask_size + 1) % 2 == 0 and (j // mask_size + 1) % 2 == 0):
                rect[:, i, j] = rect1[:, i, j, :]
            else:
                rect[:, i, j] = rect2[:, i, j, :]
    return rect, mask1, mask2

# upper bound
def run_main_ub(data, model_mae, mask_image, mask_size):
    mask, rect = run_one_image(data, model_mae, -1, mask_image, mask_size)
    return rect, mask