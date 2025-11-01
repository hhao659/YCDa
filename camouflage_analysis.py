import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

def rgb_to_ycbcr(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y) + 128
    cr = 0.713 * (r - y) + 128
    cbcr = np.stack([cb, cr], axis=-1)
    return y.astype(np.uint8), cbcr.astype(np.uint8)

def ycbcr_to_rgb(y, cbcr):
    cb, cr = cbcr[:, :, 0], cbcr[:, :, 1]
    r = y + 1.403 * (cr - 128)
    g = y - 0.714 * (cr - 128) - 0.344 * (cb - 128)
    b = y + 1.773 * (cb - 128)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def compute_f_measure(pred, gt, beta=1.0):
    pred_bin = (pred > 128).astype(np.uint8)
    gt_bin = (gt > 128).astype(np.uint8)
    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin) - tp
    fn = np.sum(gt_bin) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-6)

def decouple_camouflaged_image(image_path, output_y_path='luminance.png', output_cbcr_path='chrominance.png'):
    img = plt.imread(image_path) * 255
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    y, cbcr = rgb_to_ycbcr(img)
    
    plt.imshow(y, cmap='gray')
    plt.title('Luminance (Y)')
    plt.axis('off')
    plt.savefig(output_y_path)
    plt.close()
    
    cbcr_vis = np.concatenate([cbcr, np.zeros_like(y)[:, :, np.newaxis]], axis=2)
    plt.imshow(cbcr_vis)
    plt.title('Chrominance (CbCr)')
    plt.axis('off')
    plt.savefig(output_cbcr_path)
    plt.close()
    
    print(f"Luminance saved to {output_y_path}, Chrominance saved to {output_cbcr_path}")

def swap_channels_and_measure(salient_path, camouflaged_path, gt_mask_path=None,
                              output_swap_camouflaged='swapped_camouflaged.png',
                              output_swap_salient='swapped_salient.png'):
    salient_img = plt.imread(salient_path) * 255
    camouflaged_img = plt.imread(camouflaged_path) * 255
    
    if len(salient_img.shape) == 2:
        salient_img = np.stack([salient_img] * 3, axis=-1)
    elif len(salient_img.shape) == 3 and salient_img.shape[2] == 4:
        salient_img = salient_img[:, :, :3]
    
    if len(camouflaged_img.shape) == 2:
        camouflaged_img = np.stack([camouflaged_img] * 3, axis=-1)
    elif len(camouflaged_img.shape) == 3 and camouflaged_img.shape[2] == 4:
        camouflaged_img = camouflaged_img[:, :, :3]
    
    salient_img = salient_img.astype(np.uint8)
    camouflaged_img = camouflaged_img.astype(np.uint8)
    
    target_shape = (min(salient_img.shape[0], camouflaged_img.shape[0]),
                    min(salient_img.shape[1], camouflaged_img.shape[1]), 3)
    salient_img = resize(salient_img, target_shape, anti_aliasing=True) * 255
    camouflaged_img = resize(camouflaged_img, target_shape, anti_aliasing=True) * 255
    salient_img = salient_img.astype(np.uint8)
    camouflaged_img = camouflaged_img.astype(np.uint8)
    
    y_salient, cbcr_salient = rgb_to_ycbcr(salient_img)
    y_camouflaged, cbcr_camouflaged = rgb_to_ycbcr(camouflaged_img)
    
    swapped_camouflaged = ycbcr_to_rgb(y_camouflaged, cbcr_salient)
    swapped_salient = ycbcr_to_rgb(y_salient, cbcr_camouflaged)
    
    plt.imsave(output_swap_camouflaged, swapped_camouflaged / 255.0)
    plt.imsave(output_swap_salient, swapped_salient / 255.0)
    
    ssim_before_camouflaged = ssim(camouflaged_img, salient_img, multichannel=True, channel_axis=2)
    ssim_after_camouflaged = ssim(swapped_camouflaged, salient_img, multichannel=True, channel_axis=2)
    ssim_before_salient = ssim(salient_img, salient_img, multichannel=True, channel_axis=2)
    ssim_after_salient = ssim(swapped_salient, salient_img, multichannel=True, channel_axis=2)
    
    print(f"SSIM (Camouflaged vs. Salient): Before swap = {ssim_before_camouflaged:.4f}, After swap = {ssim_after_camouflaged:.4f}")
    print(f"SSIM (Salient vs. Salient): Before swap = {ssim_before_salient:.4f}, After swap = {ssim_after_salient:.4f}")
    
    if gt_mask_path:
        gt_mask = plt.imread(gt_mask_path) * 255
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = resize(gt_mask, target_shape[:2], anti_aliasing=True) * 255
        gt_mask = gt_mask.astype(np.uint8)
        
        pred_camouflaged = np.mean(camouflaged_img, axis=2)
        pred_swapped_camouflaged = np.mean(swapped_camouflaged, axis=2)
        
        f_before = compute_f_measure(pred_camouflaged, gt_mask)
        f_after = compute_f_measure(pred_swapped_camouflaged, gt_mask)
        
        print(f"F-measure (Camouflaged): Before swap = {f_before:.4f}, After swap = {f_after:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Camouflage analysis in YCbCr space")
    parser.add_argument('--function', choices=['decouple', 'swap'], required=True, help='Function to run: decouple or swap')
    parser.add_argument('--input', help='Path to camouflaged image (for decouple)')
    parser.add_argument('--salient', help='Path to salient image (for swap)')
    parser.add_argument('--camouflaged', help='Path to camouflaged image (for swap)')
    parser.add_argument('--gt_mask', help='Path to ground truth mask (optional, for swap)')
    parser.add_argument('--output_y', default='luminance.png', help='Output path for luminance image')
    parser.add_argument('--output_cbcr', default='chrominance.png', help='Output path for chrominance image')
    parser.add_argument('--output_swap_camouflaged', default='swapped_camouflaged.png', help='Output path for swapped camouflaged image')
    parser.add_argument('--output_swap_salient', default='swapped_salient.png', help='Output path for swapped salient image')
    
    args = parser.parse_args()
    
    if args.function == 'decouple':
        if not args.input:
            raise ValueError("Input image path required for decouple function")
        decouple_camouflaged_image(args.input, args.output_y, args.output_cbcr)
    elif args.function == 'swap':
        if not (args.salient and args.camouflaged):
            raise ValueError("Both salient and camouflaged image paths required for swap function")
        swap_channels_and_measure(args.salient, args.camouflaged, args.gt_mask,
                                 args.output_swap_camouflaged, args.output_swap_salient)

if __name__ == "__main__":
    main()
