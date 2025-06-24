import tensorflow as tf

def compute_psnr(img1, img2):
    return tf.image.psnr(img1, img2, max_val=1.0).numpy()

def compute_ssim(img1, img2):
    return tf.image.ssim(img1, img2, max_val=1.0).numpy()
