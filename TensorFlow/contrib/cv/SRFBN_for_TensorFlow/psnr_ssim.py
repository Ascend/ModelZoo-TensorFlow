import numpy as np
import cv2

#灰度图psnr
def compare_psnr_gray(im1,im2,maxI=255):
    im1=im1.astype(np.float64)
    im2=im2.astype(np.float64)
    diff=im1-im2
    mse=np.mean(np.square(diff))
    if mse==0: return float('inf')
    return 10*np.log10(maxI*maxI/mse)

#彩色图psnr
def compare_psnr_rgb(im1,im2,maxI=255):
    blue1,green1,red1=cv2.split(im1)
    blue2,green2,red2=cv2.split(im2)
    psnr_blue=compare_psnr_gray(blue1,blue2,maxI)
    psnr_green=compare_psnr_gray(green1,green2,maxI)
    psnr_red=compare_psnr_gray(red1,red2,maxI)
    
    #三通道取平均值
    return (psnr_blue+psnr_green+psnr_red)/3

#可以直接用的（不能把灰度图和彩色图直接比较）
def compare_psnr(im1,im2,maxI=255):
    #如果两张图片大小不同或者同为同一类型（灰度、彩色）图就报错
    if im1.shape!=im2.shape: raise ValueError("错误：图片大小维度不同")
    if (im1.ndim==2) and (im2.ndim==2): return compare_psnr_gray(im1,im2)
    #彩色图里可能有单通道（？）
    elif (im1.ndim==3) and (im2.ndim==3): 
        if im1.shape[2]==3:
            return compare_psnr_rgb(im1,im2)
        elif im1.shape[2]==1:
            return compare_psnr_gray(np.squeeze(im1),np.squeeze(im2))
    else: raise ValueError("错误：错误的图片维度")

#ssim
def ssim(im1,im2,maxI=255):
    #0.01和0.03是已经定下来的常数，不要改
    c1=(0.01*maxI)**2
    c2=(0.03*maxI)**2

    #转换成float64类型
    im1=im1.astype(np.float64)
    im2=im2.astype(np.float64)
    #高斯核，这里11和1.5也是定下来的不要改
    kernel=cv2.getGaussianKernel(11,1.5)
    window=np.outer(kernel,kernel.transpose())
    
    #求卷积
    #ssim先将图片分成若干个小块后分别按照公式的各元素求各种卷积
    mu1=cv2.filter2D(im1,-1,window)[5:-5,5:-5]
    mu2=cv2.filter2D(im2,-1,window)[5:-5,5:-5]
    mu1_sq=mu1**2
    mu2_sq=mu2**2
    mu1_mu2=mu1*mu2
    sigma1_sq=cv2.filter2D(im1**2,-1,window)[5:-5,5:-5]-mu1_sq
    sigma2_sq=cv2.filter2D(im2**2,-1,window)[5:-5,5:-5]-mu2_sq
    sigma12=cv2.filter2D(im1*im2,-1,window)[5:-5,5:-5]-mu1_mu2

    #ssim的计算公式
    ssim_map=((2*mu1_mu2+c1)*(2*sigma12+c2))/((mu1_sq+mu2_sq+c1)*(sigma1_sq+sigma2_sq+c2))
    #取所有小块计算结果的平均值
    return ssim_map.mean()

#可以直接用的
def compare_ssim(im1,im2,maxI=255):
    #如果两张图片大小不同或者同为同一类型（灰度、彩色）图就报错
    if im1.shape!=im2.shape:
        raise ValueError("错误：图片维度大小不同")
    if im1.ndim==2:
        return ssim(im1,im2)
    #彩色图里可能有单通道（？）
    elif im1.ndim==3:
        if im1.shape[2]==3:
            blue1,green1,red1=cv2.split(im1)
            blue2,green2,red2=cv2.split(im2)
            ssim_blue=ssim(blue1,blue2)
            ssim_green=ssim(green1,green2)
            ssim_red=ssim(red1,red2)
            
            #同psnr，取平均值
            return (ssim_blue+ssim_green+ssim_red)/3
        elif im1.shape[2]==1:
            return ssim(np.squeeze(im1),np.squeeze(im2))
    else: raise ValueError("错误：错误的图片维度")