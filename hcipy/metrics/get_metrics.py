# -*- coding: utf-8 -*-
import numpy as np

def get_focal_strehl(img, ref_img):
    strehl = img(np.argmax(ref_img)) / ref_img.max()
    return strehl

def get_pupil_strehl(aperture, ref_aperture):
    strehl = np.abs(np.sum(aperture) / np.sum(ref_aperture)) ** 2
    return strehl

def get_avg_intensity(img, ind):
    avg_intensity = np.mean(img[ind])
    return avg_intensity

def get_avg_raw_contrast(ind, img, ref_img):
    avg_intensity = get_avg_intensity(img, ind)
    strehl = get_focal_strehl(img, ref_img)
    avg_raw_contrast = avg_intensity / strehl
    return avg_raw_contrast