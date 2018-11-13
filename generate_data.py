# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--image-dir', '-i', required=True, help='the images path')
    parse.add_argument('--label-file', '-l', required=True, help='the .csv file of images label')
    parse.add_argument('--output-dir', '-o', defualt='./', help='the output path' )
    args = parse.parse_args()

    print('Preparing data....')
    data = pd.read_csv(args.label_file)
    imgs = data['filename']
    print('data shape: ', imgs.shape)
    
    train_imgs = []
    for i in range(len(imgs)):
        img_name  = imgs[i]
        img_path = args.image_dir + img_name
        img = cv2.imread(img_path)
        train_imgs.append(img)
    
    train_imgs = np.array(train_imgs)
    print('train_data shape: ', train_imgs.shape)
    np.save(args.output_dir + 'images.npy', train_imgs)
    print('over')