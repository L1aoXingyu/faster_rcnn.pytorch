#!/usr/bin/env bash

#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
#
#tar xvf VOCtrainval_06-Nov-2007.tar
#tar xvf VOCtest_06-Nov-2007.tar
#tar xvf VOCdevkit_08-Jun-2007.tar

wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/annotations.zip
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set00.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set00.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set01.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set02.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set03.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set04.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set05.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set06.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set07.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set08.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set09.tar
wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set10.tar

unzip annotations.zip
tar xvf set00.tar -C ./pedestrian
tar xvf set01.tar -C ./pedestrian
tar xvf set02.tar -C ./pedestrian
tar xvf set03.tar -C ./pedestrian
tar xvf set04.tar -C ./pedestrian
tar xvf set05.tar -C ./pedestrian
tar xvf set06.tar -C ./pedestrian
tar xvf set07.tar -C ./pedestrian
tar xvf set08.tar -C ./pedestrian
tar xvf set09.tar -C ./pedestrian
tar xvf set10.tar -C ./pedestrian