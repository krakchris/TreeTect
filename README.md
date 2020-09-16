
# Introduction

Green City Watch is an open source initiative specialized in urban ecological engineering. We leverage remote sensing and machine learning, known as “geoAI” to boost urban tree longevity.

The time is ripe for a smarter way to map, monitor, and manage one of the most important urban infrastructural assets: trees. While “smart cities” continue to innovate and invest in applications for “smart” waste management, mobility, and  lighting, trees have long been left behind.

Green City Watch has worked with 30+ (mega)cities, from Boston to Amsterdam to Jakarta, to provide open-source software to understand, oversee, and improve urban forests.

***

# TreeTect Opensource

TreeTect is and open source Tree Detection algorithm using Object detection.
TreeTect was designed to run on AWS Services using Lambda and Sagemaker, but also works fine locally.

![image](https://user-images.githubusercontent.com/32303294/92995149-1851ae00-f501-11ea-92c0-67fa6ac25f50.png)

## Input
* 4 or 8 band aerial or satellite imagery with a spatial resolution between 20-80cm including RGBI bands
* Trained model or Tree annotations

## Output
* Tree Detection Shapefile boxes
* Tree Detection Shapefile points

## Algortithm Steps
### Creating inferences from large satellite/aerial image
* Tiling a large image into small 400x400 pixel tiles
* Choosing models for ensembling and running ensemble models

### (Re)Training a model on hand annotations from Image
* Tiling a large image into small 400x400 pixel tiles
* (OPTIONAL) Choosing models for ensembling and running ensemble models
   * Improve and adjust bounding boxes in preferred GIS programme
* Create bounding boxes on image tiles
* Create Training CSV from Bounding boxes
* (Re)Train Chosen models

## Wiki
In the [Wiki](https://https://github.com/krakchris/TreeTect/wiki) you can find descriptions on how to use each of the algorithms and what their respective input conditions are.

## Basemodels
The base models are shared in THIS S3 Bucket.
We have tested the following algorithms to be working:

* faster_rcnn_inception_resnet_v2_atrous_coco(batch size -1)
* faster_rcnn_resnet101_coco_model(batch size -1)
* ssd_resnet50_fpn_coco_model(batch size-1)
* ssd_inception_v2_coco_model(batch size - 16)
* rfcn_resnet101_coco_model(batch size - 1)
* aster_rcnn_resnet50_coco_model(batch size - 1)
* faster_rcnn_nas(batch size-1)
* ssd_mobilenet_v1_coco(batch size -16)
* ssd_mobilenet_v2_coco(batch size -16)
* ssd_mobilenet_v1_fpn_coco_model(batch size- 16)
* faster_rcnn_resnet101_kitti(batch size - 1)

## OpenData
Treetect was was tested on Worldview-2/3, Pleiades and SkySat and Aerial imagery.
If you do not have access to Commercial High Resolution Imagery you can find a NAIP Dataset Here

