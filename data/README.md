# Dataset directory

Store each dataset in its own local folder, e.g. data/RiceSeg/VOC2012,
data/VegAnn/VOC2012 or data/Mixed/VOC2012. Files below data/ are excluded
from Git except this README; datasets remain on your server.

Each VOC2012 root contains JPEGImages/, SegmentationClass/ and
ImageSets/Segmentation/{train,val,test}.txt. Test is optional. Images and
masks must be 512x512. Disk mask values: 0 background, 255 plant;
loader maps them to 0/1. Lists contain filename stems without extensions.

Use --data-path data/RiceSeg/VOC2012 (or an absolute path).
No automatic mixing or split generation occurs. Use independent split IDs.
