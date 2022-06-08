# 2022_ComputerVision_TermProject ⭐
With the Mask R-CNN pretrained, the images can be segmented into the mask and bounding boxes. 

---

# 2022년 컴퓨터 비전 Term Project
## Task : Instance Segmentation, proposed by the Mask R-CNN baseline

![image](https://user-images.githubusercontent.com/40708515/168009335-9f31a818-5375-404f-8281-00a6f5ced668.png)
Instance segmentation은 Object detection Task에서 파생된 챌린지입니다. Object detection은 한 물체가 아닌 여러 물체에 대해 어떤 물체인지 클래스를 분류하는 Classification 문제와 그 물체가 어디에 있는지 바운딩 박스(Bbox)를 통해 나타내는 Localization 문제를 모두 포함합니다.   


</b>Instance segmentation</b>은 이미지 내 존재하는 모든 객체를 탐지하는 동시에, 각각의 물체를 정확하게 Pixel 단위로 분류하는 Task 입니다. Class를 기준으로 분류하는 Sementaic segmentation과는 달리, Instance segmentation은 이미지 내에 같은 클래스의 이미지도 모두 개별로 생각하여 분류를 합니다. 

- Instance segmentaion을 하기 위해 사용한 베이스라인 모델은 Mask R-CNN입니다. (https://github.com/matterport/Mask_RCNN) 
- 본 프로젝트는 해당 모델을 기준으로 베이스라인을 작성했습니다. (https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py)

## Dataset

![image](https://user-images.githubusercontent.com/40708515/168010517-dcef6192-814d-46c8-98ee-6ae127ec1c41.png)

사용한 데이터셋은 COCO (https://cocodataset.org/) 셋입니다. 그중 2017년 Validation set를 사용하였습니다. (5K, 1GB)   
본 Dataset의 document에 나와 있는대로 Results Format을 변경하였습니다. (자세한 사항은 https://cocodataset.org/#format-results 를 참조바랍니다.)


## Evaluation Metrics
Evaluation 부분도 해당 데이터셋의 Detectuion Evaluation Metrics를 참조하였습니다. https://cocodataset.org/#detection-eval

`Average Precision (AP):`
- AP% AP at IoU=.50:.05:.95 (primary challenge metric)   
- APIoU=.50% AP at IoU=.50 (PASCAL VOC metric)   
- APIoU=.75% AP at IoU=.75 (strict metric)  

`AP Across Scales:`
- APsmall% AP for small objects: area < 322   
- APmedium% AP for medium objects: 322 < area < 962   
- APlarge% AP for large objects: area > 962


`Average Recall (AR):`
- ARmax=1% AR given 1 detection per image   
- ARmax=10% AR given 10 detections per image   
- ARmax=100% AR given 100 detections per image


`AR Across Scales:`   
- ARsmall% AR for small objects: area < 322   
- ARmedium% AR for medium objects: 322 < area < 962   
- ARlarge% AR for large objects: area > 962`

## Submission Format

submittion은 pkl(Pickle) 객체를 이용해서 제출해주세요. 



해당 Competition의 제출 양식은 다음과 같습니다. 
[
{"rois"(bbox) : ~,
 "class_ids"(현 이미지에 존재하는 class_id들. class_id는 COCO 데이터셋 annotation을 참조할 것) : ~,
 "scores" (bbox의 RoI 계산 값) : ~,
 "masks" (binary mask로 객체 픽셀단위의 mask) : ~ ,}
 ]

(example)

[[{'rois': array([[ 476,  147,  650,  240],
          [ 507,  637,  674,  699],
          [ 504,  324,  660,  385],
          [ 500,  833,  589,  856],
          [ 507,  736,  683,  787],
          [ 498,  918,  612,  951],
          [ 513,  875,  590,  897],
          [ 477,  361,  577,  401],
          [ 509,  953,  602,  980],
          [ 524,  408,  590,  465],
          [ 306,  493,  376,  533],
          [ 504,  710,  565,  734],
          [ 473,  315,  581,  353],
          [ 498,  677,  549,  698],
          [ 288,  577,  372,  633],
          [ 505,  852,  577,  864],
          [ 529,  980,  607, 1007],
          [ 531,  396,  581,  425],
          [ 516,  219,  565,  237],
          [ 503,  727,  563,  747]]),
   'class_ids': array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  2, 10,  1,  1,  1, 13,  1,  1,
           2, 25,  1]),
   'scores': array([0.9996618 , 0.99922824, 0.99864393, 0.99319434, 0.9931377 ,
          0.9907314 , 0.98900324, 0.9884697 , 0.98147696, 0.9802548 ,
          0.9697119 , 0.9636799 , 0.95280224, 0.93408614, 0.9287156 ,
          0.9132278 , 0.90947175, 0.81997   , 0.8192738 , 0.7802971 ],
         dtype=float32),
   'masks': array([[[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]],
   
          [[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]],
   
          [[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]],
   
          ...,
   
          [[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]],
   
          [[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]],
   
          [[False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ...,
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]]])}],
 [{'rois': array([[ 229,  647,  544,  970],
          [ 657,  713,  782,  819],
          [ 658,  790,  783,  926],
          [ 711,  571,  782,  647],
          [ 696,    2,  782,   94],
          [ 664,  950,  794, 1024],
          [ 639,  224,  769,  450],
          [ 720,  453,  778,  497],
          [ 654,  649,  673,  698],
          [ 668,    0,  752,   33],
          [ 617,   13,  778,  326],
          [ 621,  284,  663,  481],
          [ 663,   22,  847,  988],
          [ 663,  887,  679,  955],
          [ 739,  927,  776,  961]]),


`
---
- 사용한 라이브러리
-  -  https://github.com/cocodataset/cocoapi
-  -  https://github.com/facebookresearch/detectron2
-  -  

- Reference
-  -  https://www.kaggle.com/code/iiyamaiiyama/how-to-submit-prediction#prepare-mask_rcnn
-  -  tutorial https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=h9tECBQCvMv3
-  -  https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
-  -  

---
Mask R-CNN 설명 동영상 : https://drive.google.com/file/d/1OCw-UDgVtm1iXWWc144WrGiE3pl6CoBI/view?usp=sharing

발표 자료 : https://drive.google.com/file/d/1-THbJzPxu_8T89Jzw9JhnEmUlReA5buZ/view?usp=sharing
