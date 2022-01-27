#----------------------------------------------------------------------------------------
# ▼

import json
import os
from decimal import Decimal

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer, GenericMask
import numpy as np

from PIL import Image, ImageDraw


from datasets import register_MaSTr1325
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer_sharp import Visualizer as Visualizer_sharp



COCO_CATEGORIES = [
    {"color": [100, 250, 0], "isthing": 1, "id": 4, "name": "ship_part"},
    {"color": [0, 50, 250], "isthing": 1, "id": 5, "name": "boat"},
    {"color": [250, 50, 100], "isthing": 1, "id": 6, "name": "buoy"},
    {"color": [250, 100, 150], "isthing": 1, "id": 7, "name": "ship"},
    {"color": [250, 150, 200], "isthing": 1, "id": 8, "name": "floating_fence"},
    {"color": [250, 200, 250], "isthing": 1, "id": 9, "name": "unknown_object"},
    {"color": [250, 250, 0], "isthing": 1, "id": 10, "name": "yacht"},
    {"color": [250, 250, 100], "isthing": 1, "id": 11, "name": "water_scooter"}
]




def get_coco_instances_meta_MaSTr1325():

    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def cleanInstancesJson():
    root = "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/annotations/"
    jdata = json.load(open(root + "instances_train.json"))
    flag = True
    while flag:
        flag = False
        for i, ann in enumerate(jdata["annotations"]):
            if ann["category_id"] < 4:
                del jdata["annotations"][i]
                flag = True
                break
    flag = True
    while flag:
        flag = False
        for i, cat in enumerate(jdata["categories"]):
            if cat["id"] < 4:
                del jdata["categories"][i]
                flag = True
                break
    json.dump(jdata, open(root + "instances_train_mcrnn.json","w"))

    root = "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/"
    flag = True
    while flag:
        flag = False
        for i, ann in enumerate(jdata["annotations"]):
            if ann["category_id"] < 4:
                del jdata["annotations"][i]
                flag = True
                break

    flag = True
    while flag:
        flag = False
        for i, cat in enumerate(jdata["categories"]):
            if cat["id"] < 4:
                del jdata["categories"][i]
                flag = True
                break

    json.dump(jdata, open(root + "instances_train_mcrnn.json","w"))
    # exit(1)



def createDetection(bbox, type ,id_index, area):
    detections = {
        "type": type,
        "bbox": bbox,
        "id": id_index,
        "area": area
    }
    return detections

def convertCategory(cat: int):
    # ship: 1,3,6,7
    # other: 0,2,4,5
    if cat in [0,2,4,5]:
        return "other"
    elif cat in [1,3,6,7]:
        return "ship"

    TypeError("Unknown Category")
    exit(2)





def createObstacle(cfg, model, output, large=False, show=False):
    # Large should be written in to json file
    cfg.defrost()
    cfg.set_new_allowed(True)
    thresh = 0.7
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thresh
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
    cfg.DATASETS.TEST = ("coco_train_panoptic_mastr1325")
    cfg.INPUT.CROP.ENABLED = False
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.freeze()
    x_mul = (Decimal(1278) / Decimal(512))
    y_mul = (Decimal(958) / Decimal(384))
    scale_factor = Decimal(1278 * 512) / Decimal(958 * 384)
    predictor = DefaultPredictor(cfg)
    jdata = json.load(open("d:/Segmentacija/MODS/modb_evaluation-bbox_obstacle_detection/Detectors/Mask_RCNN/mrcnn_unedited_res.json"))

    parentdir = "d:/Segmentacija/MODS/"
    mapping_kope = open(parentdir + "mods" + "/raw/" + "sequence_mapping_new.txt").readlines()
    if not large:
        mapping = os.listdir(parentdir + "mods/preprocessed/")
    else:
        mapping = os.listdir(parentdir + "mods/raw/")

    for sequence in jdata["dataset"]["sequences"]:
        for frame in sequence["frames"]:
            frame["detections"] = []
            if "obstacles" in frame:
                del frame["obstacles"]

    for folder in mapping:
        if not large:
            if os.path.isdir(parentdir + "mods/preprocessed/" + folder):
                files = os.listdir(parentdir + "mods/" + "preprocessed/" + folder + "/frames/")
            else:
                continue
        else:
            if os.path.isdir(parentdir + "mods/raw/" + folder):
                files = os.listdir(parentdir + "mods/" + "raw/" + folder + "/frames/")
            else:
                continue

        print(folder)
        if show and not large and (folder == "seq01" or folder == "seq02" or folder == "seq03"):
            continue

        if not large:
            mapping_list = open(parentdir + "mods/" + "preprocessed/" + folder + "/mapping.txt", "r").readlines()
        else:
            mapping_list = open(parentdir + "mods/" + "raw/" + folder + "/mapping.txt", "r").readlines()

        for file in files:
            if not large:
                im = cv2.imread(parentdir + "mods/" + "preprocessed/" + folder + "/frames/" + file)
            else:
                if file[-5:-4] == "L":
                    im = cv2.imread(parentdir + "mods/" + "raw/" + folder + "/frames/" + file)
                else:
                    continue

            if "instances" in predictor(im):
                instances = predictor(im)["instances"]

                # if True:
                #     panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                #     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"), scale=2.0)
                #     # v = v.draw_instance_predictions(instances.to("cpu"))
                #     v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                #     cv2.imshow(file, v.get_image()[:, :, ::-1])
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                classes = instances.pred_classes if instances.has("pred_classes") else None
                classes = classes.tolist()
                masks = np.asarray(instances.pred_masks.to("cpu"))
                masks = [GenericMask(x, im.shape[0], im.shape[1]) for x in masks]
                detections = []
                boxes = boxes.to("cpu")

                if show:
                    upscaled = cv2.resize(im, (1278, 958))
                    orig_image = Image.fromarray(im)
                    # scaled_image = Image.fromarray(upscaled)
                    ori_image = ImageDraw.Draw(orig_image)
                    # im_scaled = ImageDraw.Draw(scaled_image)

                for box, index in zip(boxes, range(len(boxes))):
                    type = convertCategory(classes[index])
                    x0, y0, x1, y1 = box.numpy().tolist()

                    if large:
                        mul_x0 = round(x0)
                        mul_x1 = round(x1)
                        mul_y0 = round(y0)
                        mul_y1 = round(y1)
                    else:
                        mul_x0 = round(Decimal(x_mul) * Decimal(x0))
                        mul_x1 = round(Decimal(x_mul) * Decimal(x1))
                        mul_y0 = round(Decimal(y_mul) * Decimal(y0))
                        mul_y1 = round(Decimal(y_mul) * Decimal(y1))

                    width = mul_x1 - mul_x0
                    height = mul_y1 - mul_y0

                    if show:
                        ori_image.rectangle((mul_x0, mul_y0, mul_x1, mul_y1))

                    bbox = [mul_x0, mul_y0, width, height]
                    area = round(Decimal(int(masks[index].area())) * Decimal(scale_factor))
                    detections.append(createDetection(bbox, type, index, area))

                if not large:
                    for mapping_line in mapping_list:
                        processed_file, raw_file = mapping_line.split(" ")
                        if processed_file == file:
                            mapper = raw_file[:-1]
                            break
                    for mapping_line in mapping_kope:
                        kope, seqstring = mapping_line.split()
                        if seqstring == folder:
                            kope_name = kope
                            break
                else:
                    kope_name = folder
                    mapper = file

                if show:
                    orig_image.show()
                    # exit(1)

                for sequence in jdata["dataset"]["sequences"]:
                    if sequence["path"].split("/")[1] == kope_name:
                        for frame in sequence["frames"]:
                            if frame["image_file_name"] == mapper:
                                frame["detections"] = detections
                                break


    with open("d:/Segmentacija/MODS/modb_evaluation-bbox_obstacle_detection/Detectors/Mask_RCNN/" + output,"w") as f:
        json.dump(jdata,f)
    print("saved: d:/Segmentacija/MODS/modb_evaluation-bbox_obstacle_detection/Detectors/Mask_RCNN/" + output)





def visualise(cfg, model):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
    predictor: DefaultPredictor = DefaultPredictor(cfg)

    parentdir = "d:/Segmentacija/MODS/"
    mapping = open(parentdir + "mods" + "/raw/" + "sequence_mapping_new.txt").readlines()

    for line in mapping:
        folder, seq = line.split(" ")
        files = os.listdir(parentdir + "mods/" + "raw/" + folder + "/frames/")
        seq = str.strip(seq)
        # print(seq)
        # if seq == "seq01":
        #     continue
        for file in files:
            if file[-5:-4] == "L":
                im = cv2.imread(parentdir + "mods/" + "raw/" + folder + "/frames/" + file)
                outputs = predictor(
                    im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                instances = predictor(im)["instances"]
                boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                classes = instances.pred_classes if instances.has("pred_classes") else None

                v = Visualizer(im[:, :, ::-1],
                               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                               scale=0.8,
                               instance_mode=ColorMode.IMAGE_BW
                               # remove the colors of unsegmented pixels. This option is only available for segmentation models
                               )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imshow("file", out.get_image()[:, :, ::-1])
                cv2.waitKey(0)



    t_folder = "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/train/"
    folder = os.listdir(t_folder)
    for file in folder:
        if "jpg" in file:
            im = cv2.imread(t_folder + file)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                           metadata=get_coco_instances_meta_MaSTr1325(),
                           scale=2.0,
                           instance_mode=ColorMode.IMAGE_BW
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("file", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
    exit(0)


if __name__ == "__main__":
    import torch

    cleanInstancesJson()



    modelpath = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

    model = model_zoo.get(modelpath, trained=True)

    register_coco_instances(
        "coco_instance_MaSTR1325",
        get_coco_instances_meta_MaSTr1325(),
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/instances_train_mcrnn.json",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/train/",
    )
    register_coco_instances(
        "coco_instance_LVIS_MaSTR1325",
        get_coco_instances_meta_MaSTr1325(),
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/annotations/instances_train_mcrnn.json",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/train/",
    )

    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(modelpath))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(modelpath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TRAIN = ("coco_instance_LVIS_MaSTR1325",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 11
    cfg.MODEL.RETINANET.NUM_CLASSES = 11
    cfg.SOLVER.MAX_ITER = 250000
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    cfg.OUTPUT_DIR = "d:/Segmentacija/panoptic-deeplab-master/tools_d2/output_maskRCNN/"



    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MIN_SIZE_TRAIN = (320, 352, 384)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = 384
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (320, 320)




    # visualise(cfg, "model_0007499_no_LVIS.pth")
    # visualise(cfg, "model_0007499.pth")

    createObstacle(cfg, "best-model_0007499_LVIS.pth", "MaskRCNN_007499_LVIS_small.json", large=False, show=False)
    createObstacle(cfg, "best-model_0007499_LVIS.pth", "MaskRCNN_007499_LVIS_large.json", large=True, show=False)
    createObstacle(cfg, "best-model_0007499_no_LVIS.pth", "MaskRCNN_007499_no_LVIS_large.json", large=True, show=False)

    # createObstacle(cfg, "model_0019999.pth", "MaskRCNN_0019999_Resnet101_FPN_small.json", large=False, show=True)
    # createObstacle(cfg, "model_0007499.pth", "MaskRCNN_0019999_Resnet101_FPN_large.json", large=True, show=True)
    # createObstacle(cfg, "model_0007499_no_LVIS.pth", "MaskRCNN_0019999_ResnetX101_FPN_small.json", large=False, show=False)
    # createObstacle(cfg, "model_0007499.pth", "MaskRCNN_0012499_ResnetX101_FPN_small_crop_resize.json", large=True, show=False)
    exit(1)


    #tensorboard --logdir="d:/Segmentacija/panoptic-deeplab-master/tools_d2/output_maskRCNN/"


    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


# ▲
#----------------------------------------------------------------------------------------
