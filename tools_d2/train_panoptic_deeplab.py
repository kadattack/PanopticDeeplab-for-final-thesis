#!/usr/bin/env python3
#
# Modified by Bowen Cheng
#
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""
# tensorboard --logdir="d:/Segmentacija/panoptic-deeplab-master/tools_d2/output/"
# python train_panoptic_deeplab.py --config-file ./configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml --num-gpus 1
# python train_panoptic_deeplab.py --config-file ./configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml --eval-only MODEL.WEIGHTS ./output/model_final.pth
import json
import math
import random
import numpy as np
from decimal import Decimal, Rounded

from PIL import Image, ImageDraw

import cv2
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.utils.visualizer_sharp import Visualizer as Visualizer_sharp
import sys

from networkx.drawing.tests.test_pylab import mpl
from tensorboard import program

import os
import torch
from datasets import register_MaSTr1325

import _init_paths

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

import d2


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
                "coco_val_panoptic_mastr1325": "coco_val_panoptic_mastr1325",
                "coco_val_panoptic_mastr1325_evaluate": "coco_val_panoptic_mastr1325_evaluate"
            }

            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )

        '''
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        '''
        dataset_name_mapper = {
            "coco_2017_val_panoptic": "coco_2017_val",
            "coco_2017_val_100_panoptic": "coco_2017_val_100",
            "coco_val_panoptic_mastr1325": "coco_val_panoptic_mastr1325",
            "coco_val_panoptic_mastr1325_evaluate": "coco_val_panoptic_mastr1325_evaluate"
        }

        return COCOPanopticEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)

        # elif len(evaluator_list) == 1:
        #    return evaluator_list[0]
        # return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # cfg.defrost()
    # cfg.INPUT.CROP.SIZE = (384,512)
    # cfg.INPUT.MIN_SIZE_TRAIN = (384,)
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # cfg.freeze()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # cfg.defrost()

    # cfg.SOLVER.MAX_ITER = 130000
    # cfg.freeze()

    return trainer.train()



#----------------------------------------------------------------------------------------
# ▼

def createDetection(bbox, type, id_index, area):
    detections = {
        "type": type,
        "bbox": bbox,
        "id": id_index,
        "area": area
    }
    return detections


def convertCategory(cat: int):
    if cat in [3, 5, 7, 8]:
        return "other"
    else:
        return "ship"


def createObstacle(args, model, output, large=False, show=False):
    cfg = setup(args)
    cfg.defrost()
    cfg.set_new_allowed(True)
    thresh = 0.7
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thresh
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
    cfg.DATASETS.TEST = ("coco_train_panoptic_mastr1325")
    cfg.INPUT.CROP.ENABLED = False
    cfg.freeze()
    x_mul = (Decimal(1278) / Decimal(512))
    y_mul = (Decimal(958) / Decimal(384))
    scale_factor = Decimal(1278 * 512) / Decimal(958 * 384)
    predictor = DefaultPredictor(cfg)
    jdata = json.load(open(
        "d:/Segmentacija/MODS/modb_evaluation-bbox_obstacle_detection/Detectors/Mask_RCNN/mrcnn_unedited_res.json"))

    parentdir = "d:/Segmentacija/MODS/"
    mapping_kope = open(parentdir + "mods" + "/raw/" + "sequence_mapping_new.txt").readlines()

    kope_arr = []
    seqstring_arr = []
    for mapping_line in mapping_kope:
        kope, seqstring = mapping_line.split()
        kope_arr.append(kope)
        seqstring_arr.append(seqstring)


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
                    #     im = cv2.imread(parentdir + "mods/" + "preprocessed/" + "seq39" + "/frames/" + "0850.jpg")
                    #
                    #     panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                    #     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"), scale=2.0)
                    #     # v = v.draw_instance_predictions(instances.to("cpu"))
                    #     v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                    #     cv2.imshow(file, v.get_image()[:, :, ::-1])
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                    #     exit()

                    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                    classes = instances.pred_classes if instances.has("pred_classes") else None
                    classes = classes.tolist()
                    masks = np.asarray(instances.pred_masks.to("cpu"))
                    masks = [GenericMask(x, im.shape[0], im.shape[1]) for x in masks]
                    detections = []
                    boxes = boxes.to("cpu")

                    if show:
                        orig_image = Image.fromarray(im)
                        ori_image = ImageDraw.Draw(orig_image)
                        # upscaled = cv2.resize(im, (1278, 958))
                        # scaled_image = Image.fromarray(upscaled)
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
                        for i, seqstring in enumerate(seqstring_arr):
                            if seqstring == folder:
                                kope_name = kope_arr[i]
                                break
                    else:
                        if folder not in kope_arr:
                            continue
                        kope_name = folder
                        mapper = file

                    if show:
                        print(detections)
                        orig_image.show()
                        exit(1)

                    for sequence in jdata["dataset"]["sequences"]:
                        if sequence["path"].split("/")[1] == kope_name:
                            for frame in sequence["frames"]:
                                if frame["image_file_name"] == mapper:
                                    frame["detections"] = detections
                                    break
    with open("d:/Segmentacija/MODS/modb_evaluation-bbox_obstacle_detection/Detectors/Mask_RCNN/" + output, "w") as f:
        json.dump(jdata, f)
    print("saved: d:/Segmentacija/MODS/modb_evaluation-bbox_obstacle_detection/Detectors/Mask_RCNN/" + output)


def predictMODS(args, model, path, large):
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA = 16
    cfg.set_new_allowed(True)
    thresh = 0.7
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thresh
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
    cfg.INPUT.CROP.ENABLED = False
    cfg.freeze()
    # model_0109999.pth
    predictor = DefaultPredictor(cfg)

    parentdir = "d:/Segmentacija/MODS/"

    if large:
        with open(parentdir + "mods" + "/raw/" + "sequence_mapping_new.txt") as m:
            mapping_kope = m.readlines()

        kope_arr = []
        seqstring_arr = []
        for mapping_line in mapping_kope:
            kope, seqstring = mapping_line.split()
            kope_arr.append(kope)
            seqstring_arr.append(seqstring)

    target_dir = "d:/Segmentacija/MODS/" + path + "/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not large:
        seqs = os.listdir(parentdir + "mods" + "/preprocessed/")
    else:
        seqs = os.listdir(parentdir + "mods" + "/raw/")

    for seq_folder in seqs:
        if not large:
            if not os.path.isdir(parentdir + "mods" + "/preprocessed/" + seq_folder):
                continue
            files = os.listdir(parentdir + "mods/" + "preprocessed/" + seq_folder + "/frames/")

            target_dir_seq = target_dir + seq_folder + "/"
            if not os.path.exists(target_dir_seq):
                os.makedirs(target_dir_seq)
            print(seq_folder)

        else:
            if seq_folder not in kope_arr:
                continue
            if not os.path.isdir(parentdir + "mods" + "/raw/" + seq_folder):
                continue
            files = os.listdir(parentdir + "mods/" + "raw/" + seq_folder + "/frames/")

            for i, ko in enumerate(kope_arr):
                if ko == seq_folder:
                    large_seq_folder = seqstring_arr[i]

            file_map = open(parentdir + "/mods/preprocessed/" + large_seq_folder + "/mapping.txt", "r").readlines()

            print(large_seq_folder)

            target_dir_seq = target_dir + large_seq_folder + "/"
            if not os.path.exists(target_dir_seq):
                os.makedirs(target_dir_seq)



        for file in files:
            if not large:
                im = cv2.imread(parentdir + "mods/" + "preprocessed/" + seq_folder + "/frames/" + file)
            else:
                if file[-5:-4] != "L":
                    continue
                im = cv2.imread(parentdir + "mods/" + "raw/" + seq_folder + "/frames/" + file)
                for f in file_map:
                    if f.split()[1] == file:
                        file = f.split()[0]

            panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
            # sem_seg = predictor(im)["sem_seg"]
            v = Visualizer_sharp(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"))
            # v.draw_sem_seg(sem_seg.to("cpu"))
            s = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)

            fin = target_dir_seq + file[:-3] + "png"
            cv2.imwrite(fin, s)




def predictMODSRAW(args, model, path):
    cfg = setup(args)
    cfg.defrost()
    cfg.set_new_allowed(True)
    thresh = 0.7
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thresh
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
    cfg.DATASETS.TEST = ("coco_train_panoptic_mastr1325")
    cfg.INPUT.CROP.ENABLED = False
    cfg.freeze()
    # model_0109999.pth
    predictor = DefaultPredictor(cfg)

    parentdir = "d:/Segmentacija/MODS/"
    target_dir = "d:/Segmentacija/MODS/" + path + "/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    mapping = open(parentdir + "mods" + "/raw/" + "sequence_mapping_new.txt").readlines()

    for line in mapping:
        folder, seq = line.split(" ")
        files = os.listdir(parentdir + "mods/" + "raw/" + folder + "/frames/")
        seq = str.strip(seq)
        target_dir_seq = target_dir + seq + "/"
        if not os.path.exists(target_dir_seq):
            os.makedirs(target_dir_seq)
        filecounter = 0
        for file in files:
            if file[-5:-4] == "L":
                im = cv2.imread(parentdir + "mods/" + "raw/" + folder + "/frames/" + file)
                panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                # v = Visualizer_sharp(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"), scale=1.8)
                # s = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                # s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
                # cv2.imshow("",s)
                # cv2.waitKey(0)
                # v = Visualizer_sharp(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"), scale=0.5)
                # s = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                # s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
                # cv2.imshow("", s)
                # cv2.waitKey(0)
                # exit(1)
                # if filecounter < 10:
                #     fc = "0" + str(filecounter)
                # else:
                #     fc = str(filecounter)
                # if filecounter > 99:
                #     fin = target_dir_seq + fc + "0" + ".png"
                # else:
                #     fin = target_dir_seq + "0" + fc + "0" + ".png"
                # cv2.imwrite(fin, s)
                # filecounter += 1

                if "instances" in predictor(im):
                    instances = predictor(im)["instances"]
                    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"), scale=1.8)
                    # v = v.draw_instance_predictions(instances.to("cpu"))
                    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

                    cv2.imshow(file, v.get_image()[:, :, ::-1])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        print(seq + " done")
    exit(0)


def predictSingleImage(args, model):
    cfg = setup(args)
    cfg.defrost()
    cfg.set_new_allowed(True)
    thresh = 0.7
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thresh
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
    cfg.INPUT.CROP.ENABLED = False
    cfg.DATASETS.TEST = ("coco_train_panoptic_mastr1325")
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    for i in range(135, 200):
        im = cv2.imread("d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/train/0" + str(i) + ".jpg")
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("coco_train_panoptic_mastr1325"), scale=2.0)
        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        cv2.imshow(str(i), v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    exit(0)

# ▲
#----------------------------------------------------------------------------------------



if __name__ == "__main__":
    DatasetCatalog.clear()
    args = default_argument_parser().parse_args()

#----------------------------------------------------------------------------------------
# ▼

    register_MaSTr1325.register_MaSTr1325()

    # tb = program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', "d:/Segmentacija/panoptic-deeplab-master/tools_d2/output/"])
    # url = tb.launch()


    # tensorboard --logdir="d:/Segmentacija/panoptic-deeplab-master/tools_d2/output/"

    # createObstacle(args, "model_0109999_no_lvis_crop.pth", "panoptic_0109999_no_LVIS_large_images.json", large=True, show=False)
    # createObstacle(args, "model_0109999_LVIS.pth", "panoptic_0109999_LVIS_large.json", large=False, show=False)
    # createObstacle(args, "model_0109999_3class.pth", "panoptic_0109999_large_3class.json", large=True, show=False)
    # predictMODS(args, "model_0109999_no_lvis_crop.pth", "panoptic_0109999_no_LVIS_large", large=True)
    # predictMODS(args, "model_0109999_3class.pth", "panoptic_0109999_large_3class", large=True)
    # predictMODS(args, "model_0109999_LVIS.pth", "panoptic_0109999_LVIS_small", large=False)
    # exit(1)

    # predictSingleImage(args,"model_final.pth")

# ▲
#----------------------------------------------------------------------------------------




    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
