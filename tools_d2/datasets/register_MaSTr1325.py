
#----------------------------------------------------------------------------------------
# ▼

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic, load_coco_panoptic_json

from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import _PanopticPrediction, ColorMode, _OFF_WHITE

COCO_CATEGORIES_MaSTr1325 = [
    {"color": [200, 0, 50], "isthing": 0, "id": 1, "name": "obstacles_and_environment"},
    {"color": [20, 70, 100], "isthing": 0, "id": 2, "name": "water"},
    {"color": [200, 100, 250], "isthing": 0, "id": 3, "name": "sky"},
    {"color": [100, 250, 0], "isthing": 1, "id": 4, "name": "ship_part"},
    {"color": [0, 50, 250], "isthing": 1, "id": 5, "name": "boat"},
    {"color": [250, 50, 100], "isthing": 1, "id": 6, "name": "buoy"},
    {"color": [250, 100, 150], "isthing": 1, "id": 7, "name": "ship"},
    {"color": [250, 150, 200], "isthing": 1, "id": 8, "name": "floating_fence"},
    {"color": [250, 200, 250], "isthing": 1, "id": 9, "name": "unknown_object"},
    {"color": [250, 250, 0], "isthing": 1, "id": 10, "name": "yacht"},
    {"color": [250, 250, 100], "isthing": 1, "id": 11, "name": "water_scooter"}
]






def get_medatadata_coco_MaSTr1325():
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in COCO_CATEGORIES_MaSTr1325]
        thing_colors = [k["color"] for k in COCO_CATEGORIES_MaSTr1325]
        stuff_classes = [k["name"] for k in COCO_CATEGORIES_MaSTr1325]
        stuff_colors = [k["color"] for k in COCO_CATEGORIES_MaSTr1325]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(COCO_CATEGORIES_MaSTr1325):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
        return meta


def register_coco_panoptic_MaSTr1325(
    name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    #TODO jure popravit load_coco_panoptic_json funkcijo in ostale podatke
    """
        Register a "standard" version of COCO panoptic segmentation dataset named `name`.
        The dictionaries in this registered dataset follows detectron2's standard format.
        Hence it's called "standard".

        Args:
            name (str): the name that identifies a dataset,
                e.g. "coco_2017_train_panoptic"
            metadata (dict): extra metadata associated with this dataset.
            image_root (str): directory which contains all the images
            panoptic_root (str): directory which contains panoptic annotation images
            panoptic_json (str): path to the json panoptic annotation file
            sem_seg_root (none): not used, to be consistent with
                `register_coco_panoptic_separated`.
            instances_json (str): path to the json instance annotation file
        """

    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, metadata),
    )

    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type=panoptic_name,  # set this to "" if you don't want to evaluate
        #evaluator_type="",  # set this to "" if you don't want to evaluate
        ignore_label=255,     # category_id
        label_divisor=1000, # label_divisor: pove koliko instanc objekta je lahko na eni sliki.
        **metadata
    )


def register_MaSTr1325():

    # register_coco_panoptic_MaSTr1325(
    #     "coco_panoptic_train_LVIS_MaSTR1325",
    #     get_medatadata_coco_MaSTr1325(),
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/train/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/panoptic_train/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/annotations/panoptic_train.json",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_LVIS/annotations/instances_train.json"
    # )
    #
    # register_coco_panoptic_MaSTr1325(
    #     "coco_panoptic_train_MaSTR1325",
    #     get_medatadata_coco_MaSTr1325(),
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/train/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/panoptic_train/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/panoptic_train.json",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/instances_train.json"
    # )


    # register_coco_instances(
    #     "coco_train_instances_mastr1325",
    #     {},
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/instances_train.json",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/train/",
    # )

    register_coco_panoptic_MaSTr1325(
        "coco_train_panoptic_mastr1325",
        get_medatadata_coco_MaSTr1325(),
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/train/",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/panoptic_train/",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/panoptic.json",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/instances.json"
    )

    register_coco_panoptic_MaSTr1325(
        "coco_val_panoptic_mastr1325",
        get_medatadata_coco_MaSTr1325(),
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/val/",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/panoptic_val/",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/panoptic_val.json",
        "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco/annotations/instances_val.json"
    )



    # register_coco_panoptic_MaSTr1325(
    #     "coco_train_panoptic_mastr1325_evaluate",
    #     get_medatadata_coco_MaSTr1325(),
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/train/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/panoptic_train/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/annotations/panoptic_train.json",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/annotations/instances_train.json"
    # )
    #
    # register_coco_panoptic_MaSTr1325(
    #     "coco_val_panoptic_mastr1325_evaluate",
    #     get_medatadata_coco_MaSTr1325(),
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/val/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/panoptic_val/",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/annotations/panoptic_val.json",
    #     "d:/Segmentacija/panoptic-deeplab-master/tools_d2/datasets/coco_evaluate/annotations/instances_val.json"
    # )



def draw_panoptic_seg_predictions(
    self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7
):
    """
    Draw panoptic prediction results on an image.

    Args:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
            segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
        area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

    Returns:
        output (VisImage): image object with visualizations.
    """
    area_threshold = 0.1
    pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.img = self._create_grayscale_image(pred.non_empty_mask())

    # draw mask for all semantic segments first i.e. "stuff"
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        try:
            mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
        except AttributeError:
            mask_color = None

        text = self.metadata.stuff_classes[category_idx]
        self.draw_binary_mask(
            mask,
            color=mask_color,
            edge_color=_OFF_WHITE,
            text=text,
            alpha=alpha,
            area_threshold=area_threshold,
        )

    # draw mask for all instances second
    all_instances = list(pred.instance_masks())
    if len(all_instances) == 0:
        return self.output
    masks, sinfo = list(zip(*all_instances))
    category_ids = [x["category_id"] for x in sinfo]

    try:
        colors = [
            self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in category_ids
        ]
    except AttributeError:
        colors = None


    self.overlay_instances(masks=masks, assigned_colors=colors, alpha=alpha)

    return self.output



# ▲
#----------------------------------------------------------------------------------------
