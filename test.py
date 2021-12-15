import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import json
import cv2
from detectron2.structures import BoxMode

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
print(torch.__version__, torch.cuda.is_available())
setup_logger()
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train",
                        {},
                        "./datasets/coco/annotations/instances_train2017.json",
                        "./datasets/coco/train2017/")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
import pycocotools.mask as mask_util


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


# SOLVER
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.LR_POLICY = "steps_with_decay"
cfg.SOLVER.STEPS = [5000, 7000, 9000]
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.WARMUP_FACTOR = 0.01
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.GAMMA = 0.1

# Train
# One size for each in feature map
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
# Three aspect ratios (same for all in feature maps)
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1, 2, 5]]
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
cfg.TEST.DETECTIONS_PER_IMAGE = 2000
cfg.INPUT.CROP.SIZE = [1.0, 1.0]
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = './detectron2_results/'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_submit.pth")
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get("my_dataset_train")
inpath = "./datasets/coco/test2017/"
test_dir = os.listdir(inpath)
test_dir

for test_name in test_dir:
    im = cv2.imread(inpath+test_name)
    outputs = predictor(im)
    v = Visualizer(
                    im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE_BW
                    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("./detectron2_results/"+test_name, v.get_image()[:, :, ::-1])

d = {
    "TCGA-A7-A13E-01Z-00-DX1.png" : 1,
    "TCGA-50-5931-01Z-00-DX1.png" : 2,
    "TCGA-G2-A2EK-01A-02-TSB.png" : 3,
    "TCGA-AY-A8YK-01A-01-TS1.png" : 4,
    "TCGA-G9-6336-01Z-00-DX1.png" : 5,
    "TCGA-G9-6348-01Z-00-DX1.png" : 6
}

# test file dir
inpath = "./datasets/coco/test2017/"
test_dir = os.listdir(inpath)
Img_ID = []
EncodedPixels = []
conv = lambda l: ' '.join(map(str, l))

data = []
for name in test_dir:

        image = cv2.imread(inpath + name)
        outputs = predictor(image)

        image_id = d[name]
        print(image_id)

        masks = outputs["instances"].to('cpu').pred_masks.cpu().numpy()

        num = len(masks)
        scores = outputs["instances"].to('cpu').scores.cpu().numpy()
        boxes = outputs["instances"].to('cpu').pred_boxes.tensor.cpu().numpy()

        # x1, y1, x2, y2
        # -- >
        # [x, y, width, height],

        for i in range(num):
            a = {}

            a["image_id"] = image_id
            a["score"] = float(scores[i])
            a["category_id"] = 1

            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2] - box[0]
            height = box[3] - box[1]

            a["bbox"] = (tuple((float(left), float(top), float(width), float(height))))

            segmentation = {}
            segmentation["size"] = [1000, 1000]
            mask = masks[i]

            segmentation["counts"] = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]["counts"].decode('utf-8')
            a["segmentation"] = segmentation
            data.append(a)

ret = json.dumps(data, indent=4)

print(len(data))
with open("./detectron2_results/answer.json", 'w') as fp:
    fp.write(ret)
