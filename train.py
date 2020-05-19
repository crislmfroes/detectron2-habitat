from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog
import argparse
import os


class CustomTrainer(DefaultTrainer):
    
    def __init__(self, cfg:CfgNode):
        super().__init__(cfg)
    
    @classmethod
    def build_evaluator(self, cfg:CfgNode, dataset_name:str):
        return COCOEvaluator("custom_dataset_val", cfg, True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./dataset")
    parser.add_argument('--weights_path', default="")
    parser.add_argument('--output_path', default="./output")
    parser.add_argument('--max_iterations', default=3000, type=int)
    parser.add_argument('--config_file', default="detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        os.mkdir(args.output_path)
    except FileExistsError:
        pass
    annotations_dir = os.path.join(args.dataset_path, "annotations")
    images_dir = os.path.join(args.dataset_path, "images")
    register_coco_instances("custom_dataset_train", {}, os.path.join(annotations_dir, "instances_train.json"), images_dir)
    register_coco_instances("custom_dataset_val", {}, os.path.join(annotations_dir, "instances_val.json"), images_dir)
    train_json = load_coco_json(os.path.join(annotations_dir, "instances_train.json"), image_root=images_dir, dataset_name="custom_dataset_train")
    val_json = load_coco_json(os.path.join(annotations_dir, "instances_val.json"), image_root=images_dir, dataset_name="custom_dataset_val")
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN = ("custom_dataset_train",)
    cfg.DATASETS.TEST = ("custom_dataset_val",)
    cfg.TEST.EVAL_PERIOD = 300
    cfg.TEST.EXPECTED_RESULTS = [['bbox', 'AP', 38.5, 0.2]]
    cfg.SOLVER.CHECKPOINT_PERIOD = 300
    #cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = args.max_iterations
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" if not args.weights_path else args.weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("custom_dataset_train").thing_classes)
    cfg.OUTPUT_DIR = args.output_path
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MIN_SIZE_TRAIN = (128,)
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.INPUT.MIN_SIZE_TEST = 128
    cfg.merge_from_list(args.opts)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True if args.weights_path else False)
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass