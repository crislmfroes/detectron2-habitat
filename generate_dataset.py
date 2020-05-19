from habitat_sim.utils.data.data_extractor import ImageExtractor
import json
import argparse
import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_sample(sample):
    img = sample["rgba"]
    semantic = sample["semantic"]

    arr = [img, semantic]
    titles = ["rgba", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

def generate_set(start_index:int, total_imgs:int, extractor:ImageExtractor, annotations_name:str):
    dataset = {}
    dataset["images"] = []
    dataset["annotations"] = []
    dataset["categories"] = []
    scene = extractor.sim.semantic_scene
    category_ids = {}
    id_counter = 1
    obj_category = {}
    for obj in scene.objects:
        if obj is not None and obj.category is not None:
            obj_category[obj.id] = obj.category.name()
            if obj.category.name() not in category_ids:
                category_ids[obj.category.name()] = id_counter
                id_counter += 1
                dataset["categories"].append({
                    "supercategory": obj.category.name(),
                    "id": category_ids[obj.category.name()],
                    "name": obj.category.name()
                })
    ann_id = 0
    for i in range(start_index + total_imgs):
        sample = extractor[i]
        rgba = np.array(sample["rgba"])
        semantic = np.array(sample["semantic"])
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(args.output_dir, "images", "{}.jpg".format(i)), bgr)
        dataset["images"].append({
            "license": 4,
            "file_name": "{}.jpg".format(i),
            "height": bgr.shape[0],
            "width": bgr.shape[1],
            "id": i
        })
        for obj_id, category_name in obj_category.items():
            n_id = int(obj_id.split('_')[-1])
            query = np.full(semantic.shape, n_id)
            mask = ((query == semantic)*255).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_poly = []
            bounding_rects = []
            for j, contour in enumerate(contours):
                if contour.shape[0] > 4:
                    poly = cv2.approxPolyDP(contour, 3, True)
                    bounding_rect = cv2.boundingRect(poly)
                    dataset["annotations"].append({
                        "segmentation": [contour.flatten().tolist()],
                        "area": bounding_rect[2]*bounding_rect[3],
                        "iscrowd": 0,
                        "image_id": i,
                        "bbox": bounding_rect,
                        "category_id": category_ids[category_name],
                        "id": ann_id
                    })
                    ann_id += 1
    with open(os.path.join(args.output_dir, "annotations", annotations_name), "w+") as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./dataset")
    parser.add_argument("--scene", type=str)
    parser.add_argument("--extraction_method", default="panorama")
    parser.add_argument("--split", default=0.8, type=float)
    args = parser.parse_args()
    try:
        os.mkdir(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, "annotations"))
        os.mkdir(os.path.join(args.output_dir, "images"))
    except BaseException as e:
        print(e)
    extractor = ImageExtractor(
        args.scene,
        labels=[0.0],
        img_size=(512, 512),
        output=["rgba", "semantic"],
        extraction_method="panorama"
    )
    generate_set(0, int(len(extractor)*args.split), extractor, "instances_train.json")
    generate_set(int(len(extractor)*args.split), int(len(extractor)*(1 - args.split)), extractor, "instances_val.json")
        
