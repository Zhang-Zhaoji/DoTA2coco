import shutil
import os
import json
import tqdm
import pathlib
import numpy as np

randomly_sample = 50 # 1 in {sample} as test

_global_image_id_counter = 0
_global_annotation_id_counter = 0

def generate_template_image_dict(image_id:int, width:int, height:int, file_name:str):
    """
    implement further processing codes later
    """
    return {
    "id": image_id,
    "width": width,
    "height": height,
    "file_name": file_name
    }

def generate_template_annotation_dict(id:int, image_id:int, category_id:int, segmentation, area, bbox, iscrowd):
    """
    implement further processing codes later
    """
    return {
    "id": id,
    "image_id": image_id,
    "category_id": category_id,
    "segmentation": segmentation,
    "area": area,
    "bbox": bbox,
    "iscrowd": iscrowd
    }

def generate_template_category_dict(supercategory, id, name):
    """
    implement further processing codes later
    """
    return {
        "supercategory": supercategory,
        "id": id,
        "name": name
        }

def generate_dota_categories():
    return [
        generate_template_category_dict('person', 1, 'person'),
        generate_template_category_dict('person', 2, 'rider'),
        generate_template_category_dict("vehicle",3, 'car'),
        generate_template_category_dict("vehicle",4, 'bus'),
        generate_template_category_dict("vehicle",5, 'truck'),
        generate_template_category_dict("vehicle",6, 'bike'),
        generate_template_category_dict("vehicle",7, 'motor')
    ]

def generate_coco_style_json():
    tgt_json = {}
    
    tgt_json['info'] = {
        "description": "DoTA Datasets",
        "url": "Who knows",
        "version": "114.514",
        "year": 2019,
        "contributor": "DoTA Team",
        "date_created": "2019/01/01"
    }

    tgt_json['licenses'] = [
        {
            "id": 1,
            "name": "Nobody knows and nowbody care what licence they use",
            "url": ""
        }
    ]
    tgt_json['images'] = []
    tgt_json['annotations'] = []
    tgt_json['categories'] = []
    return tgt_json
def copy_and_modify(annotation_file, annotation_root, image_root, folders, new_root, image_id_map):
    global _global_image_id_counter, _global_annotation_id_counter # Ensure these are accessible
    # Images used to be stored in the folders representing videos.
    # We now store them in the same but another root directory.
    
    # we need to think about new path of imgs
    # first thing is the so called root.
    # 1. we need to give them a new name, the name of folder_name, image_index might be a good choice.
    # 2. we also need to find the path of the annotation file and corresponding descriptions.
    # 3. modify annotation file.
    # LGTM, lets do it.

    for folder in tqdm.tqdm(folders, total=len(folders)):
        # folder is the video name
        images = os.listdir(os.path.join(image_root,folder))
        annotation_folder = folder.split('\\')[0] if '\\' in folder else folder
        image_names = [os.path.join(new_root,f'{annotation_folder}_{image_name}') for image_name in images]
        # load all folder names, generate corresponding new_img names and original_img_paths
        # open original annotation file and modify annotation file
        
        video_information = json.load(open(os.path.join(annotation_root,annotation_folder+'.json'),'r'))
        original_img_name_to_label_idx = {}

        for i, label_entry in enumerate(video_information['labels']):
            # Assuming label_entry['image_path'] is something like "frames/video_id/images/000052.jpg"
            # We need to extract just "000052.jpg"
            original_img_filename_from_label = os.path.basename(label_entry['image_path'])
            original_img_name_to_label_idx[original_img_filename_from_label] = i
        
        for idx, img_name in enumerate(images[:min(len(images),len(video_information['labels']))]):
            if np.random.uniform(0,1,1) > 1/randomly_sample:
                continue
            # Increment global image ID counter and use it
            current_image_id = _global_image_id_counter
            _global_image_id_counter += 1

            img_id = int(img_name.split('.')[0]) - 1 # for generated saliency map
            # print(img_id, len(images), len(video_information['labels']), video_information['num_frames'],annotation_folder)
            img_annotation = video_information['labels'][img_id]
            # example annotation format:
            # {
            #     "frame_id": 52,
            #     "image_path": "frames/0qfbmt4G8Rw_001060/images/000052.jpg",
            #     "accident_id": 8,
            #     "accident_name": "leave_to_right",
            #     "objects": [
            #         {
            #             "obj_track_id": 239,
            #             "bbox": [
            #                 622.551724137931,
            #                 451.8620689655172,
            #                 673.448275862069,
            #                 477.9310344827586
            #             ],
            #             "category": "car",
            #             "category ID": 3,
            #             "trunc": false
            #         }
            #     ]
            # },
            pre_img_path = os.path.join(image_root, folder, img_name)
            new_img_path = image_names[idx]

            image_id_map[str(current_image_id)] = {'original_folder': folder, 'original_filename': img_name, 'new_path': new_img_path}

            image_dict = generate_template_image_dict(current_image_id,1280,720,new_img_path)
            annotation_file['images'].append(image_dict)
            shutil.copy(pre_img_path, new_img_path)
            objects = img_annotation['objects']
            for obj_idx, object in enumerate(objects):
                # Increment global annotation ID counter and use it
                current_annotation_id = _global_annotation_id_counter
                _global_annotation_id_counter += 1

                bbox = object['bbox']
                bbox = [bbox[0],bbox[1],-bbox[0]+bbox[2],-bbox[1]+bbox[3]]
                category_id = object["category ID"]
                annotation_dict = generate_template_annotation_dict(id = current_annotation_id,#f'{annotation_folder}_{img_id:06d}_{obj_idx}',
                                                                image_id = current_image_id,# f'{annotation_folder}_{img_id:06d}',
                                                                category_id=category_id,
                                                                segmentation=[bbox],
                                                                area=bbox[2]*bbox[3],
                                                                bbox=bbox,
                                                                iscrowd=0)
                annotation_file['annotations'].append(annotation_dict)
    print('Done!')

def save_annotation_Files(annotation_file, output_file):
    with open(output_file, 'w') as f:
        json.dump(annotation_file, f)
        print('Saved:', output_file)
            
def prepare_dataset(_root = './', dtfolder_name = 'DoTA', anno_dir = './processedAnnotations'):
    global _global_image_id_counter, _global_annotation_id_counter # Access global counters
    
    # Reset counters for each dataset preparation run to ensure unique IDs across runs
    _global_image_id_counter = 0
    _global_annotation_id_counter = 0
    
    # load splits ids
    print('Loading splits...')
    train_split_path = os.path.join(anno_dir, 'train_split.txt')
    val_split_path = os.path.join(anno_dir, 'val_split.txt')
    train_split_txt = open(train_split_path, 'r').readlines()
    val_split_txt = open(val_split_path, 'r').readlines()
    train_split_txt = [line.strip() for line in train_split_txt]
    val_split_txt = [line.strip() for line in val_split_txt]
    # create essential folders
    print('Creating essential folders...')
    dataset_root = os.path.join(_root, dtfolder_name)
    pathlib.Path(dataset_root).mkdir(parents=True, exist_ok=True)
    train_dataset_root = os.path.join(dataset_root, 'train')
    pathlib.Path(train_dataset_root).mkdir(parents=True, exist_ok=True)
    val_dataset_root = os.path.join(dataset_root, 'val')
    pathlib.Path(val_dataset_root).mkdir(parents=True, exist_ok=True)
    annotation_folder = os.path.join(dataset_root, 'annotations')
    pathlib.Path(annotation_folder).mkdir(parents=True, exist_ok=True)
    # convert into a coco dataset 
    print('Converting into coco dataset...')
    train_annotation = generate_coco_style_json()
    train_annotation['categories'] = generate_dota_categories()

    val_annotation = generate_coco_style_json()
    val_annotation['categories'] = generate_dota_categories()

    train_image_id_map = {}
    val_image_id_map = {}

    # move images, first trains
    # txt file contain the folder names of each image, we need to move them into the corresponding folder
    # perhaps we should write a new function to do this
    # train:
    print("Copying and modifying annotations for training set")
    copy_and_modify(annotation_file=train_annotation, 
                    annotation_root=os.path.join(anno_dir,'annotations'), 
                    image_root='saliency_imgs', 
                    folders=[os.path.join(train_split_txt[i]) for i in range(len(train_split_txt))],
                    new_root=train_dataset_root,
                    image_id_map = train_image_id_map)
    # valid
    print("Copying and modifying annotations for valid set")
    copy_and_modify(annotation_file=val_annotation, 
                    annotation_root=os.path.join(anno_dir,'annotations'), 
                    image_root='saliency_imgs', 
                    folders=[os.path.join(val_split_txt[i]) for i in range(len(val_split_txt))],
                    new_root=val_dataset_root,
                    image_id_map = val_image_id_map)
    # save annotation_files
    print("Save Annotations")
    save_annotation_Files(train_annotation, os.path.join(annotation_folder, 'train.json'))
    save_annotation_Files(val_annotation, os.path.join(annotation_folder, 'val.json'))
    print("Done")

    print("Saving Image ID Maps...")
    save_annotation_Files(train_image_id_map, os.path.join(annotation_folder, 'train_image_id_map.json'))
    save_annotation_Files(val_image_id_map, os.path.join(annotation_folder, 'val_image_id_map.json'))
    print("Done")

def main():
    prepare_dataset()

if __name__ == '__main__':
    main()
