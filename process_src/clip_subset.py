import os
import sys
import pathlib
import tqdm
import json
import argparse

 

def get_subsets(frame_path):
    print('getting subset names...')
    subset_names = os.listdir(frame_path)
    subset_names = [name for name in subset_names if os.path.isdir(os.path.join(frame_path, name))]
    true_names = []
    for name in subset_names:
        default_img = os.path.join(frame_path, name,'images','000000.jpg')
        if os.path.exists(default_img):
            true_names.append(name)
        else:
            print(default_img)
            print('not exist! need further check')
    print('Done!')
    return subset_names

def process_annotation_jsons(json_root, subset_names):
    """
    json_root: path to json files
    subset_names: list of subset names
    return: list of json paths
    """
    print('processing annotation json files...')
    json_paths = list(os.listdir(json_root))
    json_paths = [path for path in json_paths if path.endswith('.json')]
    final_json_paths = []
    for json_path in tqdm.tqdm(json_paths,total=len(json_paths)):
        if json_path.replace('.json','') in subset_names:
            final_json_paths.append(os.path.join(json_root, json_path))
    print('Done!')
    return final_json_paths

def process_meta_data(meta_data_path, subset_names):
    """
    metadata should be a json file
    """
    print('processing meta data...')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    final_meta_data = {}
    for subset_name in subset_names:
        if subset_name not in meta_data: continue
        final_meta_data[subset_name] = meta_data[subset_name]
        # inverse problem
    print('Done!')
    return final_meta_data

def process_split_txt(split_txt_path, subset_names):
    """
    split txt should be a txt file
    """
    print('processing split txt...')
    with open(split_txt_path, 'r') as f:
        split_txt = f.readlines()
    final_split_txt = []
    for line in split_txt:
        line = line.strip()
        if line in subset_names:
            final_split_txt.append(line)
    print('Done!')
    return final_split_txt

def main(args):
    subset_names = get_subsets(args.frame_path) # get all subset names
    train_json_paths = process_annotation_jsons(args.json_root, subset_names) # with all annotation json paths to copy
    meta_data_train_path, meta_data_val_path = os.path.join(args.meta_data_path, 'metadata_train.json'), os.path.join(args.meta_data_path, 'metadata_val.json')
    train_split_txt_path, val_split_txt_path = os.path.join(args.split_txt_path, 'train_split.txt'), os.path.join(args.split_txt_path, 'val_split.txt')
    train_meta_data = process_meta_data(meta_data_train_path, subset_names)
    val_meta_data = process_meta_data(meta_data_val_path, subset_names)
    train_split_txt = process_split_txt(train_split_txt_path, subset_names)
    val_split_txt = process_split_txt(val_split_txt_path, subset_names)
    tgt_folder = os.path.join(args.processed_root)
    annotation_folder = os.path.join(tgt_folder, 'annotations')
    pathlib.Path(annotation_folder).mkdir(parents=True, exist_ok=True)
    # load json files
    for json_path in train_json_paths:
        json_name = os.path.basename(json_path)
        json_tgt_path = os.path.join(annotation_folder, json_name)
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        with open(json_tgt_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    # load meta data
    meta_data_train_tgt_path = os.path.join(tgt_folder, 'metadata_train.json')
    with open(meta_data_train_tgt_path, 'w') as f:
        json.dump(train_meta_data, f, indent=4)

    meta_data_val_tgt_path = os.path.join(tgt_folder, 'metadata_val.json')
    with open(meta_data_val_tgt_path, 'w') as f:
        json.dump(val_meta_data, f, indent=4)

    # load split txt
    train_split_txt_tgt_path = os.path.join(tgt_folder, 'train_split.txt')
    with open(train_split_txt_tgt_path, 'w') as f:
        for line in train_split_txt:
            f.write(line + '\n')
    
    val_split_txt_tgt_path = os.path.join(tgt_folder, 'val_split.txt')
    with open(val_split_txt_tgt_path, 'w') as f:
        for line in val_split_txt:
            f.write(line + '\n')
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_path', type=str, default='./frames')
    parser.add_argument('--json_root', type=str, default='./preprocessAnnotation/annotations')
    parser.add_argument('--meta_data_path', type=str, default='./preprocessAnnotation')
    parser.add_argument('--split_txt_path', type=str, default='./preprocessAnnotation')
    parser.add_argument('--processed_root', type=str, default='./processedAnnotations')
    args = parser.parse_args()
    main(args)




