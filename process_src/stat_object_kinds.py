import os
import json
import tqdm

# ('category', 'person', 'category ID', 1)
# ('category', 'rider', 'category ID', 2)
# ('category', 'car', 'category ID', 3)
# ('category', 'bus', 'category ID', 4)
# ('category', 'truck', 'category ID', 5)
# ('category', 'bike', 'category ID', 6)
# ('category', 'motor', 'category ID', 7)


def get_annotation_names(_annotation_root = './preprocessAnnotation/annotations'):
    return [os.path.join(_annotation_root, _file) for _file in os.listdir(_annotation_root) if _file.endswith('.json')]

def search_object_kinds(annotation_names):
    object_kinds = set()
    for _annotation_name in tqdm.tqdm(annotation_names,total=len(annotation_names)):
        _annotation = json.load(open(_annotation_name))
        labels = _annotation['labels']
        for label in labels:
            if label['objects']:
                for object in label['objects']:
                    object_kinds.add(str(('category',object["category"],"category ID",object["category ID"])))
    print('\n'.join(list(object_kinds)))

def main():
    annotation_names = get_annotation_names()
    search_object_kinds(annotation_names)

if __name__ == '__main__':
    main()