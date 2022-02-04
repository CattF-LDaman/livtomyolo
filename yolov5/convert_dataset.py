
import os
import sys
import random
import xml.etree.ElementTree

def extract_info_from_xml(xml_file):
    """
    Function to get necessary data from XML Annotation
    Source: https://blog.paperspace.com/train-yolov5-custom-data
    """
    root = xml.etree.ElementTree.parse(xml_file).getroot()

    # Initialise the info dict
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict['filename'] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict['image_size'] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)

    return info_dict

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect":2}

def convert_to_yolov5(info_dict, class_mapping):
    """
    Converting data to YOLO format
    Source: https://blog.paperspace.com/train-yolov5-custom-data
    """
    print_buffer = []

    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print(f"Invalid Class {b['class']}. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width    /= image_w
        b_height   /= image_h

        #Write the bbox details to the file
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save
    save_file_name = os.path.join("facemask/annotations", info_dict["filename"].replace("png", "txt"))

    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))

def structure_dataset(dataset_path, new_dataset_path, image_ext):

    annotation_files = []
    image_files = {}

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            file_ext = file_ext[1:].lower() # Remove dot and eliminate case-sensitivity
            if file_ext == image_ext:
                image_files[file_name] = (os.path.join(root, file))
                print(f'# Found image {file}')
            elif file_ext == "xml":
                annotation_files.append(os.path.join(root, file))
                print(f'# Found annotation {file}')

    random.shuffle(annotation_files)
    print(f'Found {len(annotation_files)} annotation files and {len(image_files)} image files')

    print("Scouring annotation files for classes...")
    #Get all classes present in annotation files
    classes = []
    for annotation_file in annotation_files:
        info = extract_info_from_xml(annotation_file)
        for bbox in info['bboxes']:
            if bbox['class'] not in classes:
                classes.append(bbox["class"])
    classes.sort() # Avoid change in class ID when converting same dataset twice
    class_mapping = {oclass:i for i, oclass in enumerate(classes)} # Assign numerical ID to class name as YOLO requires
    print(f'Found {len(classes)} classes: {", ".join(classes)}')

    print('Creating directory structure for converted dataset')
    images_directories = {
        'train': os.path.join(new_dataset_path, 'images', 'train'),
        'val': os.path.join(new_dataset_path, 'images', 'val'),
    }
    annotation_directories = {
        'train': os.path.join(new_dataset_path, 'labels', 'train'),
        'val': os.path.join(new_dataset_path, 'labels', 'val'),
    }
    for i_dir in images_directories.values():
        os.makedirs(i_dir, exist_ok=True)
    for a_dir in images_directories.values():
        os.makedirs(a_dir, exist_ok=True)

    #split and distribute files

if __name__ == "__main__":

    dataset_path = input("Dataset path: ").strip()
    new_dataset_path = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path)+"_convert")

    image_extension = input("Image file type extension (eg. jpg, png): ").strip().lower()

    print(f'Converted dataset will be stored at " {new_dataset_path} "')

    structure_dataset(dataset_path, new_dataset_path, image_extension)

    # Convert and save the annotations
    #for ann in annotations:
    #    info_dict = extract_info_from_xml(ann)
    #    convert_to_yolov5(info_dict)
    #annotations = [os.path.join('facemask/annotations', x) for x in os.listdir('facemask/annotations') if x[-3:] == "txt"]
