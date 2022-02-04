
import os
import sys
import random
import shutil
import time
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

def convert_to_yolov5(target_dir, filename, info_dict, class_mapping):
    """
    Converting data to YOLO format
    Source: https://blog.paperspace.com/train-yolov5-custom-data
    (Modified)
    """

    with open(os.path.join(target_dir, filename+".txt"), "w") as annotation_file:
        # For each bounding box
        for b in info_dict["bboxes"]:
            try:
                class_id = class_mapping[b["class"]]
            except KeyError:
                print(f"Invalid Class {b['class']}, if this error occurs, all hell broke loose, take cover")
                sys.exit(1)

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
            annotation_file.write("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

def process_dataset(dataset_path, new_dataset_path, image_ext):

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
    if len(annotation_files) != len(image_files):
        print("WARNING: Amount of annotation files doesn't match amount of image files!")
        time.sleep(2)

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

    print("Class mapping:")
    for cl, clid in class_mapping.items():
        print(f"\t{cl}: {clid}")
    input("WARNING: Make sure when training an already trained network with a new dataset that class IDs are still the same, press ENTER to continue...")

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
    for a_dir in annotation_directories.values():
        os.makedirs(a_dir, exist_ok=True)

    data_split_point = int(len(annotation_files)/100*75)

    copy = True if input("Move or copy images to converted dataset? (copy, move):").strip().lower() in ["c","copy","cp"] else False
    for n, af in enumerate(annotation_files):
        filename = os.path.splitext(os.path.basename(af))[0]
        info = extract_info_from_xml(af)
        target_dir = annotation_directories['train' if n < data_split_point else 'val']
        convert_to_yolov5(target_dir, filename, info, class_mapping)
        print(f"# Saved converted annotation {n+1} to {af}")

        target_image_dir = images_directories['train' if n < data_split_point else 'val']
        image_name = filename+"."+image_ext

        if not filename in image_files:
            print(f"CRITICAL: Annotation for image {image_name} exists, image not found")
            time.sleep(1)
            continue

        if copy:
            shutil.copyfile(image_files[filename], os.path.join(target_image_dir, image_name))
        else:
            shutil.move(image_files[filename], target_image_dir)
        print(f"{'Copied' if copy else 'Moved'} {image_name} to {target_image_dir}")

    data_config_file = os.path.join(new_dataset_path, os.path.basename(dataset_path)+"_data.yaml")
    with open(data_config_file, "w") as df:
        data_conf=f"""train: '{images_directories['train']}'\nval: '{images_directories['val']}'\n\nnc: {len(classes)}\nnames: {classes}"""
        df.write(data_conf)
    print("\n\nCreated data config file in new dataset directory, check its contents and move this file to the yolov5/data directory")
    print("Data config file points to dataset location so when moving dataset or if you used relative paths, check data config file if paths are correct")
    print("Finished.")

if __name__ == "__main__": #Entry point

    dataset_path = input("Dataset path: ").strip()
    if not os.path.exists(dataset_path):
        print("FATAL: Dataset not found, check path and retry")
        sys.exit(1)

    new_dataset_path = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path)+"_convert")

    image_extension = input("Image file type extension (eg. jpg, png): ").strip().lower()

    print(f'Converted dataset will be stored at " {new_dataset_path} "')

    process_dataset(dataset_path, new_dataset_path, image_extension)
