
import torch
import os
import json
import PIL

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

new_training_images_dir = "testimg"
img_extension = ".jpg"

object_annotation = """
    <object>
		<name>{name}</name>
		<bndbox>
			<xmin>{dmin[0]}</xmin>
			<ymin>{dmin[1]}</ymin>
			<xmax>{dmax[0]}</xmax>
			<ymax>{dmax[1]}</ymax>
		</bndbox>
	</object>"""

annofile = """<annotation>
	<folder>{fldr}</folder>
	<filename>{img}</filename>
	<path>{img_path}</path>
	<size>
		<width>{img_size[0]}</width>
		<height>{img_size[1]}</height>
		<depth>3</depth>
	</size>
{objects}
</annotation>"""

for img in os.listdir(new_training_images_dir):
    if img.endswith(img_extension):
        img_path = os.path.abspath(os.path.join(new_training_images_dir, img))
        fldr = os.path.dirname(img_path)

        detections = model(img_path)
        detections_list = json.loads(detections.pandas().xyxy[0].to_json(orient="records")) # load detections from json

        img_size = PIL.Image.open(img_path).size

        objs = []
        for det in detections_list:
            dmin = list(map(int, (det["xmin"],det["ymin"])))
            dmax = list(map(int, (det["xmax"],det["ymax"])))
            objs.append(object_annotation.format(name=det["name"], dmin=dmin, dmax=dmax))

        with open(os.path.join(new_training_images_dir, os.path.splitext(img)[0]+".xml"), "w") as af:
            af.write(annofile.format(fldr=fldr, img=img, img_path=img_path, img_size=img_size, objects="\n".join(objs)))
        print(f"Wrote annotation for {img}")