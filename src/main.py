import os

import supervisely as sly
import json
from os.path import join

from dotenv import load_dotenv

# load ENV variables for debug, has no effect in production
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


STORAGE_DIR = sly.app.get_data_dir()
ANN_FILE = "labels.json"


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        # create api object to communicate with Supervisely Server
        api = sly.Api.from_env()

        # get project info from server
        project_info = api.project.get_info_by_id(id=context.project_id)

        # make project directory path
        data_dir = join(STORAGE_DIR, f"{project_info.id}_{project_info.name}")

        # get project meta
        meta_json = api.project.get_meta(id=context.project_id)
        project_meta = sly.ProjectMeta.from_json(meta_json)

        # get datasets info from server
        dataset_infos = api.dataset.get_list(project_id=context.project_id)

        # iterate over datasets in project
        for dataset in dataset_infos:
            result_anns = {}

            # create Progress object to track progress
            ds_progress = sly.Progress(
                f"Processing dataset: '{dataset.name}'",
                total_cnt=dataset.images_count,
            )

            # get dataset images info
            images = api.image.get_list(dataset.id)

            # iterate over images in dataset
            for image in images:
                labels = []

                # create path for eaach image and download it from server
                image_path = join(data_dir, dataset.name, image.name)
                api.image.download(image.id, image_path)

                # download annotation for current image
                ann_json = api.annotation.download_json(image.id)
                ann = sly.Annotation.from_json(ann_json, project_meta)

                # iterate over labels in current annotation
                for label in ann.labels:
                    # get obj class name
                    name = label.obj_class.name

                    # get bounding box coordinates for label
                    bbox = label.geometry.to_bbox()
                    labels.append(
                        {
                            "class_name": name,
                            "coordinates": [
                                bbox.top,
                                bbox.left,
                                bbox.bottom,
                                bbox.right,
                            ],
                        }
                    )

                result_anns[image.name] = labels

                # increment the current progress counter by 1
                ds_progress.iter_done_report()

            # create JSON annotation in new format
            filename = join(data_dir, dataset.name, ANN_FILE)
            with open(filename, "w") as fout:
                json.dump(result_anns, fout, indent=2)

        return data_dir


app = MyExport()
app.run()
