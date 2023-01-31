import os

import supervisely as sly
from supervisely.io.json import dump_json_file
from supervisely.io.fs import silent_remove, remove_dir, copy_file
from os.path import join

from dotenv import load_dotenv

# load ENV variables for debug, has no effect in production
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


STORAGE_DIR = sly.app.get_data_dir()
ANN_FILE = "labels.json"


def download_project(api, project_name, project_id, dataset_id):
    # check if dataset id is specified
    dataset_ids = [dataset_id] if dataset_id is not None else None

    # make project directory path and download project
    project_dir = join(STORAGE_DIR, f"{project_id}_{project_name}")
    sly.download_project(api, project_id, project_dir, dataset_ids=dataset_ids, log_progress=True)
    sly.logger.info(f"Project: '{project_name}' has been successfully downloaded.")
    return project_dir


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        # create api object to communicate with Supervisely Server
        api = sly.Api.from_env()

        # get project info from server
        project_info = api.project.get_info_by_id(id=context.project_id)

        # download project
        project_dir = download_project(api, project_info.name, project_info.id, context.dataset_id)

        # read local Supervisely project
        project = sly.Project(directory=project_dir, mode=sly.OpenMode.READ)

        # iterate over datasets in project
        for dataset in project:
            result_anns = {}

            # create Progress object to track progress
            ds_progress = sly.Progress(
                f"Processing dataset: '{dataset.name}'",
                total_cnt=len(dataset),
            )
            # iterate over images in dataset
            for item_name in dataset:
                labels = []

                # get image and annotation path for item
                img_path, ann_path = dataset.get_item_paths(item_name)

                # copy image to new path for export in new format
                new_item_path = join(project_dir, dataset.name, item_name)
                copy_file(img_path, new_item_path)

                # download annotation for current image
                ann = sly.Annotation.load_json_file(ann_path, project.meta)

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

                result_anns[item_name] = labels

            # create JSON annotation in new format
            dump_json_file(result_anns, join(dataset.directory, ANN_FILE), indent=2)

            # remove 'img' and 'ann' folders in Supervisely format
            remove_dir(dataset.img_dir)
            remove_dir(dataset.ann_dir)

            # increment the current progress counter by 1
            ds_progress.iter_done_report()

        # remove project meta file in Supervisely format from project
        silent_remove(join(project_dir, "meta.json"))

        # archive project directory and return path to archive
        # sly.app.Export will use it to upload result in Team Files
        archive_name = f"{project_info.id}_{project_info.name}.tar"
        archive_path = join(STORAGE_DIR, archive_name)
        sly.fs.archive_directory(project_dir, archive_path)
        sly.logger.info("Result directory is archived")
        return archive_path


app = MyExport()
app.run()
