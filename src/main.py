import os
import supervisely as sly
from supervisely.io.json import dump_json_file
from supervisely.io.fs import silent_remove, remove_dir, copy_file
from os.path import join

from dotenv import load_dotenv

# load ENV variables for debug, has no effect in production
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


STORAGE_DIR = sly.app.get_data_dir()
ANN_FILE = "labels.json"


def download_project(api, project_name, project_id, dataset_id):
    dataset_ids = None
    if dataset_id is not None:
        dataset_ids = [dataset_id]
    sly.logger.info("DOWNLOAD_PROJECT", extra={"title": project_name})
    dest_dir = join(STORAGE_DIR, f"{project_id}_{project_name}")
    sly.download_project(api, project_id, dest_dir, dataset_ids=dataset_ids, log_progress=True)
    sly.logger.info(
        f"Project '{project_name}' has been successfully downloaded. Starting convert to rectangles."
    )
    return dest_dir


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        # create api object to communicate with Supervisely Server
        api = sly.Api.from_env()
        # get project information from context
        project_info = api.project.get_info_by_id(id=context.project_id)
        # download project in remote container or local hard drive, if export is performed from dataset context menu, project will be download with only current dataset
        project_dir = download_project(api, project_info.name, project_info.id, context.dataset_id)

        # start converting data for export
        project = sly.Project(directory=project_dir, mode=sly.OpenMode.READ)
        # loop from all project datasets, if export is performed from dataset context menu, loop will contain only 1 item
        for dataset in project:
            result_anns = {}
            ds_progress = sly.Progress(
                f"Processing dataset: '{dataset.name}'",
                total_cnt=len(dataset),
            )
            # loop from all images in dataset
            for item_name in dataset:
                curr_labels = []
                item_paths = dataset.get_item_paths(item_name)
                new_img_path = join(project_dir, dataset.name, item_name)
                # copy image for save result export in new format
                copy_file(item_paths.img_path, new_img_path)
                # download annotation for current image
                ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
                # loop from labels in current annotation
                for label in ann.labels:
                    name = label.obj_class.name
                    # get bounding box coordinates for label
                    rect_geom = label.geometry.to_bbox()
                    curr_labels.append(
                        {
                            "class_name": name,
                            "coordinates": [
                                rect_geom.top,
                                rect_geom.left,
                                rect_geom.bottom,
                                rect_geom.right,
                            ],
                        }
                    )

                result_anns[item_name] = curr_labels
            # create JSON annotation in new format
            dump_json_file(result_anns, join(dataset.directory, ANN_FILE), indent=2)

            # remove folders in old Supervisely format
            remove_dir(dataset.img_dir)
            remove_dir(dataset.ann_dir)

            ds_progress.iter_done_report()

        # remove project meta file in old Supervisely format
        silent_remove(join(project_dir, "meta.json"))
        sly.logger.info("Finished converting.".format(project_info.name))
        # create result archive name
        full_archive_name = f"{project_info.id}_{project_info.name}.tar"
        # create resutl archive path, sly.app.Export will use it to upload result in Team Files
        result_archive_path = join(STORAGE_DIR, full_archive_name)
        # create result archive
        sly.fs.archive_directory(project_dir, result_archive_path)
        sly.logger.info("Result directory is archived")

        return result_archive_path


app = MyExport()
app.run()
