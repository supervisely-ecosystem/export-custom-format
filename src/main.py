import os
import supervisely as sly
from supervisely.io.json import dump_json_file
from supervisely.io.fs import silent_remove, remove_dir, copy_file
from os.path import join

from dotenv import load_dotenv


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


STORAGE_DIR = sly.app.get_data_dir()
ann_file = "labels.json"


def download_project(api, project_name, project_id):
    dataset_ids = [ds.id for ds in api.dataset.get_list(project_id)]
    sly.logger.info("DOWNLOAD_PROJECT", extra={"title": project_name})
    dest_dir = join(STORAGE_DIR, f"{project_id}_{project_name}")
    sly.download_project(api, project_id, dest_dir, dataset_ids=dataset_ids, log_progress=True)
    sly.logger.info(
        "Project {!r} has been successfully downloaded. Starting convert to rectangles.".format(
            project_name
        )
    )
    return dest_dir


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):

        api = sly.Api.from_env()
        project_info = api.project.get_info_by_id(id=context.project_id)
        project_dir = download_project(api, project_info.name, project_info.id)

        project = sly.Project(directory=project_dir, mode=sly.OpenMode.READ)

        if context.dataset_id is not None:
            ds_name = api.dataset.get_info_by_id(id=context.dataset_id).name
            converting_data = [
                sly.Dataset(
                    join(project_dir, ds_name),
                    mode=sly.OpenMode.READ,
                )
            ]
        else:
            converting_data = project

        for dataset in converting_data:
            result_anns = {}
            ds_progress = sly.Progress(
                "Processing dataset: {!r}/{!r}".format(project.name, dataset.name),
                total_cnt=len(dataset),
            )
            for item_name in dataset:
                curr_labels = []
                item_paths = dataset.get_item_paths(item_name)
                new_img_path = join(project_dir, dataset.name, item_name)
                copy_file(item_paths.img_path, new_img_path)
                ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
                for label in ann.labels:
                    name = label.obj_class.name
                    rect_geom = label.geometry.convert(sly.Rectangle)
                    for curr_geom in rect_geom:
                        curr_labels.append(
                            {
                                "class_name": name,
                                "coordinates": [
                                    curr_geom.top,
                                    curr_geom.left,
                                    curr_geom.bottom,
                                    curr_geom.right,
                                ],
                            }
                        )

                result_anns[item_name] = curr_labels
            dump_json_file(result_anns, join(dataset.directory, ann_file), indent=2)

            remove_dir(dataset.img_dir)
            remove_dir(dataset.ann_dir)

            ds_progress.iter_done_report()

        silent_remove(join(project_dir, "meta.json"))
        sly.logger.info("Finished converting.".format(project_info.name))

        full_archive_name = str(project_info.id) + "_" + project_info.name + ".tar"
        result_archive = join(STORAGE_DIR, full_archive_name)
        sly.fs.archive_directory(project_dir, result_archive)
        sly.logger.info("Result directory is archived")

        return result_archive


app = MyExport()
app.run()