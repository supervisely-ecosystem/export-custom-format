# export-custom-format

Example app that demonstrates how to make export apps.

In this example export app converts selected images project or dataset to the following format and prepares downloadable .tar archive:

- original images
- annotations in json format

Output example:

```text
<id_project_name>.tar
- ds0
  - image_1.jpg
  - image_2.jpg
  - labels.json
- ds1
  - image_1.jpg
  - image_2.jpg
  - labels.json
```

Labels.json:

Contain bounding box coordinates(top, left, right, bottom) of all objects in project or dataset

```text
{
"image_1.jpg": [
          {
               "class_name": "cat",
               "coordinates": [top, left, right, bottom]
           },
            {
               "class_name": "cat",
               "coordinates": [top, left, right, bottom]
           },
           {
               "class_name": "dog",
               "coordinates": [top, left, right, bottom]
           },
     ]
}

```