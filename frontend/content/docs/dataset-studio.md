# Dataset Studio

Dataset Studio is the core data preparation and management tool within Onestep AI Platform. It supports importing, labeling, annotating, splitting, preprocessing, and versioning datasets without rewriting original files.

![Dataset List](/brand/2_dataset_list.png)

## Image Tasks

Dataset Studio provides robust support for Computer Vision datasets across Image Classification, Object Detection, and Segmentation tasks.

![Dataset Studio Image Segmentation](/brand/3_dataset_studio_image_segmentation_task.png)

- **Annotation Tools:** A dedicated Annotation editor provides tools for select/move, bounding box drawing, and polygon segmentation drawing.
- **Classification:** Upload classification images with labels or perform bulk relabeling on selected images.

![Dataset Studio Image Classification](/brand/4_dataset_studio_image_clasification_task.png)

## Dataset Management Features

- **Unassigned Inbox:** New uploads default to an `unassigned` inbox. Proceeding with these will automatically split them into train/valid/test datasets with a default 70/20/10 proportion.
- **Exploratory Data Analysis (EDA):** A dedicated EDA tab gives you a real-time health check on your dataset, showing split counts, class balance, geometry constraints, and warnings.
- **Preprocessing & Augmentation:** Apply allowlisted preprocessing (e.g., resize, normalize) and augmentations. The studio stores these materialized copies under versioned artifacts, ensuring original uploads are never altered.
- **Format Agnostic Imports:** Supports YOLO and COCO formats natively for images, while keeping reference datasets completely read-only.
