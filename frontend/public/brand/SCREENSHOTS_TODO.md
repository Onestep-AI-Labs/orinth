# Screenshots to add

The documentation, landing page, and README were updated to cover every feature.
The following **new** captures are referenced but not yet in this folder. Drop each
PNG here (in `frontend/public/brand/`) with the exact filename below. Use the same
unretouched, real-app style as the existing captures (roughly 1915×927).

| Filename | Capture | Referenced by |
| --- | --- | --- |
| `9_data_recipes.png` | Data Recipes workspace (`/datasets/recipes`) — the sources → generate → review → commit flow | docs/data-recipes, README |
| `10_dataset_hub_import.png` | Dataset catalog with the "Browse HuggingFace" import panel open (`/datasets`) | docs/dataset-studio |
| `11_llm_training_detail.png` | A live LLM fine-tuning run detail (`/training/[jobId]`) with progress and logs | docs/llm |
| `12_llm_chat.png` | Chat with a served GGUF model (`/inference/chat`) — served-model rail, sampler, transcript | docs/llm, landing, README |
| `13_model_catalog.png` | Models page grouped by source: Reference / Trained / Uploaded (`/models`) | docs/models |
| `14_model_detail_export.png` | Model detail page with GGUF export / serve actions (`/models/[modelId]`) | docs/models |
| `15_testing_comparison.png` | Testing page comparison panel across multiple models (`/testing`) | docs/testing, README |
| `16_guided_tour.png` | Any surface with a react-joyride tour step visible (the Tour button + tooltip) | docs/guided-tours |

Existing captures already in this folder (reused, no action needed):
`1_project_list.png`, `2_dataset_list.png`, `3_dataset_studio_image_segmentation_task.png`,
`4_dataset_studio_image_clasification_task.png`, `5_available_trained_model_list.png`,
`5_dataset_studio_nlp_clasification_task.png`, `6_training_details.png`,
`7_inference_image_segmentation_task.png`, `8_inference_image_clasification_task.png`.

Until each new file is added, its `<Image>` / `![]()` slot renders as a broken image.
