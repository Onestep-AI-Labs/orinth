export type ModelInfo = {
  id: string;
  name: string;
  family: "yolo" | "unet_inception";
  description: string;
  available: boolean;
  paths: Record<string, string>;
  promoted: boolean;
};

export type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  polygon: number[][];
  mask_area: number;
};

export type InferenceResult = {
  id: string;
  model_id: string;
  image_level_label: string;
  detections: Detection[];
  overlay_url: string | null;
  original_url: string | null;
  parameters: {
    confidence_threshold: number;
    iou_threshold: number;
  };
  created_at: string;
};

export type EvaluationDataset = {
  key: string;
  name: string;
  format: "yolo" | "coco";
  split: string;
  available: boolean;
  path: string;
};

export type EvaluationJob = {
  id: string;
  model_id: string;
  dataset_key: string;
  status: string;
  limit: number | null;
  metrics: Record<string, any>;
  artifacts: Record<string, any>;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type TrainingJob = {
  id: string;
  model_family: string;
  status: string;
  parameters: Record<string, any>;
  artifacts: Record<string, any>;
  promoted_model_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
};
