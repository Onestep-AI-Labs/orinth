import type { SplitKey } from "@/types/api";

export const SPLITS: SplitKey[] = ["unassigned", "train", "valid", "test"];
export const TRAINING_SPLITS: SplitKey[] = ["train", "valid", "test"];
export const TERMINAL_STATUSES = new Set(["completed", "failed", "canceled"]);
