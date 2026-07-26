import type { RecipeOutputFormat, RecipePromptFlavor } from "@/types/api";

/**
 * Declarative recipe presets (phase 11), mirroring Unsloth Studio's
 * `learning-recipes/*.json` idea at our scale. Each preset prefills the output
 * format, chunking defaults, records-per-chunk, and prompt flavor; everything
 * stays editable after a recipe is created from it. "Start blank" is rendered
 * separately by the gallery.
 */
export type RecipePreset = {
  id: string;
  name: string;
  description: string;
  concepts: string[];
  output_format: RecipeOutputFormat;
  prompt_flavor: RecipePromptFlavor;
  chunk_size: number;
  chunk_overlap: number;
  records_per_chunk: number;
};

export const RECIPE_PRESETS: RecipePreset[] = [
  {
    id: "document-qa",
    name: "Document QA",
    description: "Turn documents into question-and-answer pairs a model can learn to answer.",
    concepts: ["Documents", "LLM-assisted", "QA pairs"],
    output_format: "instruction_jsonl",
    prompt_flavor: "qa",
    chunk_size: 3000,
    chunk_overlap: 200,
    records_per_chunk: 3
  },
  {
    id: "instruction-from-answer",
    name: "Instruction from Answer",
    description: "Frame each section as an instruction with the passage as the target response.",
    concepts: ["Documents", "Instructions", "SFT"],
    output_format: "instruction_jsonl",
    prompt_flavor: "instruction",
    chunk_size: 3000,
    chunk_overlap: 200,
    records_per_chunk: 2
  },
  {
    id: "conversation",
    name: "Conversation",
    description: "Generate short single-turn chats grounded in your source material.",
    concepts: ["Documents", "Chat", "Single-turn"],
    output_format: "chat_jsonl",
    prompt_flavor: "conversation",
    chunk_size: 3000,
    chunk_overlap: 200,
    records_per_chunk: 3
  }
];
