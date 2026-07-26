# Data Recipes

Data Recipes turn your own source documents into a structured dataset — most often an instruction dataset for LLM fine-tuning — without hand-writing every record. A recipe is a small, reviewable pipeline: point it at sources, generate candidate records, review them, and commit the result as a real dataset in the project.

![Data Recipes builder](/brand/9_data_recipes.png)

## The recipe workflow

1. **Sources** — Add the documents the records should be grounded in.
2. **Generate** — Produce candidate records from those sources. Generation is deterministic and rule-based by default; when an OpenRouter key is configured under **Settings**, generation can be LLM-assisted for richer records.
3. **Review** — Inspect the generated records, edit or drop weak ones, and confirm the set reads the way you want before it becomes data.
4. **Commit** — Write the reviewed records into the project as a first-class dataset, ready to split, version, and train on.

## Deterministic by default

Recipes never require a network call to produce output. Without an OpenRouter key, they fall back to deterministic rule-based records, so the feature works fully offline and the same sources yield the same baseline records. The key only changes generation quality — never whether the feature runs.

## Where recipes fit

- Use **Dataset Studio** to import an existing dataset or upload and label your own items.
- Use **Data Recipes** when you have raw documents and want a supervised dataset generated from them.
- Either path produces a normal project dataset — from there, **Models & Training** and **LLM Fine-tuning & Chat** work the same way.
