# Guided Tours

Every main surface in the workspace has a short, interactive walkthrough that points out its key controls in place. Tours are built on [react-joyride](https://react-joyride.com/).

![A guided tour step highlighting a control](/brand/16_guided_tour.png)

## First-run walkthrough

The first time you open the workspace, a welcome tour plays automatically on the **Projects** page. It introduces the five-stage workflow — Organize → Prepare → Train → Test → Inspect — and points out where projects, documentation, and settings live. It runs once; after that, tours are launch-on-demand.

## Per-page tours

Each of these surfaces has its own tour, launched from the **Tour** button that floats in the bottom-right corner of the page:

- **Projects** — creating a project and opening a workspace.
- **Datasets** — the catalog, adding a dataset, Hub import, and opening Dataset Studio.
- **Dataset Studio** — once a dataset is open: items, annotation, EDA, and config/versioning.
- **Training** — picking a task and base model, choosing a dataset, tuning parameters, and starting a run.
- **Testing** — scoring models and reading the comparison.
- **Inference** — running a single item and reading the result and history.
- **Chat** — serving a model and sending a grounded message.
- **Models** — the catalog and uploading custom weights.
- **Settings** — configuring the Hugging Face token and OpenRouter key.

## Notes

- The **Tour** button appears only on surfaces that have a tour, and hides while a tour is running.
- Progress through a tour with **Next** / **Back**, or end it with **Skip**.
- Tours never change your data — they only highlight controls and describe what they do.
