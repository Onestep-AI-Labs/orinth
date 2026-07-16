"""NLP training runner implementations, split by model family.

`app.training.runners.nlp_train` remains the stable subprocess entrypoint
(invoked as ``python -m app.training.runners.nlp_train``); this package
holds the per-family implementations it dispatches to.
"""
