# Spec: Phase 17 Model Architecture Studio

## Status

**Implemented — all five stages, plus a second emitter.** Compose a graph (69 node types, 38 starter
templates), validate it, save it, read and download the generated Python as **either TensorFlow or
PyTorch**, and train it: image and text classifiers into the model registry, and from-scratch
transformers on next-token prediction with sample generation.

### Revision 15: the stubs were being clipped, and zoom was still ambiguous

**Every free-handle stub was clipped out of existence.** Revision 14 fixed the
cascade that dropped them into the node's flow; that only revealed the real
reason none of them were visible. `.arch-node` sets `overflow: hidden` — which
is what keeps the header tint and the category stripe inside the rounded
corners — so a child positioned 46px past its edge is cut off completely. The
one node type that *did* show a stub was a block, because `.arch-node-block`
overrides with `overflow: visible`.

`CanvasNode` now returns an `.arch-node-shell`: an unclipped, shrink-wrapping
box holding `.arch-node` and the stubs as *siblings*. The node keeps its clip;
the stubs are outside it.

They also read as controls now rather than as absences. The lead *line* stays
dashed and faint — it stands for "no connection here" — but the *button* is
solid `--line-input` on `--surface` with muted ink. At a dashed hairline and
0.6 opacity it disappeared into the dot grid, which is the one thing an
affordance offering the next action cannot do.

**Zoom in and zoom out looked identical, and size was never the reason.**
base.css styles a control glyph for its own *filled* icon set:

```css
.react-flow__controls-button svg {
  width: 100%; max-width: 12px; max-height: 12px; fill: currentColor;
}
```

All four declarations are wrong for a lucide outline glyph, and `fill` is the
one that actually broke it. Lucide carries `fill="none"` as a **presentation
attribute**, which any CSS rule outranks — so the magnifier's lens filled solid
and swallowed the `+`/`−` sitting inside it. Both buttons rendered as the same
dark disc, which is why two rounds of resizing changed nothing.

`fill: none` and `stroke: currentcolor` fix it; the size caps are overridden
alongside, since with the fill off the sign was still only a couple of pixels.
The corner-badge composition tried earlier in this pass is gone — the
conventional glyph *is* a lens with the sign inside it, and once it renders as
an outline it reads correctly at 22px / `strokeWidth` 2.4.

**Clicking the canvas closes the layer panel.** `onPaneClick` and `onNodeClick`
both clear the intent: clicking the graph is how you say "not that" to anything
the studio is showing, and reaching for the panel's close button to dismiss
something you had already dismissed is a second gesture for one decision.

**A node added from the panel no longer lands on top of another node.**
`freeSpot` — moved into `auto-layout.ts` with `NODE_WIDTH`/`NODE_HEIGHT`, where
it is covered by `auto-layout.test.ts` — walks down a row at a time from the
requested point until nothing overlaps, keeping the column so the node stays
next to whatever it will connect to. `addNode` starts it 48px below the centre
of the view, clearing the floating toolbar; `extendFrom` starts it one column
along from its anchor. The walk is bounded at 40 steps so a pathological graph
cannot spin.

### Revision 14: the right-click menu never ran anything, and two CSS bugs

**No context-menu command had ever worked.** `CanvasContextMenu` dismisses on
`pointerdown` in the *capture* phase — one step ahead of the `click` its rows
are wired to — and the listener did not check where the press landed. Pressing
a row tore the menu down before the browser dispatched `click`, so `onSelect`
never fired and every entry looked inert: Duplicate, Settings…, Add node, all
of them. `dismiss` now ignores presses inside the menu.

`context-menu.test.tsx` covers it. The load-bearing assertion is that clicking
a row closes the menu *exactly once* — a second close means the outside-press
listener also fired, which in a real browser is the moment the row stops
existing. jsdom still dispatches `click` on a detached node, so asserting
`onSelect` alone would not have caught this.

**Every stub `+` rendered inside its node.** `.arch-node-stub` sets
`position: absolute`; `.arch-tip` sets `position: relative`; the stubs carry
both classes, both selectors are one class wide, and the tooltip block was
written later in the file — so source order handed it the cascade and dropped
every stub back into the node's normal flow. The tooltip block now sits above
anything that positions a tipped element, with a comment saying why it has to.

**Tooltip placement, three cases.** The toolbar is pinned to the canvas's left
edge and the canvas clips overflow, so a centred tip under the first button was
cut in half — toolbar tips align to the button's left edge. The wire pill sits
*on* a connection, so a tip below it landed on whatever the wire runs into —
`.arch-tip-up` puts those above. The zoom stack keeps `.arch-tip-right`.

**The zoom glyphs were indistinguishable.** base.css caps a control glyph at
12px and stretches it to `width: 100%`, which shrank the `+`/`−` inside the
magnifiers to a couple of pixels; both buttons read as plain circles. Fixed
19px, cap removed, `strokeWidth` 2.1.

**Fit to view was two buttons.** The top-left toolbar's `Frame` copy is gone;
fit stays in the bottom-left stack with the other view controls. `F` and the
menu entry are unchanged.

**A node added from the palette landed near the flow origin**, so panning two
screens right to work on the tail of a graph and adding a layer put it off-view
behind you — the add looked like it had failed. `viewportCenter()` reads the
canvas rect (inset past the layer panel while that is open) through
`screenToFlowPosition`, and `addNode` centres the node's box on it with a small
stagger for successive adds.

**Picking a layer now closes the panel and opens the new node's settings.**
Both halves of adding a node in one gesture: a `Conv2D` with catalog defaults is
rarely the `Conv2D` you wanted, and the panel left open sat over the node it had
just made. Dragging from the panel is exempt — that gesture places something
exactly and usually belongs to a run of drops, which a dialog would interrupt.

### Revision 13: the canvas says what its buttons do, and offers the next one

Follow-up pass on revision 12.

**Every icon-only control now has hover text.** Canvas controls — the wire pill,
the toolbar, the zoom stack, the new handle stubs — carry `data-tip` and the
`.arch-tip` CSS tooltip rather than `title`: a native tooltip waits about a
second and then paints an OS chrome box over the graph, which is the wrong
latency and the wrong surface for a control hit dozens of times a session. Page
chrome outside the canvas keeps `title`, where the delay is a feature; the ones
that had neither (fullscreen, panel close, drawer collapse, dialog close) got
one.

Two containers had to give up `overflow: hidden` for the tooltip to escape —
`.arch-tool-group` and `.react-flow__controls`. Both had it only to clip their
children's corners, so the end buttons carry the rounding instead.

**Zoom is a magnifier, not a `+`.** React Flow's stock controls render a bare
`+`/`−` pair, which on a canvas whose *other* `+` adds a node is the wrong
glyph. `showZoom`/`showFitView` are off and three `ControlButton`s carry lucide
`ZoomIn`/`ZoomOut`/`Maximize`. These stroke `currentColor` rather than filling,
so `.react-flow__controls-button` needed `color` alongside its existing `fill`.

**Inserting into a wire re-lays-out the graph.** The gap between two columns is
not a node wide, so a node dropped at a wire's midpoint landed *on* both its
neighbours and every insert was followed by dragging the rest of the chain out
of the way. `insertOnEdge` now runs `autoLayout` over the graph it just built.

**A free handle grows a stub `+`.** `ArchNodeData` gains derived
`hasIncoming`/`hasOutgoing` — folded in beside `shape` and `issues`, never
stored, never sent — and `CanvasNode` draws a dashed lead line and a `+` on any
side nothing is wired to. It opens the palette with an `extend` intent, and the
node that comes back is *connected*: an unwired box placed nearby would move
the same dead end one column over. Drawn quietly (dashed, `--line-strong`,
0.6 opacity) because mid-build most handles are free.

**The column pitch went 240 → 300.** At 240 a 176px node left 64px of wire,
which read as nodes touching rather than as a graph with connections; 300
leaves 124px — enough for the hover pill to sit on without covering either
endpoint. `backend/app/ml/architecture/layout.py` carries the same constant and
moved with it, or Tidy would shift every node on a freshly opened template.

**The right-click menu is complete.** Added **Disconnect** (drop every wire on
the selection, keeping the nodes — picking them off one at a time is at least
two operations for a node mid-chain), **Deselect all**, **Tidy layout**, and
**Fit to view**, so every canvas action reachable by keyboard is also reachable
from the menu.

Also: the palette's Blocks/Layers segmented control gained a top margin — in the
overlay drawer the search row sits directly above it, and the two read as one
crowded block without a gap.

### Revision 12: the canvas navigates like a node editor, not like a form

Six changes, all one complaint: moving around this canvas cost more than the
graph did.

**Trackpad gestures are the n8n set.** `panOnScroll` with
`PanOnScrollMode.Free`, `zoomOnScroll={false}`, `zoomOnPinch`. Two fingers pan
in both axes; pinch zooms. Before, a two-finger scroll *zoomed*, so every
attempt to look at the node to the right changed the scale instead — and a
trackpad's sideways component made that zoom jitter mid-gesture. ⌘/Ctrl+scroll
still zooms for a mouse, and the `Controls` cluster still has explicit buttons.

**Drag on empty canvas always selects.** `selectionOnDrag` is unconditional and
`panOnDrag` is the middle mouse button, plus `true` while `Space` is held.

**The select/move tool pair is gone**, along with its `V`/`H` shortcuts and the
`.arch-canvas-select` / `.arch-canvas-move` cursor rules. Revision 11 fixed
those rules by making the wrapper render the tool class; the better fix was
that there is no tool to indicate. A modal pointer in a two-mode editor is a
mode you have to look at the toolbar to check, and both modes now have a
gesture that does not cost one — trackpad or middle-drag for pan, plain drag
for select. `.arch-canvas-pan` remains for the grab cursor while `Space` is
down, which is a *held* state rather than a mode.

**The layer panel and the settings rail no longer hold columns.**
`.arch-studio-grid` went from three panes to one; the canvas gets the row.

- The palette is now the body of `.arch-palette-drawer`, overlaid on the
  canvas's left edge and closed by default. It opens from the canvas `+`
  (`N`), a right-click, or a wire's insert button, and `PaletteIntent` carries
  *why* — `canvas`, `at` a point, or `edge` — so one panel serves three entry
  points and the pick lands where it was asked for. The targeted intents close
  the panel on pick; browsing from `+` leaves it open, because building a stack
  is a run of picks rather than one.
- Settings are `SettingsModal`, opened by double-clicking the node or frame
  (`zoomOnDoubleClick` is off to free that gesture), by `↵`, by the header's
  slider button, or from the node's context menu. `NodeInspector` and
  `GroupInspector` are unchanged: `.arch-modal .arch-inspector` strips the
  rail's border and radius in CSS, so neither component knows where it renders.
  The dialog closes itself when its subject is deleted or deselected.

**Right-click adds a node where you clicked.** "Add node" is the first row of
the pane menu, above the clipboard block, because that is why the menu gets
opened. The node menu gains "Settings…", the edge menu "Insert node here".

**A hovered wire carries its own two buttons** — `+` inserts a node into the
middle of the connection (`a → b` becomes `a → new → b` in one action, rather
than delete-wire, add-node, draw-two-wires), `×` removes it. All edges are one
`ActionEdge` type now; the curve/step choice moved from React Flow's built-in
types into its `data.branching`.

The pill renders inside `EdgeLabelRenderer`, a portal *outside* the edge's own
DOM subtree, so `.react-flow__edge:hover .pill` cannot reach it — hover has to
be state. `onEdgeMouseEnter`/`Leave` set it, the pill re-asserts it on its own
`pointerenter`, and clearing is deferred by 90ms: the pill sits on the line, so
moving onto a button fires `mouseleave` on the path one frame before
`pointerenter` on the button, and without the grace period the affordance
blinks out from under the pointer.

### Revision 11: the canvas tool never changed the cursor

`platform.css` carried the cursor affordance for both canvas tools —
`grab`/`grabbing` on the pane under `.arch-canvas-move`, `crosshair` under
`.arch-canvas-select` — but the wrapper only ever rendered the bare
`arch-canvas` class, so neither rule could match. The tool buttons changed
`selectionOnDrag` and `panOnDrag` correctly, so the *behaviour* switched while
the pointer kept saying "select" in both modes, which is the one signal that
tells you which tool is live before you drag. The wrapper now renders
`arch-canvas arch-canvas-${tool}`.

Also removed two unused helpers in `blocks.py`, `is_vision_block` and
`block_source`; `VISION_BLOCKS_BY_TYPE` and `BlockFamily.source` are still read
by the catalog.

**`test_augmentation_is_training_only_in_both_frameworks` was flaky**, on both
the Keras and the Torch side. It asserted that one training-mode draw differs
from the settled inference output, but `random_flip` mirrors each image
independently with p=0.5, so a 4-image batch comes back untouched once every 16
draws and the layer looks broken while behaving exactly as specified. The
assertion now takes the property over 8 draws — still "training perturbs,
inference is the identity", without the 6% false failure.

### Revision 10: a selected connector could not be deleted

`deleteKeyCode` is `null` so the studio's own key handler can take group
frames — which React Flow does not own — with the selection. But
`deleteSelection` only ever gathered nodes and groups, so nothing removed a
selected edge: `Delete` was a no-op on a wire, and the context menu's generic
**Delete** was disabled because `selectionCount` did not count edges either.
The only way to remove a connector was the edge context menu's **Delete
connection**. `deleteSelection` now collects selected edge ids alongside the
nodes and groups, and a right-click on a wire selects it first, so the menu
acts on what is highlighted.

**A clicked wire also came out grey, not blue** — and it was a cascade tie, not
a colour choice. React Flow gives a focusable edge `tabIndex=0`, so clicking one
focuses it, and `base.css` carries

```css
.react-flow__edge.selectable:focus .react-flow__edge-path { stroke: var(--xy-edge-stroke-selected, #555); }
```

at specificity `(0,4,0)` — exactly equal to `platform.css`'s
`.arch-canvas .react-flow__edge.selected .react-flow__edge-path`. `base.css` is
imported from the studio *client component*, so it loads after the global sheet
and wins the tie. Only `stroke` was overridden, which is why the selected wire
still thickened to 3.5px: it rendered as a *thick grey* line. Hover stayed
correct throughout because `base.css` styles no hover state, which is what made
the bug look like "hover works, selection doesn't".

The fix feeds React Flow's own theming hook — `--xy-edge-stroke-selected` set on
`.arch-canvas` — so its rule paints the accent rather than being fought, and the
studio's own rules carry `.selectable` to sit one class above the vendor rule.
The stroke also moved from `--color-accent-strong` (`#1e40af`, meant for accent
*text* on a tint) to `--color-accent` (`#2563eb`), the solid canvas blue. The
surface-coloured casing stays.

### Revision 9: graph models could not be reloaded, and augmentation reached the canvas

A graph-built model trained to completion and then failed every downstream use.
Two independent causes, both in what the emitter wrote rather than in training:

- **`Activation` was given a lambda.** `inverted_residual` emitted
  `Activation(lambda t: tf.clip_by_value(t, 0.0, 6.0))` for relu6 and an
  inline hard-swish. Keras saves a lambda as marshalled bytecode, so
  `load_model` raised "Could not interpret activation function identifier"
  against a base64 blob, and testing and inference were unreachable for every
  MobileNet graph. Keras ships `relu6` and `hard_silu` as named activations;
  both now emit by name. The classic FFN's `Lambda(gelu)` went the same way.
- **Custom layers were never registered in the serving process.** `SqueezeExcite`,
  `PatchEmbedding`, `RMSNorm` and the rest are resolved by name at load time
  from a registry populated by *running* their definitions. Training does that
  by importing `generated_model.py`; nothing did it in the API process, so
  MobileNetV3 and ViT graphs failed with "Could not locate class". The
  generated file cannot be the fix — [Custom code and trust](#custom-code-and-trust)
  forbids the FastAPI process executing it. `ml/architecture/runtime.py`
  instead executes the same helper *source strings* the emitter writes, which
  are repository code with no user content, and the predictor calls it before
  `load_model`. Executing the strings rather than restating the classes keeps
  one definition: a second copy would drift from the trainer's the first time a
  block was fixed, and a model would then be rebuilt with the wrong layer.

Both are invisible to any test that only builds a model, which is why training
never caught them. `test_architecture_families` now saves and reloads every
vision block and compares outputs.

**Augmentation nodes** (`random_flip`, `random_rotation`, `random_zoom`,
`random_contrast`) close the gap this exposed: a from-scratch graph had no
regularization at all, while the transfer-learning runner had four
augmentation toggles. They are layers rather than training-form knobs so the
graph stays the single source of truth — a run is reproducible from the canvas,
and the augmentation travels through export and import. All four are active
only under `training=True`, so a saved model still serves deterministically.

Note that from-scratch and transfer learning are not the same experiment and
augmentation does not make them one: on 1770 images the hand-built MobileNetV2
reached 0.53 validation accuracy at 435 ms/step against the frozen ImageNet
backbone's 0.81 at 63 ms/step, because the first trains 2.26M parameters and
the second trains 7,686. The `pretrained_backbone` node already emits a model
identical to `keras_classification_train`'s, and remains the answer when the
goal is accuracy rather than studying the architecture.

### Revision 8: group frames were unmovable, and the drag never followed the pointer

Two of Revision 5's frame fixes were built on wrong readings of React Flow, and
together they left a frame that could not be moved at all once it had been
resized. Verified against `@xyflow/react` 12.11.2 / `@xyflow/system` 0.0.79
rather than from the type signatures.

**A resize drag never ended, so it swallowed everything after it.** `XYResizer`
emits two shapes: during the drag `{ resizing: true, setAttributes, dimensions }`,
and on release `{ resizing: false, dimensions }` — **no `setAttributes`**.
`resizePhase` tested `setAttributes` on both edges, so it never returned `"end"`,
`resizingRef` latched open after the first resize, and every later change to that
frame — position, size, selection — was pushed into the pending buffer and never
applied. The phase now keys on `resizing` being a boolean, which is the only
field both edges share, and `"end"` wins over `"start"` in a batch carrying both
so a drag cannot latch. `isResize` keys on the same field, so the end change
commits its size instead of being mistaken for a measurement.

**"React Flow moves the frame's DOM from its own store" is false for a
controlled flow.** `triggerNodeChanges` only applies changes to the store when
`hasDefaultNodes` — that is, when `nodes` was *not* passed as a prop. The studio
passes it, so nothing moves until state feeds back, and Revision 5's buffering
meant the rectangle stayed put for the whole drag and jumped on release. The
buffer is gone; group changes apply immediately, exactly as the model nodes'
already do. `resizingRef` survives for its one honest job: one undo step per drag.

**The real stutter was `measured`, not the re-render.** `adoptUserNodes` copies
`measured` from the node object it is handed, and `parseHandles` drops
`handleBounds` for any node without one. The group nodes are rebuilt from
`groups` on every change and carried neither, so each pointer event blanked both,
forcing a ResizeObserver re-observation and a store-wide re-render — and left
`XYResizer` reading `node.measured.width ?? 0` at the start of the next drag.
Group nodes now carry `measured` alongside `width`/`height`, which is what makes
model nodes drag smoothly (`applyNodeChanges` writes it back onto them). The
`onNodesChange` id set moved to a ref for the same reason: `groups` changes on
every pointer event, and re-registering the handler mid-drag is not free.

`setAttributes` is in fact always `true` here, because `NodeResizer` never passes
a `resizeDirection` and `XYResizer` already reports the untouched axis at its
previous value. The axis check stays in `resizedTo` as a guard for a
direction-locked resizer, not as the fix Revision 5 claimed it was.

**Title sizes run the full ramp** — 8 through 144px in word-processor steps,
fine at the small end and coarse at the top. A frame title is set against the
canvas zoom rather than against body copy, so a banner over a whole stage is a
real case. The inspector preview caps at 40px in CSS: the rail is a few hundred
pixels wide, and a 144px preview would push the colour and size controls off it.

### Revision 7: predictors read the model, not the metadata

Testing and inference both failed on every studio-trained model:

```
Input 0 with name 'x_input_1' of layer 'small_cnn' is incompatible with the layer:
expected shape=(None, 64, 64, 3), found shape=(1, 512, 512, 3)
```

`KerasClassificationPredictor` resized to `spec.artifacts["image_size"]`, which
`_register_training_model` copied from the training form. For a transfer-learning
run those two agree by construction — the backbone is *built* at the form's
`image_size`. A graph run takes its resolution from the Input node and ignores
the form entirely, so a 64×64 graph was registered as 512 and every prediction
died on the shape.

The fix inverts the authority. A saved model states what it accepts and
physically cannot accept anything else, so the predictor derives its geometry
from `model.input_shape` and keeps the recorded value only as a fallback for a
fully convolutional model that declares none. This repairs **already-registered
models without re-training them**, which matters because the broken ones are
already in the registry. `KerasTextClassificationPredictor` gets the same
treatment for sequence length, since a text graph has the identical mismatch.

Two supporting changes so the metadata stops lying as well: `architecture_train`
writes the resolution it actually trained at into `metadata.json`, and
`_register_training_model` prefers the runner's value over the form's. Neither
is load-bearing — the predictor no longer trusts either — but a registry that
records a resolution the model never had is a trap for the next reader.

### Revision 6: BatchNorm statistics, the emitter disagreement, and text routing

**The flat validation curve was BatchNorm, not the data.** A run on the 210-image
sample dataset reached 0.79 training accuracy while validation sat at exactly
0.1667 — 10 of 60, one class — with val loss pinned at 1.7919, which is `ln(6)`.
The model was emitting a uniform distribution on every validation sample.

BatchNorm normalises with *batch* statistics while training and with running
averages at inference, and those averages move by `1 - momentum` per step. Keras
defaults to `momentum=0.99`, which assumes thousands of steps per epoch; 210
images at batch 32 is seven. After eleven epochs the averages were less than
halfway from their initial mean 0 / variance 1, so at inference the network was
normalised by numbers describing nothing and its output collapsed.

Proven by holding everything else fixed — same seed, same data, same graph, only
momentum changed:

| BN momentum | train acc | train loss | val acc | val loss | classes predicted on valid |
| --- | --- | --- | --- | --- | --- |
| 0.99 | 0.4714 | 1.3798 | 0.3167 | 1.7724 | 3 of 6 |
| 0.9 | 0.4714 | 1.3798 | **0.5333** | **1.3944** | 6 of 6 |

Training is bit-identical; only inference differs. Three changes follow:

- The catalog default drops to **0.9**, which is what PyTorch's `momentum=0.1`
  means and is right for datasets at this scale.
- `graph._check_batch_norm_momentum` warns above 0.95. A graph saved from the
  studio bakes in every param, so lowering the default cannot reach existing
  architectures — this can.
- The `momentum` help text explains the train/inference split, because the
  symptom looks like a data bug and never like a normalisation setting.

Measured on the sample dataset afterwards, validation tracks training instead of
flatlining: val accuracy 0.167 → 0.42, val loss 1.795 → 1.46.

**The PyTorch emitter was describing a different model.** `_norm("BatchNorm2d",
"eps=1e-3")` never read `ctx.params`: momentum and epsilon set on the canvas
applied in the Keras module and were silently dropped from the PyTorch one. The
conventions are also inverted — Keras `momentum` is the fraction of the *old*
statistic kept, torch's is the fraction of the *new* one taken, so Keras 0.9 and
torch 0.1 are the same layer. `_batch_norm` now reads both params and converts,
and a parametrised test pins the mapping. This is the same rule the spec already
states for the LLM blocks: two emitters that disagree are describing two
different models.

**Text architectures could not be trained from the form.** `training-page.tsx`
matched a single hardcoded option id, `"architecture_graph"`, so selecting
"Visual architecture (text classifier)" rendered no Architecture field at all and
sent no `architecture_id`. The studio's Train button compounded it by hardcoding
`setTaskType("classification")`, routing every graph — text or not — to the image
runner. The option ids are now a task→id map, the Train link carries the
architecture's `task_type`, and the Architecture dropdown filters to graphs
matching the selected task, since an image graph cannot start under the text
runner.

**Frame resizing is uncontrolled during the drag.** ~~React Flow already moves
the frame's DOM from its own store as the pointer moves~~ — false for a
controlled `nodes` prop, and this buffering is what made the frame unmovable.
Reverted in Revision 8.

### Revision 5: the shuffle bug, preset visibility, and frame editing

**Every image classification run in the platform was training on class-ordered
batches.** Not a studio bug — `keras_common.image_datasets` is shared with
`keras_classification_train`, so this affected transfer-learning runs too.

`load_split` walks the split directory in sorted filename order and every dataset
the platform writes names files by class, so the item list arrives in perfect
class order — for the trash dataset, 282 cardboard, then 351 glass, then 287
metal, then 416 paper, then 338 plastic, then 96 trash. The tf.data `shuffle`
only mixes within its buffer, and that buffer holds *decoded images*, so it
cannot be dataset-sized. With 512 images over a class-ordered 1770, batch
composition tracked position in the stream rather than the dataset. The model
chased whichever class was streaming past: training loss climbed *through* each
epoch (1.17 → 1.85 in epoch 1 on the reproduction) and validation accuracy sat
at 0.087 for six classes.

Fixed by permuting the file list in Python before tf.data sees it — exact rather
than approximate, and free, because it is a list of paths. The buffered shuffle
stays on top for per-epoch variation. Determinism holds: the runners call
`keras.utils.set_random_seed`, which seeds Python's `random`. Measured on the
same graph and dataset afterwards, training loss falls monotonically
(1.62 → 1.06 over 25 epochs) and validation accuracy climbs 0.20 → 0.53.
`test_keras_common.py` asserts batch composition directly and fails without the
permutation.

**NLP and LLM presets were unreachable.** Revision 4's task picker filtered the
preset grid and drew its options from the project's `task_types`. But
`language_modeling` exists for architectures composed in the studio and no
project declares it, so all seventeen LLM presets — GPT-2, Llama, Qwen, Mixtral,
DeepSeek, Kimi — were invisible in every project, and the four text-classifier
presets were invisible in any vision project. The picker now offers every task
the studio can build (project tasks first), and the selected task **sorts** the
preset grid rather than filtering it; narrowing is an explicit opt-in. A preset
is the fastest way to learn what the canvas can express, so hiding one is
expensive in a way that showing an extra row is not.

**Group frames were resizing on the wrong axes.** ~~`setAttributes` is the axis
name for an edge drag, and falsy for React Flow's post-mount measurement.~~
Neither holds in 12.11: `NodeResizer` passes no `resizeDirection`, so
`setAttributes` is always `true`, and the resizer's end change omits it
entirely. Corrected in Revision 8; the axis rule survives only as a guard.

**Resizing lagged the pointer because frames were part of the model.** Group
geometry rode in `toGraph`'s `training_defaults`, so `groups` was a dependency of
the graph memo: nudging a decorative rectangle rebuilt every node and edge in the
IR and re-armed shape validation, once per pointer event. Frames are now kept out
of `graph` entirely and merged back in at save time, which is the only moment
their geometry has to travel anywhere. The React Flow `nodes` array is memoized
for the same reason.

### UI revision 5

- **The group title is its own text box** — click to select, drag to place
  anywhere inside the frame (clamped to it), double-click to retype. Size and
  colour are set in the inspector; `titleTint` of `-1` means neutral ink and
  anything else indexes `GROUP_TINTS`, so a title stays a token colour. All four
  fields are optional on `StoredGroup` so an architecture saved earlier still
  opens.
- **Position X/Y left the inspector.** A frame is placed by dragging it; typing
  coordinates for an annotation is a control nobody reaches for, and it took the
  room the title controls needed.
- **Font size is an editable dropdown** (`NumberCombo`, a new shared primitive):
  a menu of common sizes for the usual case, a number field for the value the
  layout actually wants. Native `<input list>` was rejected — no control over the
  popup, different in every browser, and no affordance that a list exists.
- **Drag affordances are visible at rest.** Pane dividers and frame edges show a
  hairline grip in `--line-strong` that promotes to the accent and lengthens on
  hover, so a resizable edge is discoverable without first hovering it.
- **A selected edge is the solid accent blue with a surface-coloured casing**
  behind the stroke, so it reads against the dot grid, a group frame's pastel
  wash, or a node it passes behind — which one stroke colour cannot do alone.
- **Explanatory prose is two tiers, not five sizes.** Card, preset, and inspector
  descriptions share one size, colour, leading, and gap; the clamped one-liner in
  a palette row is the deliberate denser tier.

### Revision 4: the head trap, text training, and editor conventions

**The bug this revision exists for.** A VGG-16 built from a blank canvas failed to train, while the
VGG-16 preset trained but never left chance accuracy. Two silent head defects, neither visible on the
canvas:

- `dense.units_from_dataset` **defaults off**, so a hand-built head emits a fixed width. The image
  runner refuses a run whose head and label set disagree, which is a launch-time failure with no hint
  on the graph that produced it.
- `dense.activation` **defaults to `linear`**, so a hand-built head emits logits. The runner compiled
  `CategoricalCrossentropy` over probabilities, which does not raise on logits — Keras clips and
  renormalises them. The run starts, the loss parks at `ln(num_classes)`, and accuracy never moves.
  This is what "trains but a bit issue" was.

Fixed on both sides. `graph._check_classifier_head` warns on the layer feeding the Output node for
each case, so the mistake is on the canvas before training is attempted; an intermediate Dense is not
checked. Both image and text runners now read the built model's last-layer activation and compile
with `from_logits=True` when it is not already normalised, so an existing graph trains correctly
rather than merely being diagnosed.

**Image batching states its cardinality.** `keras_common.image_datasets` builds from a generator,
whose length Keras cannot know: the first epoch rendered as `1/Unknown` and every run ended on a
spurious "your input ran out of data" warning. The item count is known at construction, so
`assert_cardinality` now states it, and `fit` passes `shuffle=False` because the pipeline already
shuffles.

**Text classification is trainable from the canvas.** The studio shipped `bert_classifier` and
`transformer_text_classifier` presets that no training option could run — the catalog covered only
`classification` and `language_modeling`. New family `architecture_text` (`TEXT_OPTION_ID`) with
runner `architecture_text_train`: the sequence length comes from the Input node and the vocabulary
from the Embedding node, the same "the graph owns the shape" rule the image runner follows for
resolution. Capping the tokenizer at the embedding table size is a correctness requirement, not a
nicety — an out-of-range id crashes the first batch. Artifacts match `nlp/keras.py`'s text classifier
exactly, so promotion, testing, and serving go through the existing `KerasTextClassificationPredictor`.

### UI revision 4: editor conventions

- **Undo and redo** (`use-history.ts`), snapshot-based over `{nodes, edges, groups}`. Snapshots are
  taken *before* a change and at drag/resize start, which is what makes granularity match intent: a
  forty-event drag is one undo step. `⌘Z` / `⌘⇧Z` / `⌘Y`, plus toolbar buttons.
- **Cut, copy, paste, duplicate, select all** (`clipboard.ts`) over the whole selection, group frames
  included. Payloads ride `text/plain` behind a marker string rather than a custom MIME type, so a
  copy survives a round trip through another tab; an in-app clipboard backs the context menu, which
  has no `ClipboardEvent` to read from. Pasting re-ids nodes around what is already on the canvas and
  rewrites internal edges onto the new ids, so a pasted block arrives wired. An edge with one endpoint
  outside the selection is dropped rather than left dangling.
- **A right-click menu** on the pane, nodes, edges, and frames, carrying the same commands with their
  accelerators printed. Right-clicking outside the selection selects that thing first.
- **Group frames are first-class.** Selection is tracked (it previously was not, so `NodeResizer`
  never appeared and frames could not be resized at all), `Delete`/`Backspace` removes them with the
  rest of the selection, the title renames on double-click, and colour, exact size, and position moved
  into a `GroupInspector` in the right rail — a colour palette floating over the canvas covered the
  nodes the frame was describing.
- **Selection is visible.** Edges gain an 18px interaction width and turn solid accent blue at 3.5px
  when selected (see [Revision 10](#revision-10-a-selected-connector-could-not-be-deleted));
  connectors went from 9px to 14px, above the threshold where hitting one is aim rather than intent.
- **The canvas gets the space.** Narrower default rails, a shorter drawer, a taller studio, one scroll
  region covering the inspector header, palette entries with room between them, a palette filter that
  survives a 160px rail (container query), and toolbar icons enlarged by removing padding rather than
  adding width.
- **Creating an architecture picks its task.** A preset carries its own, so choosing one auto-detects;
  the blank canvas gets a `TaskSelect` that also filters the preset grid, with an escape hatch to show
  every task.

### Revision 3: named blocks, not one generic transformer

The complaint this answers: every LLM preset was the same six nodes with different numbers, so a Qwen
graph and a DeepSeek graph were indistinguishable on the canvas and generated near-identical code.
They are not the same architecture, and the studio now says so.

**Fifteen named block node types**, each a palette entry in its own right with defaults read from the
model's published `config.json` or paper, recorded in `NodeSpec.source` and shown in the inspector.
They live in `app/ml/architecture/blocks.py`, the single registry `catalog`, `shapes`, `emit_keras`
and `emit_torch` all consume.

| Family | What makes it that family |
| --- | --- |
| `llama_block` | Pre-norm RMSNorm, GQA, rotary applied inside attention, SwiGLU |
| `qwen3_block` | + QK-norm over each query/key head vector |
| `mistral_block` | Sliding-window attention (4096) |
| `mixtral_block` | Sparse MoE: softmax over all experts → top-2 → renormalize |
| `gemma3_block` | Four norms a layer (sandwich), GeGLU, 5:1 local-to-global, `query_pre_attn_scalar` |
| `deepseek_block` | Multi-head latent attention, shared + 256 routed experts, sigmoid routing with a selection bias, first 3 layers dense |
| `kimi_block` | The same at 64 heads and 384 experts, one dense layer |
| `gpt2_block` | Pre-LayerNorm, full MHA, projection biases, no rotary |
| `bert_block` | Bidirectional, post-norm |
| `resnet_block` | Basic or bottleneck stage with a projection shortcut |
| `inverted_residual_block` | MobileNetV2's MBConv, optional SE and hard-swish |
| `dense_block` | DenseNet-BC concatenating growth |
| `inception_block` | Four parallel receptive fields |
| `convnext_block` | 7×7 depthwise, LayerNorm, 4× pointwise, LayerScale |
| `vit_block` | Pre-LN encoder over a patch sequence |

Plus five new primitives: `depthwise_conv2d`, `squeeze_excite`, `patch_embedding`, `geglu`,
`mla_attention`.

**Every preset now reproduces its published parameter count exactly**, and a test asserts it — which
is a check on the *structure*, not just the arithmetic, because a GQA model counted as MHA or an MLA
model counted as GQA lands somewhere else entirely:

| Preset | Parameters | Preset | Parameters |
| --- | --- | --- | --- |
| GPT-2 124M | 124,439,808 | Qwen3 8B | 8,190,735,360 |
| Llama 3.2 1B | 1,235,814,400 | Qwen3 30B-A3B | 30,532,122,624 |
| Qwen3 0.6B | 596,049,920 | Mixtral 8×7B | 46,702,792,704 |
| Llama 3.2 3B | 3,212,749,824 | Llama 3 70B | 70,553,706,496 |
| Gemma 3 4B (text) | 3,880,263,168 | DeepSeek V3 | 671,026,419,200 |
| Mistral 7B | 7,248,023,552 | Kimi K2 | 1,026,408,232,448 |
| Llama 3 8B | 8,030,261,248 | | |

Getting there required fixing three fidelity bugs the old generic block carried, each of which had
been quietly inflating every preset:

- **Rotary was applied to the token embeddings**, via a `rotary_embedding` node between the embedding
  and the block. No model does that — rotary is a property of the query·key dot product, and rotating
  the residual stream instead is a different and worse model. The family blocks apply it inside
  attention, and preset graphs no longer carry a separate RoPE node.
- **Attention and feed-forward projections carried biases.** Every RoPE-era config ships
  `attention_bias: false`; GPT-2 and BERT are the exceptions, and now the only ones with `use_bias`
  on. This alone was 4.9M parameters on Mistral 7B.
- **The LM head carried a bias.** The softmax that follows is shift-invariant, so no published head
  has one; at a 262k vocabulary it is not a rounding error.

**Both emitters generate only what the architecture reaches.** `blocks.llm_block_structures` reads a
block's parameters and returns the structures it uses; the block's own source is assembled from that
set. A Gemma module contains no mixture-of-experts branch and no `LatentAttention`; a DeepSeek module
contains no `FamilyAttention`; a GPT-2 module has neither RMSNorm nor a gated feed-forward. A dead
branch would not merely be untidy — the helper classes are emitted on the same basis, so it would
call a class the file never defines.

**Verified by building, not by reading.** `test_architecture_families.py` builds every block under
real TensorFlow *and* real PyTorch and asserts the analytic estimate equals both `count_params()` and
torch's parameter total. Two emitters that disagree are describing two different models. The
comparison excludes `num_batches_tracked`, a torch-only counter with no Keras counterpart, and
includes BatchNorm's running statistics, which Keras counts as non-trainable weights.

**Thirteen new templates**, at their published sizes (each within 3% of the paper's ImageNet figure,
the difference being the class count): ResNet-18, ResNet-50, VGG-16, GoogLeNet, MobileNetV2,
MobileNetV3, DenseNet-121, ConvNeXt-Tiny, ViT-B/16, U-Net, a squeeze-excite CNN, a BERT-base
classifier, and a hand-wired mixture-of-experts layer.

### UI revision 3

- **Layout draws branches as branches.** An edge spanning more than one column — a residual shortcut,
  a U-Net skip — now pushes the layers it goes around onto their own row, so the shortcut can be drawn
  straight. The Residual block template used to lay out as a single line with the skip edge hidden
  underneath the convolutions it was supposed to be routing around, which is a picture of the wrong
  architecture. Nested skips stack rather than overlap. `layout.py` and `auto-layout.ts` implement
  the same three rules, and `_round_half_up` exists because Python's `round` breaks ties to even while
  JavaScript's rounds half up — a one-row disagreement that made Tidy move every node in a freshly
  opened template.
- **The palette separates blocks from layers**, with a three-way filter and counts. "Which operation
  comes next" and "which architecture am I building" are different questions, and sixty-five flat
  entries answer neither.
- **Node headers carry a pastel category wash**, and a block is drawn with a stacked edge and a `×N`
  badge for the layers it expands into. `categoryColorVar` was returning `var(--label-N)` — the raw
  oklch triple, not a colour — so every node stripe had been painting nothing at all. It now goes
  through `labelColor`, and tints through `labelFill` per `DESIGN.md` §8.
- **Tools moved onto the canvas** (React Flow `Panel`, top-left): select, move, frame-selection, tidy,
  fit — with `V`/`H`/`G`/`T`/`F` shortcuts that stand down while a field has focus. They act on what is
  under the pointer, so a round trip to the page header was the most repeated wasted motion in the
  editor. Group frames gained a remove control, and adding one with nodes selected frames them.
- **The minimap colours nodes by category**, so it reads as the canvas rather than as grey dots.

### Revision 2: real model architectures, not just parameter counts

Configs verified against published sources in August 2026 (HuggingFace
`transformers` model docs and the DeepSeek-V3 report), recorded per preset in
`LlmPreset.source`. The features below are implemented in **both** emitters,
so a preset's generated code has the shape of the real model rather than a
generic transformer carrying its parameter count.

- **QK-norm** — RMSNorm over the query and key head dimensions before the dot
  product. Qwen3's signature; Gemma 3+ uses it too.
- **Sliding-window attention with a hybrid layer pattern** — `sliding_window`
  plus `global_every`, so most layers see a local band and every Nth sees the
  whole sequence. Gemma 4 runs 512-token windows at 5 local to 1 global.
- **Tied embeddings** — the LM head projects with the transpose of the
  embedding table instead of learning a second matrix. Materially changes the
  count: at Gemma 4's 262k vocabulary and 2304 width it removes 604M
  parameters.
- **Logit softcapping** — `tanh(logit / cap) * cap`, Gemma's
  `final_logit_softcapping`.
- **Shared experts** — always-on unrouted experts alongside the routed ones.
  DeepSeek V3 pairs 1 shared with 256 routed.
- **Decoupled head dimension** — Gemma's 8 heads of 256 against a 2304-wide
  residual stream. Deriving `head_dim` as width/heads would give 288 and build
  a different model, so presets state it.

**New named presets, with the counts a test asserts:** Qwen3 0.6B (596M),
Qwen3 8B (8.19B), Qwen3 30B-A3B (30.55B — the published figure exactly),
Gemma 4 sparse and dense, and a DeepSeek-style shared-expert MoE.

**Two presets deliberately do not match their namesake's headline size, and
say so in their own descriptions.** Gemma 4's published 26B-A4B adds per-layer
embeddings (PLE) and an expert count its public config does not state; the
preset is built from the documented `Gemma4TextConfig` defaults and lands near
16B. DeepSeek V3's multi-head latent attention has no node on this canvas, so
its attention is modelled as grouped-query and the total lands above 671B.
Naming them after the checkpoints they approximate would have been the easy
lie; the descriptions name the gap instead.

### UI revision 2

- **Select / Move tools.** Drag-on-canvas either marquee-selects or pans,
  rather than requiring a modifier for whichever one is not the default.
- **Group frames** — titled, resizable, colour-pickable annotation rectangles
  drawn behind the nodes. They live in `training_defaults.groups`, never in
  `nodes`, so no backend validator, shape rule, or emitter has to learn to skip
  them. Tints come from `labelFill` at 9%, the sanctioned translucent helper.
- **Branch-aware edges.** An edge leaving a fan-out or entering a merge is
  drawn as a bezier; a plain link in a chain stays a step. A step edge at a
  branch reads as one path, which is the opposite of the truth exactly where a
  residual or a router needs to be legible.

### Revision: scale, sparsity, and a second framework

- **A PyTorch emitter** (`emit_torch.py`) renders the same resolved graph as an `nn.Module`. Keras
  stays the compile target that trains in-platform; torch is an export. Both read one IR and one
  shape pass, so they cannot describe different models — a `slow` test asserts they agree on a
  transformer's parameter count exactly. Two nodes are untranslatable and say so rather than
  emitting something plausible: a `tf.keras.applications` backbone, and a custom layer written as a
  Keras `Layer`. The torch module is NCHW while the canvas shows NHWC, which the generated header
  states.
- **Grouped-query attention and mixture-of-experts nodes**, plus `num_kv_heads` / `ffn: "moe"` on the
  composite block. Both are emitted as generated helper classes in each framework. MoE experts are
  gated (SwiGLU), matching Mixtral — an earlier draft emitted 2-matrix GELU experts while the
  estimate assumed 3, which the torch build caught as a 50M-parameter disagreement.
- **Nine LLM scale presets, 6M to 70B**, in `llm_presets.py`. Each reproduces a published
  open-weight configuration, and a test asserts the parameter count matches: Mistral 7B →
  7,243,140,352; Llama 3 70B → 70,560,552,192; Mixtral 8×7B → 46,711,541,248. They are *blueprints* —
  the spec is explicit that nothing above ~200M trains on a workstation, and the local-build test
  suite skips them by name.
- **Catalog maximums were raised** (`layers` 64→256, heads 64→256, embedding width 8192→32768).
  A silent clamp had been turning the 70B preset into a 57B one; a test now asserts no preset is
  clamped.
- **`AdvancedParameterSpec` gained a `code` type**, and the studio renders it as a real editor —
  syntax-highlighted, with line numbers and static lint for the mistakes that actually happen when
  writing a Keras layer in a field (no `call()`, missing `super().__init__`, tab indentation,
  unbalanced brackets).
- **Pane sizes are user-controlled and persisted.** Palette, inspector, and the code/issues drawer
  each have a draggable, keyboard-operable separator; sizes live in `localStorage` per pane.
- **The class-count preview is stored on the architecture** (`training_defaults.num_classes_preview`)
  rather than reset to 2 every visit — the concrete complaint that a small CNN "always gives only 2
  classes" on the canvas.
- **Canvas interaction:** box selection and multi-select (delete acts on the whole selection),
  smoothstep edges so a branch reads as two paths rather than one, and a Tidy that fans siblings out
  symmetrically around their parent instead of stacking them downward.
- **Category icons and colours** from the `--label` data-visualization ramp — the sanctioned
  categorical exception in `DESIGN.md` §2, confined to the node stripe and palette dot.
- **List search and pagination** (12 per page, client-side over the graph-less summaries), and a
  search plus pagination over the starter presets in the "Start from" picker (8 per page, with the
  blank-canvas card pinned to every page and a `n of total` count beside the panel title).

### Bugs fixed in this revision

- **`/api/datasets` was returning 500 for every caller.** A dataset written by the removed
  Parquet-backed `tabular` kind (commit b70baa3) failed `DatasetSummary` validation, and one stale
  manifest took the whole catalog down. Added `tabular → text_classification` and `table → csv` to
  the existing alias mechanism, and wired the alias validators onto `DatasetSummary` so a manifest
  that outlives its schema still lists.
- **`.visually-hidden` was never defined**, so the Import file input rendered as a raw
  `Choose File / No file chosen` control next to the styled button.
- **Two tests passed vacuously on a developer machine.** Both built `Settings` without stating the
  credentials they assert on, so the real `.env` leaked in — one asserted "no OpenRouter key" on a
  machine that has one. Both now state their own preconditions.

### Stage D and E deviations from the draft

- **`AdvancedParameterSpec` gained a `code` type**, rendered as a mono textarea. Custom layer bodies
  are the first param that cannot fit a single-line input, and adding a type kept the inspector on
  the one shared field renderer rather than forking it.
- **A new `Generation` advanced group** holds sampling temperature and length. It is registered in
  `app/ml/common/advanced.py`, the frontend `GROUP_ORDER`, and
  `test_advanced_training.py`'s documented set — that test caught the omission, which is what it
  exists for.
- **There is no generic Group/subgraph node.** Stacking is a `layers` param on a composite
  `transformer_block` node, which emits a Python loop rather than N copies of a subgraph. Generic
  nested subgraphs would have doubled the IR and emitter for one use case, and the palette also
  ships the individual primitives (RMSNorm, RoPE, MHA, SwiGLU) so a block can still be wired by hand
  and edited piece by piece — the `transformer_primitives` template does exactly that.
- **Keras ships no RMSNorm, RoPE, SwiGLU, or learned positional layer**, so the emitter generates
  them as helper classes above `build_model`, and only when the graph actually uses them. A plain
  CNN's generated file carries no transformer machinery.
- **A backbone-style causal mask node was dropped** in favour of a `causal` toggle on the attention
  and block nodes. Keras exposes it as `use_causal_mask=`; a separate node would have been a
  parameter pretending to be a layer.
- **Custom layers are hoisted once per class name.** Two nodes may share a class; emitting it twice
  would be a redefinition, so the first wins and later nodes reuse it.
- **Custom nodes may declare an output shape.** The analytic pass cannot read Python, so a blank
  declaration means "shape unchanged" (true of most custom layers, and it keeps downstream nodes
  resolvable) and anything else must be stated. Custom nodes always make the parameter estimate
  `None`.
- **`language_modeling` is a real `TaskType`**, resolving the spec's open question. It is grouped
  with NLP on the frontend (`isNlpTask` returns true, so the forms never offer it an image size)
  rather than with `llm_finetune`, which drives a different set of form branches entirely.
- **A separate `architecture_lm` family and runner.** The two graph runners share no dataset
  pipeline — one windows a corpus into next-token pairs, the other batches labelled images — so
  splitting them beat threading a mode flag through one runner. They share the command builder.
- **The LM runner has its own tokenizer**, word or character level, built from the training corpus
  and capped at `vocab_size`. It writes `tokenizer.json` beside the weights, because the vocabulary
  is not recoverable from the weights and the model is unusable without it.
- **LM sample generations reuse phase 14's `sample_generations.json`**, so the training detail page
  renders them with no new frontend code.
- **A from-scratch LM has no inference path.** `model_registry.predictor_for` raises a specific,
  actionable message rather than a bare "unsupported family". Samples on the run's detail page are
  the way to see what it learned.

### Stage B and C deviations from the draft

- **`@xyflow/react` is styled from `base.css` only**, not its `style.css` theme. Every visible
  surface — nodes, handles, edges, controls, minimap — is restyled from platform tokens, so no
  vendor theme enters the app and `frontend/DESIGN.md` §5's border-first rule holds on the canvas.
- **Architectures live under a `Catalog | Architectures` segmented control on `/models`**, not a
  new sidebar entry. Architectures *are* models; a top-level nav item would imply otherwise, and
  the sidebar's `pathname.startsWith("/models")` already lights up for the new routes.
- **The node inspector is the phase-13 `AdvancedField`**, exported from
  `frontend/features/training/advanced-settings.tsx` for reuse. This is the payoff of reusing
  `AdvancedParameterSpec` for node params: a node type declared on the backend gets its settings UI
  with no frontend change.
- **A "Preview classes" control lives in the studio header.** `units_from_dataset` heads have no
  resolvable width until a dataset is chosen, so the canvas needs a stand-in to show real shapes and
  parameter counts before a training run exists.
- **Training is one catalog option, `architecture_graph`**, with the graph chosen via
  `hyperparameters.architecture_id` — no `TrainingJobCreate` field, exactly as phase 13 intended.
  The key is stripped from the advanced payload before the runner sees it, since it is routing
  rather than a hyperparameter, and both `create_job` and the command builder reject a missing or
  broken architecture before any subprocess starts.
- **`_command_for_job` gained an optional `db` parameter.** Only the architecture family needs to
  read a row to build its command; every other family builds purely from the job's own parameters,
  so the session stays optional rather than coupling them all to the database.
- **The runner takes no `--image-size`.** Resolution comes from the built model's input shape, so
  the graph's Input node genuinely owns preprocessing and a form value cannot silently contradict
  the canvas. `TrainingJobCreate.image_size` is still sent (the schema requires it) and ignored.
- **`keras_common.py` was extracted** from `keras_classification_train.py` and is shared by both
  runners: dataset loading, `image_datasets`, optimizers, LR schedules, callbacks, class weights,
  and `write_evaluation`. Behavior of the existing runner is unchanged; the module-level names it
  used to own are re-exported from it so nothing that imported them breaks.
- **A graph-built model registers with `family="keras_classification"`.** The artifact is an
  ordinary `.keras` file, so testing, inference, and export need no knowledge that a canvas produced
  it — the whole point of choosing Keras as the compile target. `generated_model.py` is copied
  beside the weights for provenance.
- **Import from `.keras` and from a registry model is still deferred** (drafted for stage B, moved
  to a later pass). `.json` import, export, auto-layout, and templates all ship.
- **`POST /{id}/compile` remains unimplemented.** With training working end to end, the subprocess
  build check is now largely redundant — the analytic pass catches structural errors and a real
  training run is the authoritative build. It stays in the plan but is no longer on the critical
  path.

### Stage A deviations from the draft

All made either to match a registry/type constraint the draft did not account for, or because a
feature belongs to a later stage:

- **The stage A catalog omits Group, CustomLayer, CustomFunction, and every transformer primitive.**
  Those are stages D and E. What ships is I/O, Core, Convolution, Normalization, Recurrent, Merge,
  Regularization, and Backbone — 31 node types. `ArchitectureGraph` correspondingly has no
  `custom_nodes` or `groups` field yet; adding them is additive and needs no migration.
- **`POST /{id}/compile` is not implemented.** Stage A ships the analytic `/validate` only. The
  compile subprocess lands with the training runner in stage C, which is where the subprocess
  plumbing it needs already exists. `ArchitectureCompileResult` is likewise deferred.
- **Import accepts `.json` only.** `.keras`/registry-model import needs a TensorFlow subprocess to
  read the model config, so it lands in stage B alongside the UI that surfaces it. There is no
  `importers.py` yet; the JSON path lives in the service.
- **Bidirectional is a param on LSTM/GRU, not a wrapper node.** A node that wraps another node has
  no natural representation on a flat canvas, and this is the only Keras wrapper the palette needs.
- **Shape-valued params ride as comma-separated text** (`"224,224,3"`), parsed by
  `graph.parse_shape`. `AdvancedParameterSpec` has no tuple type, and inventing one would have
  meant a new field renderer on the frontend — the whole point of reusing the type. For the same
  reason `kernel_size`, `strides`, `pool_size`, and `padding` are single ints: square kernels and
  symmetric padding only. Non-square kernels would need a tuple type.
- **`MaxPool2D`/`AvgPool2D` spell "match the pool size" as `strides: 0`**, since the spec type has
  no unset value for a number. The emitter maps 0 back to Keras's `None`.
- **Arity is only enforced on nodes that reach the Output node.** A node the user has dropped but
  not wired up yet is a warning, not an error — otherwise placing a node would block saving or
  training an otherwise-complete model.
- **A pretrained backbone is emitted without `name=`.** `tf.keras.applications` derives its weights
  download filename from the model's name, so naming a backbone after its canvas node sends Keras
  looking for `<node>_notop.h5` and the ImageNet download 403s. Found by building the generated
  code rather than by reading it. The backbone keeps its stock name in `model.summary()`.
- **`total_params_estimate` is `None` whenever any contributing node's count is unknown** rather
  than a partial sum. A total silently missing a backbone's several million parameters is worse
  than no total. For graphs without a backbone the estimate matches TensorFlow exactly, which the
  slow tests assert.
- **The graph builders `node`/`edge`/`chain` live in `ml/architecture/templates.py`** and are shared
  by the templates and the test suite, so a graph in a test is constructed exactly like a real one.
- **Endpoints added beyond the draft table:** `GET /node-categories` (palette order, so the
  frontend does not hardcode a second copy), `GET /{id}/code/download` (attachment response), and
  `POST /{id}/validate` (validate a saved graph without re-posting it).
- **`NodeSpec.task_types` filters the palette.** An empty list means universal. Filtering is a
  palette hint only: a saved graph is never rejected for using a node the current task type would
  not have offered.
- **Unrelated bug fixed in `app/core/database.py`.** `_stamp_or_upgrade` stamped a matching
  pre-Alembic database at the *baseline* revision and then upgraded, which replays every migration
  since baseline against tables that already exist. That was invisible while baseline *was* head;
  this phase's migration is the first one after the baseline, so it surfaced immediately. An empty
  metadata diff means the live schema equals current ORM metadata — which is head by definition —
  so the stamp now goes at `head`.

### Decisions locked before drafting

- **Compile target is the Keras functional API.** A graph becomes a `tf.keras.Model`, so the existing
  Keras training runner, `.keras` artifacts, `KerasClassificationPredictor`, `model_registry`, testing,
  inference, and export all apply to graph-built models with no parallel pipeline.
- **LLM support is layer-level.** The palette carries transformer primitives (embedding, RoPE, MHA,
  RMSNorm, SwiGLU, LM head) that compose into a full transformer built from scratch. This is a
  research/teaching surface at toy scale — a from-scratch model trained on local hardware will not be
  competitive with a pretrained base. `specs/phase-14-llm-finetuning.md` remains the path for real
  LLM work; this phase does not replace it.
- **Custom code nodes hold real Python**, materialized into the generated module and executed in the
  training/compile subprocess. See [Custom code and trust](#custom-code-and-trust).
- **Canvas is `@xyflow/react`** (React Flow), a new pnpm dependency. Nodes are plain React components
  styled from `frontend/DESIGN.md` tokens; no vendor theme is imported.

## Goal

A visual, node-based editor under the Models menu where a user assembles a deep-learning or transformer
architecture by dragging layer nodes onto a canvas and wiring them together, sees live output shapes and
parameter counts, and then trains, tests, runs inference on, saves, and exports the result through the
platform's existing model pipeline. The same graph exports to runnable Python and imports back from a
saved file, and any node the palette does not cover can be written as custom Python inline.

## Scope

In:

- A versioned graph IR (JSON) as the single source of truth for an architecture.
- A server-declared node catalog covering core, convolutional, normalization, recurrent, merge,
  attention/transformer, augmentation, and pretrained-backbone nodes.
- Analytic validation and shape inference (no TensorFlow import), plus an authoritative subprocess
  compile check that builds the real model.
- Keras code generation: a standalone `generated_model.py` exposing `build_model(...)`.
- Canvas studio UI at `/models/architectures`, with palette, inspector, generated-code panel, and
  layer summary.
- Persistence in a new `architectures` table with per-save snapshots under `storage/`.
- Import from platform `.json`, from a Keras `model.to_json()` config, and from an existing registry
  model; export to `.json` and `.py`.
- Custom layer / custom function nodes holding Python.
- A new `architecture_graph` training family that trains a graph-built model and promotes it into the
  model registry like any other trained model.
- Group nodes with a `repeat` count, so a transformer block is authored once and stacked N times.

Out:

- A PyTorch emitter. The IR keeps emitters pluggable, but only Keras ships here.
- Distributed or multi-GPU training. Graph training uses the existing single-process runner path.
- Editing an architecture concurrently from two sessions. Last write wins, guarded by a version check.
- A full inference surface for from-scratch language models. Stage E ships training plus in-job sample
  generation; a `keras_lm` predictor on the `/inference` surface is deferred to a follow-up phase.
- Any change to phase 14 LLM fine-tuning. That path is untouched.
- Automatic hyperparameter search or architecture search.

## Interfaces

### API

New domain router `backend/app/api/routers/architectures.py`, registered in
`backend/app/api/routes.py` alongside the existing routers.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/architectures` | List, filtered by `project_id`, `task_type`. |
| `POST` | `/api/architectures` | Create blank, or from `template_id`. |
| `GET` | `/api/architectures/{id}` | Fetch graph + metadata. |
| `PUT` | `/api/architectures/{id}` | Save graph. Body carries `version`; a stale version returns 409. |
| `DELETE` | `/api/architectures/{id}` | Delete architecture and its snapshots. |
| `POST` | `/api/architectures/{id}/duplicate` | Copy as a new architecture. |
| `GET` | `/api/architectures/node-catalog` | Palette definitions, filtered by `task_type`. |
| `GET` | `/api/architectures/templates` | Starter graphs (small CNN, ResNet-ish block, BiLSTM classifier, decoder-only transformer). |
| `POST` | `/api/architectures/validate` | Analytic pass over a posted graph. Returns per-node shapes, param estimates, and issues. Stateless, so an unsaved canvas can validate. |
| `POST` | `/api/architectures/{id}/compile` | Subprocess TF build. Returns real shapes, exact param counts, `model.summary()` text, or the build traceback. |
| `GET` | `/api/architectures/{id}/code` | Generated Python as text. |
| `GET` | `/api/architectures/{id}/export` | `.json` download of the graph. |
| `POST` | `/api/architectures/import` | Multipart: `file` (`.json` or `.keras`) **or** `model_id`. Returns a new architecture. |

Training reuses the existing endpoints unchanged: `POST /api/training/jobs` with
`model_id: "architecture_graph"` and `hyperparameters.architecture_id`. No `TrainingJobCreate` change —
the dict was left open-ended for exactly this, per phase 13.

### Schemas

In `backend/app/schemas.py`:

- `ArchitectureNode` — `{id, type, label, position: {x, y}, params: dict, group_id: str | None}`
- `ArchitectureEdge` — `{id, source, source_port, target, target_port}`
- `ArchitectureCustomNode` — `{id, name, kind: "layer" | "function", code, params: list[AdvancedParameterSpec]}`
- `ArchitectureGraph` — `{schema_version, nodes, edges, custom_nodes, groups, training_defaults}`
- `Architecture` — `{id, project_id, name, description, task_type, framework, version, graph, created_at, updated_at}`
- `NodeSpec` — `{type, name, category, description, inputs, outputs, params: list[AdvancedParameterSpec], min_inputs, max_inputs}`
- `ArchitectureIssue` — `{severity: "error" | "warning", node_id: str | None, message}`
- `ArchitectureValidation` — `{ok, issues, node_shapes: dict[str, list], total_params_estimate, layer_count}`
- `ArchitectureCompileResult` — `{ok, summary, node_shapes, total_params, trainable_params, error}`

**Node params reuse `AdvancedParameterSpec` from phase 13.** That type already carries
`{key, label, type, default, min, max, step, options, help, group}` and the frontend already renders it
generically in `frontend/features/training/advanced-settings.tsx`. The node inspector is therefore the
existing field renderer pointed at a node's spec — a new node type gets its UI for free, and a param
cannot drift from what codegen consumes.

### Frontend surfaces

- `frontend/app/(platform)/models/architectures/page.tsx` — list, thin entrypoint.
- `frontend/app/(platform)/models/architectures/[architectureId]/page.tsx` — studio, thin entrypoint.
- `frontend/features/models/architectures/` — `architectures-page.tsx`, `studio-page.tsx`,
  `node-palette.tsx`, `node-inspector.tsx`, `canvas-nodes.tsx`, `code-panel.tsx`, `import-dialog.tsx`,
  `auto-layout.ts`, `graph-client.ts`. Revision 4 adds `use-history.ts` (undo stack), `clipboard.ts`
  (selection payloads, re-id on paste), `context-menu.tsx`, and `group-inspector.tsx`; revision 5
  adds `group-changes.ts` (the group-frame equivalent of `applyNodeChanges`, kept pure so the
  resize-axis rules are testable) and the shared `features/platform/ui/number-combo.tsx`.
- `frontend/lib/api/architectures.ts`, re-exported from `frontend/lib/api/index.ts` so
  `import { api } from "@/lib/api"` keeps working.
- The Models page header gains a segmented control, `Catalog | Architectures`. The sidebar keeps one
  Models entry; `pathname.startsWith("/models")` already covers the new routes.
- Platform selectors go in `frontend/app/styles/platform.css` under an `.arch-*` prefix.

### Backend modules

- `backend/app/ml/architecture/blocks.py` — the named block registry. One entry per family with the
  defaults its published config states and the `source` they were read from, plus
  `llm_block_structures`, which maps a block's parameters to the code structures it reaches. Consumed
  by `catalog`, `shapes`, and both emitters, so a family cannot mean one thing in the palette and
  another in the generated file.
- `backend/app/ml/architecture/keras_helpers.py` / `torch_helpers.py` — the generated layer source for
  those blocks, kept out of the emitters because it is reference implementation meant to be read in
  the file a user downloads.
- `backend/app/ml/architecture/catalog.py` — node specs, grouped by category.
- `backend/app/ml/architecture/graph.py` — IR dataclasses, parse, topological sort, cycle detection.
- `backend/app/ml/architecture/shapes.py` — pure-Python analytic shape inference. No TF import.
- `backend/app/ml/architecture/emit_keras.py` — graph → Python source.
- `backend/app/ml/architecture/importers.py` — Keras config → IR, registry model → IR.
- `backend/app/ml/architecture/layout.py` — layered auto-layout for imported graphs.
- `backend/app/services/architectures.py` — CRUD, snapshots, compile orchestration; singleton wired in
  `backend/app/container.py`.
- `backend/app/training/runners/architecture_compile.py` — subprocess build check.
- `backend/app/training/runners/architecture_train.py` — image training runner.
- `backend/app/training/runners/architecture_text_train.py` — text-classification training runner
  (revision 4). Shares the generated module and the `build_model(num_classes=...)` contract with the
  image runner; the dataset pipeline in front of it is `nlp/common.py`'s, and the artifacts it writes
  match `nlp/keras.py`'s so promotion and serving need no new branch beyond the family name.
- `backend/app/training/runners/keras_common.py` — dataset loading, metrics, callbacks, and CSV
  normalization extracted from `keras_classification_train.py` and shared by both runners. The existing
  runner keeps its current behavior and CLI; this is a refactor, not a rewrite. `image_datasets`
  permutes the training file list before building the pipeline (revision 5) and declares the
  resulting cardinality (revision 4); both matter to every image runner, not just the studio's.

### Storage and DB

New table `architectures`:

```
id           String(64)  PK
project_id   String(64)  index, nullable
name         String(120)
description  Text        nullable
task_type    String(64)  index
framework    String(32)  default "keras"
version      Integer     default 1
graph        JSON
created_at   DateTime    index
updated_at   DateTime
```

Alembic migration generated with `uv run alembic revision --autogenerate`, reviewed, and committed with
the model change, per `docs/ai/rules.md`.

Snapshots: every successful `PUT` writes `storage/architectures/{id}/v{version}.json` before bumping the
row. Generated code and compile output live in the run dir, never in git.

## Data Flow

```
Canvas edit
  → graph IR (client state)
  → POST /validate            → analytic shapes + issues, rendered on nodes and in the Issues tab
  → PUT  /architectures/{id}  → DB row + storage snapshot
  → POST /compile             → subprocess: emit .py, import, build, model.summary()
  → GET  /code                → generated_model.py, shown in the Code tab and downloadable
  → POST /api/training/jobs   (model_id="architecture_graph", hyperparameters.architecture_id=...)
        → service emits generated_model.py into run_dir
        → architecture_train.py imports it, builds, trains
        → best_model.keras / last_model.keras / metrics.json / results.csv in run_dir
        → promote → model_registry as source="trained"
        → /testing, /inference, /models export all work unchanged
```

The graph is the source of truth at every step. Generated code is always a pure function of the graph
and is regenerated rather than stored, so a hand-edit to exported code never silently diverges from what
trains. Editing exported code is a one-way door: the user owns the file from that point.

### Validation tiers

1. **Analytic** (`/validate`, synchronous, no TF). Structural checks — DAG-ness, exactly one reachable
   input and output path, port arity, orphan nodes — plus per-node shape propagation from rules declared
   on each `NodeSpec`. Fast enough to run on every canvas change, debounced.
2. **Compile** (`/compile`, subprocess). Emits the module, imports it, builds the model, returns the real
   summary. This is authoritative and catches anything the analytic pass approximates. It is a
   subprocess for the same reason the training runners are: TensorFlow must not be imported by the API
   process, per `AGENTS.md`.

### Code generation

Deterministic. Variable names derive from node ids, so regenerating an unchanged graph produces a
byte-identical file and diffs are meaningful.

```python
# Generated by Onestep AI Platform — architecture "small-cnn" (arch_7f2a, v4)
# Edits to this file are not read back into the studio.
import tensorflow as tf


def build_model(num_classes: int = 2, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x_conv_1 = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv_1")(inputs)
    x_bn_1 = tf.keras.layers.BatchNormalization(name="bn_1")(x_conv_1)
    x_act_1 = tf.keras.layers.Activation("relu", name="act_1")(x_bn_1)
    x_pool_1 = tf.keras.layers.GlobalAveragePooling2D(name="pool_1")(x_act_1)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="head")(x_pool_1)
    return tf.keras.Model(inputs, outputs, name="small_cnn")


if __name__ == "__main__":
    build_model().summary()
```

Custom layer classes are hoisted above `build_model`. `num_classes` is a parameter rather than a literal
so the same architecture retrains against a different dataset without editing the graph; the runner
passes the dataset's label count.

## Node Catalog

Every `NodeSpec` carries `kind: "layer" | "block"`. A layer is one operation; a block is a named
multi-layer structure that expands into many, and carries a `source` naming the config or paper its
defaults were read from.

| Category | Nodes |
| --- | --- |
| I/O | Input, Output |
| Vision blocks | ResNet stage, Inverted residual (MBConv), DenseNet block, Inception module, ConvNeXt block, ViT encoder block |
| LLM blocks | Llama, Qwen3, Mistral, Mixtral, Gemma 3, DeepSeek V3, Kimi K2, GPT-2, BERT, and the generic transformer block |
| Core | Dense, Activation, Dropout, Flatten, Reshape, Permute |
| Convolution | Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, MaxPool2D, AvgPool2D, GlobalAvgPool2D, GlobalMaxPool2D, UpSampling2D, ZeroPadding2D, Squeeze-and-excite |
| Normalization | BatchNormalization, LayerNormalization, GroupNormalization, RMSNorm |
| Recurrent | LSTM, GRU, Bidirectional |
| Merge | Add, Concatenate, Multiply, Subtract, Average |
| Transformer | Embedding, PositionalEmbedding, RotaryPositionalEmbedding, MultiHeadAttention (with `causal` flag), GroupedQueryAttention, MultiHeadLatentAttention, FeedForward, SwiGLU, GeGLU, MixtureOfExperts, PatchEmbedding, LMHead, GlobalAvgPool1D |
| Regularization | SpatialDropout2D, GaussianNoise, ActivityRegularization |
| Backbone | PretrainedBackbone — wraps `tf.keras.applications.*` with `include_top=False`, reusing `KERAS_APPLICATION_OPTIONS`, so a graph can start from transfer learning |
| Structure | Group (subgraph with a `repeat` count) |
| Custom | CustomLayer, CustomFunction |

`RMSNorm`, `RotaryPositionalEmbedding`, and `SwiGLU` have no stock Keras layer and emit small generated
classes from templates in `emit_keras.py`.

The Group node is what makes from-scratch transformers tractable: author one block
(RMSNorm → MHA(causal) → residual → RMSNorm → SwiGLU → residual), set `repeat: 12`, and codegen emits a
loop rather than twelve copies of the same subgraph.

## Custom code and trust

A CustomLayer node holds a Python class body; a CustomFunction node holds an expression body wrapped in
a `Lambda`. Both are written verbatim into the generated module.

- Custom code is **never** imported or executed by the FastAPI process. It runs only in the compile and
  training subprocesses, which is exactly how every existing runner already works.
- This is arbitrary code execution by design, appropriate for the current single-user local deployment
  and no further. `docs/ai/rules.md` already requires auth before multi-user or network-exposed
  deployment; this feature makes that requirement load-bearing rather than precautionary, and the
  studio surfaces a one-line notice on the custom node saying the code runs locally with full
  permissions.
- Compile failures surface the real traceback, trimmed to the generated file's frames, so a syntax error
  in a custom node points at the node rather than at the runner.
- Custom code is part of the graph JSON, so it travels through export, import, and snapshots. Importing
  a `.json` from an untrusted source imports its code; the import dialog says so and shows the custom
  node bodies before the import is confirmed.

## Import

| Source | Behavior |
| --- | --- |
| Platform `.json` | Validated against `schema_version`, positions preserved. |
| Keras `.keras` / `model.to_json()` | Config parsed in a subprocess, layers mapped to catalog node types, unmapped layers become CustomLayer nodes carrying their config so nothing is silently dropped. Auto-layout assigns positions. |
| Registry `model_id` | Same path, reading the registered `.keras` artifact. Lets a user open a trained or uploaded model, see its architecture, fork it, and retrain. |

Auto-layout is a pure layered pass — longest-path layering, in-layer ordering by median predecessor
position, fixed column and row pitch. No new dependency.

## Edge Cases

- **Cyclic graph** — analytic pass reports the participating node ids; save is allowed, compile and
  train are blocked.
- **Disconnected / orphan nodes** — warning, excluded from codegen, node dimmed on canvas.
- **No Input or no Output node** — error; train blocked.
- **Shape mismatch on a merge node** — analytic pass flags it on the merge node with both incoming
  shapes in the message.
- **Unknown node type in an imported graph** — imported as a disabled placeholder node with its raw
  params retained, flagged as an error, rather than dropped.
- **Stale save** — `PUT` with an older `version` returns 409 and the studio offers reload or overwrite.
- **Compile timeout** — subprocess capped at 120s; timeout is reported as a compile failure.
- **Custom code raising at build time** — traceback surfaced in the Issues tab and the toast layer, per
  `docs/ai/rules.md` "avoid hiding errors".
- **Training a graph whose architecture changed mid-run** — the run dir holds the emitted module, so a
  running job is pinned to the graph as of job creation. The job record stores the architecture id and
  version.
- **Deleting an architecture that trained models** — allowed; registered models keep their weights and
  their `architecture_id` metadata, which resolves to "deleted" in the UI.
- **`num_classes` mismatch** — the runner passes the dataset's label count; a graph whose Output node
  hardcodes a different unit count is rejected before training starts with an explicit message.
- **Very large graphs** — canvas virtualizes above 200 nodes; validation is debounced at 300ms.

## Implementation Stages

Ordered so each stage is independently shippable and testable.

- **A. Foundation (backend, no UI). Done.** IR, node catalog, DAG validation, analytic shape inference,
  Keras emitter, auto-layout, starter templates, CRUD service, migration, router. 109 tests, of which
  5 are `slow`-marked and build the generated modules under real TensorFlow — those assert that every
  predicted shape equals the built layer's actual shape and that the parameter estimate matches
  `model.count_params()` exactly.
- **B. Studio UI.** React Flow canvas, palette, inspector via the phase 13 field renderer, code panel,
  summary panel, save/load, export, import, auto-layout, templates.
- **C. Training.** Extract `keras_common.py`, add `architecture_graph` to the training catalog, add
  `architecture_train.py`, wire promotion into the registry, add the studio's Train action prefilling
  the existing training form.
- **D. Custom code. Done.** `custom_layer` (a hoisted Keras subclass) and `custom_function` (a
  single expression in a Lambda), a `code` param type rendered as a mono textarea, per-class
  hoisting, and optional declared output shapes.
- **E. Transformer primitives and causal LM. Done.** RoPE, RMSNorm, SwiGLU, feed-forward, causal
  multi-head attention, learned positional embeddings, LM head, GlobalAvgPool1D, and a composite
  `transformer_block` with a repeat count. Plus the `architecture_lm` family: a word/char tokenizer
  over a text corpus, next-token windowing, perplexity, and temperature-sampled generations written
  to `sample_generations.json`.

## Acceptance Criteria

Behavior:

- A user creates an architecture from the Models page, drags Input → Conv2D → BatchNorm → ReLU →
  GlobalAvgPool → Dense onto the canvas, wires it, sees per-node output shapes and a total parameter
  count, saves it, and trains it against a project dataset from the existing training surface.
- The resulting model appears in the model catalog as a trained model and is usable on `/testing` and
  `/inference` with no code path specific to graph-built models.
- Exported `generated_model.py` runs standalone: `python generated_model.py` prints a model summary.
- Exporting an architecture to `.json` and importing that file reproduces an identical graph, positions
  included.
- Importing a trained `.keras` model produces an editable graph whose compile summary matches the
  original model's layer count and parameter count.
- A CustomLayer node with a valid Keras layer subclass compiles, trains, and round-trips through export
  and import. An invalid one reports its traceback against that node.
- A decoder-only transformer built from Embedding + RoPE + causal MHA + RMSNorm + SwiGLU + LMHead, with
  a Group node repeated N times, compiles and trains to decreasing loss on a small text dataset.

Tests:

- `backend/tests/test_architecture_graph.py` — parse, topological sort, cycle detection, port arity.
- `backend/tests/test_architecture_shapes.py` — analytic shapes across conv/pool/merge/attention chains,
  including deliberate mismatches.
- `backend/tests/test_architecture_emit.py` — golden-file generated source for fixture graphs, including
  a Group repeat and a custom node.
- `backend/tests/test_architecture_api.py` — CRUD, stale-version 409, validate, import from `.json`.
- `backend/tests/test_architecture_train.py` — the runner builds and trains a tiny graph for one epoch on
  a synthetic dataset, and writes the artifacts `_register_training_model` expects.
- `frontend/features/models/architectures/*.test.tsx` — graph reducer, auto-layout determinism, palette
  filtering by task type.

Manual checks:

- Studio renders correctly in light and dark, uses no hardcoded color, and adds no `box-shadow` to a
  resting panel.
- Canvas is keyboard-navigable: node selection, delete, and inspector focus without a pointer.
- Every backend failure surfaces in both the Issues panel and the toast layer.

## Validation

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

Plus one end-to-end smoke run: build a small CNN in the studio, train one epoch against a local dataset,
and confirm the promoted model runs on `/inference`.

## Open Questions

- Whether trained-model → architecture import should also reconstruct weights into the studio, or stay
  architecture-only. Drafted as architecture-only; weights stay with the registered model.
- Whether `language_modeling` should be added to the project task-type vocabulary or stay internal to
  the architecture studio until a `keras_lm` predictor exists.
