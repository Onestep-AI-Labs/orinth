"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Background,
  BackgroundVariant,
  ConnectionLineType,
  ControlButton,
  Controls,
  MiniMap,
  Panel,
  PanOnScrollMode,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  SelectionMode,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange
} from "@xyflow/react";
import "@xyflow/react/dist/base.css";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  ChevronDown,
  Clipboard,
  ClipboardPaste,
  Code2,
  Copy,
  Download,
  Group,
  Maximize,
  Maximize2,
  Minimize2,
  Play,
  Plus,
  Redo2,
  Save,
  Scissors,
  SlidersHorizontal,
  SquareDashedMousePointer,
  Trash2,
  Undo2,
  Unlink,
  Wand2,
  X,
  ZoomIn,
  ZoomOut
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import {
  Button,
  ButtonLink,
  MutationError,
  NumberInput,
  IconButton,
  PageSkeleton
} from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import type { ArchitectureIssue, NodeSpec } from "@/types/api";
import type { StoredGroup } from "./graph-state";
import type { Framework } from "@/lib/api/architectures";
import {
  autoLayout,
  freeSpot,
  COLUMN_PITCH,
  NODE_HEIGHT,
  NODE_WIDTH
} from "./auto-layout";
import { useResizable } from "./use-resizable";
import { useHistory, type GraphSnapshot } from "./use-history";
import { instantiate, parsePayload, selectionPayload, serializePayload } from "./clipboard";
import { applyGroupChanges, isGroupEdit, resizePhase } from "./group-changes";
import { CanvasContextMenu, type ContextMenuItem, type ContextMenuState } from "./context-menu";
import { CanvasNode, NODE_EXTEND_EVENT } from "./canvas-node";
import {
  ActionEdge,
  EDGE_DELETE_EVENT,
  EDGE_HOVER_EVENT,
  EDGE_INSERT_EVENT,
  type ActionEdgeData
} from "./action-edge";
import { SettingsModal } from "./settings-modal";
import { GroupFrame, TITLE_DEFAULTS } from "./group-frame";
import { GroupInspector } from "./group-inspector";
import { CodePanel } from "./code-panel";
import { NodeInspector } from "./node-inspector";
import { NodePalette } from "./node-palette";
import { categoryColorVar } from "./node-visuals";
import {
  defaultParams,
  edgeId,
  formatCount,
  issuesByNode,
  nextNodeId,
  toFlow,
  toGraph,
  type ArchFlowNode
} from "./graph-state";

const nodeTypes = { arch: CanvasNode, group: GroupFrame };
const edgeTypes = { action: ActionEdge };
const VALIDATE_DEBOUNCE_MS = 350;
/**
 * How long the edge hover pill survives after the pointer leaves the wire.
 *
 * The pill sits on the line it belongs to, so moving onto a button fires
 * `mouseleave` on the path a frame before `pointerenter` on the button. Without
 * the grace period the affordance blinks out from under the pointer.
 */
const EDGE_HOVER_GRACE_MS = 90;
/** How long after the viewport settles the minimap stays up before fading. */
const MINIMAP_IDLE_MS = 1400;
/** Breathing room between a framed selection's nodes and the frame's edge. */
const GROUP_PADDING = 28;
/** Widened hit area on an edge, so selecting a wire does not need pixel aim. */
const EDGE_INTERACTION_WIDTH = 18;

/** Ctrl on Windows and Linux, ⌘ on a Mac — used only for shortcut captions. */
function accelerator(key: string): string {
  if (typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)) {
    return `⌘${key}`;
  }
  return `Ctrl+${key}`;
}

/** The bounding box of a set of nodes, padded, for framing a selection. */
function boundsOf(nodes: ArchFlowNode[]) {
  const left = Math.min(...nodes.map((node) => node.position.x));
  const top = Math.min(...nodes.map((node) => node.position.y));
  const right = Math.max(...nodes.map((node) => node.position.x + (node.width ?? NODE_WIDTH)));
  const bottom = Math.max(...nodes.map((node) => node.position.y + (node.height ?? NODE_HEIGHT)));
  return {
    x: left - GROUP_PADDING,
    y: top - GROUP_PADDING * 1.6,
    width: right - left + GROUP_PADDING * 2,
    height: bottom - top + GROUP_PADDING * 2.6
  };
}

/** Minimap dots inherit the node's category colour, so the map reads as the canvas. */
function minimapColor(node: { data?: unknown; type?: string }) {
  if (node.type === "group") return "transparent";
  const spec = (node.data as { spec?: { category?: string } } | undefined)?.spec;
  return categoryColorVar(spec?.category ?? "Core");
}

/** Whether a keystroke landed in a text field, where editor shortcuts must not fire. */
function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable;
}

/**
 * Whether a keystroke landed on something Space already means something to.
 *
 * Space holds the pan tool, which needs `preventDefault` to stop the page
 * scrolling — but Space is also how a focused button is pressed, so swallowing
 * it everywhere would break every toolbar control by keyboard.
 */
function isSpaceConsumer(target: EventTarget | null): boolean {
  if (isTextEntry(target)) return true;
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(target.closest("button, a, [role='menuitem'], [role='separator']"));
}

/**
 * What the layer panel will do with the next node picked out of it.
 *
 * The panel is one component serving three entry points — the canvas **+**, a
 * right-click on empty space, and the **+** on a wire — and each drops the node
 * somewhere different. Carrying that in the open/closed state keeps the palette
 * itself ignorant of all three.
 */
type PaletteIntent =
  | { kind: "canvas" }
  | { kind: "at"; position: { x: number; y: number } }
  | { kind: "edge"; edgeId: string }
  | { kind: "extend"; nodeId: string; side: "in" | "out" };

const PALETTE_TITLES: Record<PaletteIntent["kind"], string> = {
  canvas: "Add a node",
  at: "Add a node here",
  edge: "Insert on this connection",
  extend: "Extend the chain"
};

export function ArchitectureStudioPage({ architectureId }: { architectureId: string }) {
  return (
    <ReactFlowProvider>
      <Studio architectureId={architectureId} />
    </ReactFlowProvider>
  );
}

function Studio({ architectureId }: { architectureId: string }) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { projectId } = useProject();
  const { screenToFlowPosition, fitView, zoomIn, zoomOut } = useReactFlow();

  const [nodes, setNodes] = useState<ArchFlowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [selectedGroupIds, setSelectedGroupIds] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [version, setVersion] = useState(1);
  const [dirty, setDirty] = useState(false);
  const [numClasses, setNumClasses] = useState(2);
  const [tab, setTab] = useState<"issues" | "code">("issues");
  const [loaded, setLoaded] = useState(false);
  // Canvas real estate is the scarce resource in a node editor. Nothing that is
  // not the canvas holds a column: the layer panel is an overlay opened on
  // demand, settings are a dialog opened on the thing being edited.
  const [fullscreen, setFullscreen] = useState(false);
  const [showDrawer, setShowDrawer] = useState(true);
  /** Non-null while the layer panel is open; carries where the pick lands. */
  const [paletteIntent, setPaletteIntent] = useState<PaletteIntent | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  /**
   * Space is held: drag pans instead of marquee-selecting.
   *
   * Dragging on empty canvas selects, unconditionally — that is the gesture the
   * pointer is for. Panning is the trackpad's job (two fingers) or the middle
   * mouse button's, and Space is the escape hatch for a one-button mouse.
   */
  const [spacePan, setSpacePan] = useState(false);
  /** Which wire is under the pointer, so it can show its own edit buttons. */
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  /**
   * The minimap is only useful while you are lost, which is while you are
   * moving. It fades out when the viewport settles and comes back on the next
   * pan or zoom, so a parked canvas gets its bottom-right corner back.
   */
  const [minimapAwake, setMinimapAwake] = useState(false);
  const [groups, setGroups] = useState<StoredGroup[]>([]);
  const [framework, setFramework] = useState<Framework>("keras");
  const [menu, setMenu] = useState<ContextMenuState | null>(null);
  // The two remaining pane edges are draggable and remember where they were
  // left: the overlay panel's width and the bottom drawer's height.
  const palettePane = useResizable("palette", 264, { min: 200, max: 460 });
  const drawerPane = useResizable("drawer", 168, { min: 132, max: 560, axis: "vertical", invert: true });

  const architectureQuery = useQuery({
    queryKey: ["architecture", architectureId],
    queryFn: () => api.architecture(architectureId)
  });
  const catalogQuery = useQuery({
    queryKey: ["architecture-node-catalog"],
    queryFn: () => api.architectureNodeCatalog()
  });
  const categoriesQuery = useQuery({
    queryKey: ["architecture-node-categories"],
    queryFn: () => api.architectureNodeCategories()
  });

  const specs = useMemo(
    () => new Map((catalogQuery.data ?? []).map((spec) => [spec.type, spec])),
    [catalogQuery.data]
  );

  // --- undo / redo ---------------------------------------------------------

  // The history stack reads and writes whole graphs, so it needs the committed
  // state from inside event handlers that also set it. An effect-synced ref is
  // the only version of that which is safe under concurrent rendering.
  const canvasRef = useRef<HTMLDivElement>(null);

  const stateRef = useRef<GraphSnapshot>({ nodes: [], edges: [], groups: [] });
  useEffect(() => {
    stateRef.current = { nodes, edges, groups };
  }, [nodes, edges, groups]);

  const readSnapshot = useCallback(() => stateRef.current, []);
  const applySnapshot = useCallback((snapshot: GraphSnapshot) => {
    setNodes(snapshot.nodes);
    setEdges(snapshot.edges);
    setGroups(snapshot.groups);
    setDirty(true);
  }, []);
  // Destructured because the hook returns a fresh object each render: closing
  // over `history` would re-subscribe the window key listener every frame.
  const { take, undo, redo, reset: resetHistory, canUndo, canRedo } = useHistory(
    readSnapshot,
    applySnapshot
  );

  const markDirty = useCallback(() => setDirty(true), []);
  /** Snapshot the pre-change state, then mark the graph unsaved. */
  const commit = useCallback(() => {
    take();
    markDirty();
  }, [take, markDirty]);

  // Hydrate the canvas once, after both the graph and the catalog have landed.
  // Re-running on every refetch would discard in-progress edits.
  useEffect(() => {
    if (loaded || !architectureQuery.data || !catalogQuery.data) return;
    const flow = toFlow(architectureQuery.data.graph, specs);
    setNodes(flow.nodes);
    setEdges(flow.edges);
    setName(architectureQuery.data.name);
    setVersion(architectureQuery.data.version);
    // The class count belongs to the architecture, not the session: a graph
    // whose head is sized from the dataset is meaningless on the canvas until
    // you say how many classes to preview, and re-typing it every visit is
    // exactly the complaint that "small CNN always shows 2 classes" describes.
    const defaults = architectureQuery.data.graph.training_defaults ?? {};
    const stored = defaults.num_classes_preview;
    if (typeof stored === "number" && stored > 0) setNumClasses(stored);
    if (Array.isArray(defaults.groups)) setGroups(defaults.groups as StoredGroup[]);
    setLoaded(true);
    // Loading a graph is not an edit, so it must not be undoable back to blank.
    resetHistory();
  }, [architectureQuery.data, catalogQuery.data, specs, loaded, resetHistory]);

  /**
   * The model, without the annotations drawn over it.
   *
   * Group frames are deliberately *not* a dependency. They used to be, because
   * they ride to the server inside `training_defaults` — but that meant nudging
   * a purely decorative rectangle rebuilt every node and edge in the IR and
   * re-armed shape validation, sixty times a second while dragging its corner.
   * That round trip is what made resizing a frame lag the pointer.
   *
   * Frames are merged back in at save time by `graphToSave`, which is the only
   * moment their geometry actually has to travel anywhere.
   */
  const graph = useMemo(
    () => toGraph(nodes, edges, { num_classes_preview: numClasses }),
    [nodes, edges, numClasses]
  );

  // Debounced so dragging a node does not fire a request per frame.
  const [validation, setValidation] = useState<{
    ok: boolean;
    issues: ArchitectureIssue[];
    shapes: Record<string, (number | null)[] | null>;
    params: number | null;
    layers: number;
  }>({ ok: true, issues: [], shapes: {}, params: null, layers: 0 });

  useEffect(() => {
    if (!loaded || nodes.length === 0) return;
    const timer = window.setTimeout(async () => {
      try {
        const result = await api.validateArchitectureGraph(graph, numClasses);
        setValidation({
          ok: result.ok,
          issues: result.issues,
          shapes: result.node_shapes,
          params: result.total_params_estimate ?? null,
          layers: result.layer_count
        });
      } catch (error) {
        // Validation is advisory; a transient failure must not blank the canvas.
        toast.error(error instanceof Error ? error.message : "Validation failed");
      }
    }, VALIDATE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [graph, numClasses, loaded, nodes.length]);

  // Fold validation results back onto the nodes so each one renders its own
  // shape and flags. Kept out of node state so validation never marks the
  // graph dirty.
  const nodeIssues = useMemo(() => issuesByNode(validation.issues), [validation.issues]);

  // Group frames are drawn as React Flow nodes so they pan and zoom with the
  // canvas, but they are prepended (behind everything) and are never part of
  // `nodes`, so no validator or emitter has to know they exist. Selection is
  // held here rather than inside React Flow because `groups` is the source of
  // truth these are rebuilt from on every render.
  const groupNodes = useMemo(
    () =>
      groups.map((group) => ({
        id: group.id,
        type: "group" as const,
        position: { x: group.x, y: group.y },
        width: group.width,
        height: group.height,
        // `measured` as well as `width`/`height`, because these node objects are
        // rebuilt from `groups` on every change: React Flow copies `measured`
        // from the node it is handed, so omitting it blanked the frame's known
        // size on each pointer event and left the resizer starting its next drag
        // from zero. NodeResizer reads `measured` and nothing else.
        measured: { width: group.width, height: group.height },
        selectable: true,
        draggable: true,
        selected: selectedGroupIds.includes(group.id),
        data: {
          title: group.title,
          tint: group.tint,
          // Defaults resolved here rather than in the frame so a graph saved
          // before titles were movable still renders one.
          titleX: group.titleX ?? TITLE_DEFAULTS.x,
          titleY: group.titleY ?? TITLE_DEFAULTS.y,
          titleSize: group.titleSize ?? TITLE_DEFAULTS.size,
          titleTint: group.titleTint ?? group.tint
        }
      })),
    [groups, selectedGroupIds]
  );

  /**
   * Edges that leave a fan-out or enter a merge are drawn as curves; a plain
   * link in a chain stays a step.
   *
   * On a straight chain a step edge reads as "one path". At a branch it reads
   * as one path too, which is the opposite of the truth — the residual in a
   * ResNet block and the router in an MoE layer are exactly where the picture
   * has to show two things happening. Bezier separates them visually.
   *
   * Every wire is the same `action` edge type; the curve/step choice and the
   * hover state ride in its `data` so one component draws both.
   */
  const degrees = useMemo(() => {
    const out = new Map<string, number>();
    const into = new Map<string, number>();
    for (const edge of edges) {
      out.set(edge.source, (out.get(edge.source) ?? 0) + 1);
      into.set(edge.target, (into.get(edge.target) ?? 0) + 1);
    }
    return { out, into };
  }, [edges]);

  // Keyed on `hoveredEdge` as well as `edges`, so it rebuilds on every hover —
  // which is why the degree counts above are their own memo rather than being
  // recomputed in here.
  const routedEdges = useMemo(
    () =>
      edges.map((edge) => {
        const branching =
          (degrees.out.get(edge.source) ?? 0) > 1 ||
          (degrees.into.get(edge.target) ?? 0) > 1;
        return {
          ...edge,
          type: "action",
          className: branching ? "arch-edge-branch" : undefined,
          // A 1.5px wire is a 1.5px click target without this.
          interactionWidth: EDGE_INTERACTION_WIDTH,
          animated: false,
          data: { branching, hovered: edge.id === hoveredEdge } satisfies ActionEdgeData
        };
      }),
    [edges, degrees, hoveredEdge]
  );

  const decoratedNodes = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          shape: validation.shapes[node.id] ?? null,
          issues: nodeIssues.get(node.id) ?? [],
          // A side with nothing on it gets a stub `+`.
          hasIncoming: degrees.into.has(node.id),
          hasOutgoing: degrees.out.has(node.id)
        }
      })),
    [nodes, validation.shapes, nodeIssues, degrees]
  );

  // One array identity per change, rather than a fresh one on every render.
  // React Flow diffs this prop against its store; rebuilding it during a drag
  // makes every frame reconcile the whole graph.
  const flowNodes = useMemo(
    () => [...groupNodes, ...decoratedNodes],
    [groupNodes, decoratedNodes]
  );

  const saveMutation = useMutation({
    mutationFn: () =>
      api.updateArchitecture(architectureId, {
        name,
        // Frames rejoin the payload here — the one place their geometry has to
        // travel. See the `graph` memo for why they are kept out of it.
        graph: toGraph(nodes, edges, { num_classes_preview: numClasses, groups }),
        version
      }),
    onSuccess: async (saved) => {
      setVersion(saved.version);
      setDirty(false);
      toast.success("Architecture saved");
      await queryClient.invalidateQueries({ queryKey: ["architectures"] });
    }
  });

  const codeQuery = useQuery({
    queryKey: ["architecture-code", architectureId, version, numClasses, framework, tab],
    queryFn: () => api.architectureCode(architectureId, numClasses, framework),
    enabled: tab === "code" && loaded,
    retry: false
  });

  /**
   * Add a titled frame. With nodes selected it wraps them, which is the useful
   * case — you have just built an encoder and want to say so, and drawing a
   * rectangle around it by hand afterwards is busywork.
   */
  const addGroup = useCallback(() => {
    const id = `group_${Date.now().toString(36)}`;
    const chosen = nodes.filter((node) => node.selected);
    const bounds = chosen.length > 0 ? boundsOf(chosen) : null;
    commit();
    setGroups((current) => [
      ...current,
      {
        id,
        title: chosen.length > 0 ? "New group" : "Untitled group",
        tint: current.length % 8,
        x: bounds ? bounds.x : 40 + current.length * 24,
        y: bounds ? bounds.y : 40 + current.length * 24,
        width: bounds ? bounds.width : 420,
        height: bounds ? bounds.height : 280
      }
    ]);
  }, [nodes, commit]);

  const patchGroup = useCallback(
    (id: string, patch: Partial<StoredGroup>) => {
      commit();
      setGroups((current) =>
        current.map((group) => (group.id === id ? { ...group, ...patch } : group))
      );
    },
    [commit]
  );

  // The frame renders inside React Flow and cannot reach this state directly,
  // so title edits come back as an event.
  useEffect(() => {
    const onTitle = (event: Event) => {
      const { id, title } = (event as CustomEvent).detail;
      setGroups((current) =>
        current.map((group) => (group.id === id ? { ...group, title } : group))
      );
      markDirty();
    };
    // The title is dragged inside the frame, so it is clamped to the frame:
    // a label parked outside the rectangle it names is not an annotation.
    const onTitleMove = (event: Event) => {
      const { id, titleX, titleY } = (event as CustomEvent).detail;
      setGroups((current) =>
        current.map((group) => {
          if (group.id !== id) return group;
          const size = group.titleSize ?? TITLE_DEFAULTS.size;
          return {
            ...group,
            titleX: Math.round(Math.min(Math.max(titleX, 0), Math.max(group.width - 24, 0))),
            titleY: Math.round(
              Math.min(Math.max(titleY, 0), Math.max(group.height - size - 8, 0))
            )
          };
        })
      );
      markDirty();
    };
    window.addEventListener("arch-group-title", onTitle);
    window.addEventListener("arch-group-title-move", onTitleMove);
    return () => {
      window.removeEventListener("arch-group-title", onTitle);
      window.removeEventListener("arch-group-title-move", onTitleMove);
    };
  }, [markDirty]);

  /**
   * Whether a NodeResizer drag is open, so one drag is one undo step.
   *
   * Only that. Nothing about a resize is buffered: `nodes` is a controlled prop,
   * so React Flow's store is rebuilt from it and never applies a change on its
   * own — every intermediate size has to go back through `setGroups` or the
   * rectangle simply does not move until the pointer is released. Holding the
   * changes until then is what made stretching feel broken.
   */
  const resizingRef = useRef(false);

  // The id set is read inside `onNodesChange` but must not be a dependency of
  // it: `groups` changes on every pointer event of a drag, and rebuilding the
  // handler that often re-registers it with React Flow mid-drag.
  const groupIdsRef = useRef<Set<string>>(new Set());
  groupIdsRef.current = useMemo(() => new Set(groups.map((group) => group.id)), [groups]);

  const onNodesChange = useCallback(
    (changes: NodeChange<ArchFlowNode>[]) => {
      // Group frames share the change stream; route theirs to `groups` and
      // let the rest fall through to the model nodes.
      const groupIds = groupIdsRef.current;
      const groupChanges = changes.filter(
        (change) => "id" in change && groupIds.has(change.id)
      );

      const phase = resizePhase(changes);
      if (phase === "start" && !resizingRef.current) {
        resizingRef.current = true;
        commit();
      }

      if (groupChanges.length > 0) {
        setGroups((current) => applyGroupChanges(current, groupChanges));
        if (isGroupEdit(groupChanges)) markDirty();
      }

      if (phase === "end") resizingRef.current = false;

      const nodeChanges = changes.filter(
        (change) => !("id" in change) || !groupIds.has(change.id)
      );
      setNodes((current) => applyNodeChanges(nodeChanges, current));
      if (nodeChanges.some((change) => change.type !== "select" && change.type !== "dimensions")) {
        markDirty();
      }

      // Selection drives which inspector the right rail shows, so it is tracked
      // for both kinds and cleared when the thing it points at is deselected.
      for (const change of changes) {
        if (change.type !== "select") continue;
        const isGroup = groupIds.has(change.id);
        if (change.selected) {
          if (isGroup) {
            setSelectedGroupIds((current) =>
              current.includes(change.id) ? current : [...current, change.id]
            );
            setSelectedId("");
          } else {
            setSelectedId(change.id);
            setSelectedGroupIds([]);
          }
        } else if (isGroup) {
          setSelectedGroupIds((current) => current.filter((id) => id !== change.id));
        } else {
          setSelectedId((current) => (current === change.id ? "" : current));
        }
      }
    },
    [markDirty, commit]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (changes.some((change) => change.type === "remove")) commit();
      setEdges((current) => applyEdgeChanges(changes, current));
      if (changes.some((change) => change.type !== "select")) markDirty();
    },
    [markDirty, commit]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      commit();
      // No `type` here: `routedEdges` assigns it, and a stale one stored on the
      // edge would only be overwritten a render later.
      setEdges((current) =>
        addEdge({ ...connection, id: edgeId(connection.source, connection.target) }, current)
      );
    },
    [commit]
  );

  /**
   * A clear spot near the middle of what is currently on screen.
   *
   * A node added from the palette used to land at a fixed flow position near
   * the origin. Pan two screens right to work on the tail of a graph, add a
   * layer, and it appeared off-view behind you — the add looked like it had
   * failed. The left edge is inset past the layer panel while that is open, so
   * "centre" means the centre of the part you can actually see; the bias below
   * it clears the floating toolbar, and `freeSpot` walks down from there so the
   * node never lands on one that is already sitting in the middle of the view.
   */
  const placeInView = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return { x: 320, y: 120 };
    const left = rect.left + (paletteIntent ? palettePane.size : 0);
    const centre = screenToFlowPosition({
      x: (left + rect.right) / 2,
      y: (rect.top + rect.bottom) / 2
    });
    return freeSpot(
      { x: centre.x - NODE_WIDTH / 2, y: centre.y - NODE_HEIGHT / 2 + 48 },
      nodes
    );
  }, [screenToFlowPosition, paletteIntent, palettePane.size, nodes]);

  const addNode = useCallback(
    (spec: NodeSpec, position?: { x: number; y: number }) => {
      // The id and position are derived before the updater runs: a state
      // updater must stay pure, and React invokes it twice under StrictMode.
      const id = nextNodeId(spec.type, nodes.map((node) => node.id));
      // A dropped node goes exactly where it was dropped. One added from the
      // panel gets the middle of the view, biased a little below centre so it
      // clears the floating toolbar, then walked down until it is not on top
      // of something.
      const spot = position ?? placeInView();
      commit();
      setNodes((current) => [
        ...current.map((node) => ({ ...node, selected: false })),
        {
          id,
          type: "arch" as const,
          position: spot,
          selected: true,
          data: {
            spec,
            params: defaultParams(spec),
            label: spec.name,
            shape: null,
            issues: []
          }
        }
      ]);
      setSelectedId(id);
      setSelectedGroupIds([]);
    },
    [nodes, commit, placeInView]
  );

  /**
   * Drop a node into the middle of an existing connection.
   *
   * `a → b` becomes `a → new → b` in one action. Doing it by hand is delete the
   * wire, add the node, draw two wires — three operations for the single most
   * common edit there is on a graph that is mostly a chain.
   *
   * The graph is re-laid-out afterwards. A node dropped at the midpoint of a
   * wire lands *on* both its neighbours — the gap between two columns is not a
   * node wide — so leaving it there means every insert is followed by dragging
   * the rest of the chain out of the way by hand.
   */
  const insertOnEdge = useCallback(
    (spec: NodeSpec, targetEdgeId: string) => {
      const edge = edges.find((item) => item.id === targetEdgeId);
      if (!edge) return;
      const id = nextNodeId(
        spec.type,
        nodes.map((node) => node.id)
      );
      const nextEdges = [
        ...edges.filter((item) => item.id !== targetEdgeId),
        { id: edgeId(edge.source, id), source: edge.source, target: id },
        { id: edgeId(id, edge.target), source: id, target: edge.target }
      ];
      const nextNodes: ArchFlowNode[] = [
        ...nodes.map((node) => ({ ...node, selected: false })),
        {
          id,
          type: "arch" as const,
          // Overwritten by the layout below; a real position first so the node
          // never renders at the origin for a frame.
          position: { x: 0, y: 0 },
          selected: true,
          data: {
            spec,
            params: defaultParams(spec),
            label: spec.name,
            shape: null,
            issues: []
          }
        }
      ];
      commit();
      setNodes(autoLayout(nextNodes, nextEdges));
      setEdges(nextEdges);
      setSelectedId(id);
      setSelectedGroupIds([]);
      setHoveredEdge(null);
    },
    [nodes, edges, commit]
  );

  /**
   * Add a node onto a free handle, wired to the one it came from.
   *
   * The stub `+` beside an unconnected handle is the answer to "what now" on a
   * node the graph dead-ends at, so it has to produce a *connected* node — an
   * unwired box placed nearby would just move the same dead end one column over.
   */
  const extendFrom = useCallback(
    (spec: NodeSpec, nodeId: string, side: "in" | "out") => {
      const anchor = nodes.find((node) => node.id === nodeId);
      if (!anchor) return;
      const id = nextNodeId(
        spec.type,
        nodes.map((node) => node.id)
      );
      // One column along, on the anchor's row: the chain reads left to right,
      // so an extension belongs where Tidy would have put it anyway — unless
      // that column is already taken, in which case it drops to clear space.
      const position = freeSpot(
        {
          x: anchor.position.x + (side === "out" ? COLUMN_PITCH : -COLUMN_PITCH),
          y: anchor.position.y
        },
        nodes
      );
      commit();
      setNodes((current) => [
        ...current.map((node) => ({ ...node, selected: false })),
        {
          id,
          type: "arch" as const,
          position,
          selected: true,
          data: {
            spec,
            params: defaultParams(spec),
            label: spec.name,
            shape: null,
            issues: []
          }
        }
      ]);
      setEdges((current) => [
        ...current,
        side === "out"
          ? { id: edgeId(nodeId, id), source: nodeId, target: id }
          : { id: edgeId(id, nodeId), source: id, target: nodeId }
      ]);
      setSelectedId(id);
      setSelectedGroupIds([]);
    },
    [nodes, commit]
  );

  /**
   * A node picked out of the layer panel, routed by why the panel is open.
   *
   * Whichever route it took, the pick ends the same way: the panel closes and
   * the new node's settings open. Picking a layer is one half of adding it —
   * a `Conv2D` with catalog defaults is rarely the `Conv2D` you wanted — and
   * leaving the panel open over the node you just made meant closing it by hand
   * before you could see what you had done.
   *
   * Dragging from the panel is deliberately exempt (see the canvas `onDrop`):
   * that gesture is about placing something exactly, and interrupting it with a
   * dialog would fight the run of drops it usually belongs to.
   */
  const handlePaletteAdd = useCallback(
    (spec: NodeSpec) => {
      if (!paletteIntent || paletteIntent.kind === "canvas") addNode(spec);
      else if (paletteIntent.kind === "at") addNode(spec, paletteIntent.position);
      else if (paletteIntent.kind === "edge") insertOnEdge(spec, paletteIntent.edgeId);
      else extendFrom(spec, paletteIntent.nodeId, paletteIntent.side);
      setPaletteIntent(null);
      setSettingsOpen(true);
    },
    [paletteIntent, addNode, insertOnEdge, extendFrom]
  );

  // --- wire hover affordance ------------------------------------------------

  const minimapTimerRef = useRef<number | undefined>(undefined);
  /** Show the minimap, and restart the countdown that hides it again. */
  const wakeMinimap = useCallback(() => {
    window.clearTimeout(minimapTimerRef.current);
    setMinimapAwake(true);
    minimapTimerRef.current = window.setTimeout(
      () => setMinimapAwake(false),
      MINIMAP_IDLE_MS
    );
  }, []);
  useEffect(() => () => window.clearTimeout(minimapTimerRef.current), []);

  const hoverTimerRef = useRef<number | undefined>(undefined);
  const setEdgeHover = useCallback((id: string | null) => {
    window.clearTimeout(hoverTimerRef.current);
    if (id) setHoveredEdge(id);
    else hoverTimerRef.current = window.setTimeout(() => setHoveredEdge(null), EDGE_HOVER_GRACE_MS);
  }, []);
  useEffect(() => () => window.clearTimeout(hoverTimerRef.current), []);

  // The buttons render inside React Flow's edge-label portal and cannot reach
  // this state directly, so they come back as events — the same contract the
  // group frame's title already uses.
  useEffect(() => {
    const onInsert = (event: Event) => {
      const { id } = (event as CustomEvent).detail as { id: string };
      setEdgeHover(null);
      setPaletteIntent({ kind: "edge", edgeId: id });
    };
    const onDelete = (event: Event) => {
      const { id } = (event as CustomEvent).detail as { id: string };
      commit();
      setEdges((current) => current.filter((edge) => edge.id !== id));
      setHoveredEdge(null);
    };
    const onHover = (event: Event) => {
      const { id } = (event as CustomEvent).detail as { id: string | null };
      setEdgeHover(id);
    };
    // A node's own stub +, which reaches this state the same way for the same
    // reason: it renders inside React Flow's node tree.
    const onExtend = (event: Event) => {
      const { id, side } = (event as CustomEvent).detail as { id: string; side: "in" | "out" };
      setPaletteIntent({ kind: "extend", nodeId: id, side });
    };
    window.addEventListener(EDGE_INSERT_EVENT, onInsert);
    window.addEventListener(EDGE_DELETE_EVENT, onDelete);
    window.addEventListener(EDGE_HOVER_EVENT, onHover);
    window.addEventListener(NODE_EXTEND_EVENT, onExtend);
    return () => {
      window.removeEventListener(EDGE_INSERT_EVENT, onInsert);
      window.removeEventListener(EDGE_DELETE_EVENT, onDelete);
      window.removeEventListener(EDGE_HOVER_EVENT, onHover);
      window.removeEventListener(NODE_EXTEND_EVENT, onExtend);
    };
  }, [commit, setEdgeHover]);

  // Hold Space to pan. Released on blur as well as keyup, so alt-tabbing away
  // mid-pan does not leave the canvas stuck in the pan tool.
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (event.code !== "Space" || event.repeat || isSpaceConsumer(event.target)) return;
      event.preventDefault();
      setSpacePan(true);
    };
    const up = (event: KeyboardEvent) => {
      if (event.code === "Space") setSpacePan(false);
    };
    const release = () => setSpacePan(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", release);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", release);
    };
  }, []);

  const selected = decoratedNodes.find((node) => node.id === selectedId) ?? null;
  const selectedGroup =
    groups.find((group) => group.id === selectedGroupIds.at(-1)) ?? null;

  // The dialog edits the selection, so deleting or deselecting that closes it
  // rather than leaving an empty panel over the canvas.
  useEffect(() => {
    if (settingsOpen && !selected && !selectedGroup) setSettingsOpen(false);
  }, [settingsOpen, selected, selectedGroup]);

  const updateSelected = useCallback(
    (mutate: (node: ArchFlowNode) => ArchFlowNode) => {
      setNodes((current) => current.map((node) => (node.id === selectedId ? mutate(node) : node)));
      markDirty();
    },
    [selectedId, markDirty]
  );

  const tidy = useCallback(() => {
    commit();
    setNodes((current) => autoLayout(current, edges));
  }, [edges, commit]);

  // --- selection, clipboard, and deletion ----------------------------------

  const selectAll = useCallback(() => {
    setNodes((current) => current.map((node) => ({ ...node, selected: true })));
    setSelectedGroupIds([]);
  }, []);

  const deselectAll = useCallback(() => {
    setNodes((current) =>
      current.some((node) => node.selected)
        ? current.map((node) => ({ ...node, selected: false }))
        : current
    );
    setEdges((current) =>
      current.some((edge) => edge.selected)
        ? current.map((edge) => ({ ...edge, selected: false }))
        : current
    );
    setSelectedId("");
    setSelectedGroupIds([]);
  }, []);

  /**
   * The in-app clipboard.
   *
   * Held here as well as in the system clipboard because the app's own
   * right-click menu has no `ClipboardEvent` to read from, and writing to the
   * system clipboard programmatically needs a permission the browser may
   * refuse. The system clipboard is written best-effort on top, so a copy in
   * one tab can be pasted into another.
   */
  const clipboardRef = useRef<string | null>(null);
  /** Last pointer position over the canvas, in flow coordinates. */
  const pointerRef = useRef<{ x: number; y: number } | null>(null);

  const copySelection = useCallback((): string | null => {
    const payload = selectionPayload(nodes, edges, groups, new Set(selectedGroupIds));
    if (!payload) return null;
    const serialized = serializePayload(payload);
    clipboardRef.current = serialized;
    void navigator.clipboard?.writeText(serialized).catch(() => {
      // No clipboard permission: the in-app copy above still works.
    });
    return serialized;
  }, [nodes, edges, groups, selectedGroupIds]);

  const deleteSelection = useCallback(() => {
    const doomed = new Set(nodes.filter((node) => node.selected).map((node) => node.id));
    if (selectedId) doomed.add(selectedId);
    const doomedGroups = new Set(selectedGroupIds);
    // A selected wire is deleted here too: `deleteKeyCode` is off, so React
    // Flow never removes one on its own and a lone edge selection would
    // otherwise be undeletable by any means but the edge context menu.
    const doomedEdges = new Set(
      edges.filter((edge) => edge.selected).map((edge) => edge.id)
    );
    if (doomed.size === 0 && doomedGroups.size === 0 && doomedEdges.size === 0) return;
    commit();
    setNodes((current) => current.filter((node) => !doomed.has(node.id)));
    setEdges((current) =>
      current.filter(
        (edge) =>
          !doomedEdges.has(edge.id) &&
          !doomed.has(edge.source) &&
          !doomed.has(edge.target)
      )
    );
    setGroups((current) => current.filter((group) => !doomedGroups.has(group.id)));
    setSelectedId("");
    setSelectedGroupIds([]);
  }, [nodes, edges, selectedId, selectedGroupIds, commit]);

  const cutSelection = useCallback(() => {
    if (!copySelection()) return;
    deleteSelection();
  }, [copySelection, deleteSelection]);

  /**
   * Drop every wire attached to the selected nodes, keeping the nodes.
   *
   * The alternative is picking each connector off one at a time, and a node in
   * the middle of a chain has at least two.
   */
  const disconnectSelection = useCallback(() => {
    const chosen = new Set(nodes.filter((node) => node.selected).map((node) => node.id));
    if (selectedId) chosen.add(selectedId);
    if (chosen.size === 0) return;
    if (!edges.some((edge) => chosen.has(edge.source) || chosen.has(edge.target))) return;
    commit();
    setEdges((current) =>
      current.filter((edge) => !chosen.has(edge.source) && !chosen.has(edge.target))
    );
  }, [nodes, edges, selectedId, commit]);

  /**
   * Paste `raw` — or whatever was last copied in-app — at `at`, or offset from
   * the original when no position is given.
   */
  const paste = useCallback(
    (raw: string | null, at: { x: number; y: number } | null) => {
      const payload = parsePayload(raw ?? clipboardRef.current ?? "");
      if (!payload || (payload.nodes.length === 0 && payload.groups.length === 0)) return;
      const added = instantiate(
        payload,
        nodes.map((node) => node.id),
        groups.map((group) => group.id),
        at
      );
      commit();
      setNodes((current) => [
        ...current.map((node) => ({ ...node, selected: false })),
        ...added.nodes
      ]);
      setEdges((current) => [...current, ...added.edges]);
      setGroups((current) => [...current, ...added.groups]);
      setSelectedId(added.nodes.at(-1)?.id ?? "");
      setSelectedGroupIds(added.groups.map((group) => group.id));
    },
    [nodes, groups, commit]
  );

  const duplicateSelection = useCallback(() => {
    const payload = selectionPayload(nodes, edges, groups, new Set(selectedGroupIds));
    if (!payload) return;
    paste(serializePayload(payload), null);
  }, [nodes, edges, groups, selectedGroupIds, paste]);

  // Cut / copy / paste ride the real clipboard events rather than a keydown
  // handler, so the browser's own right-click menu drives the same code path
  // that Ctrl+C does, and a paste carries whatever is actually on the clipboard.
  useEffect(() => {
    const onCopy = (event: ClipboardEvent) => {
      if (isTextEntry(event.target)) return;
      const serialized = copySelection();
      if (!serialized) return;
      event.preventDefault();
      event.clipboardData?.setData("text/plain", serialized);
    };
    const onCut = (event: ClipboardEvent) => {
      if (isTextEntry(event.target)) return;
      const serialized = copySelection();
      if (!serialized) return;
      event.preventDefault();
      event.clipboardData?.setData("text/plain", serialized);
      deleteSelection();
    };
    const onPaste = (event: ClipboardEvent) => {
      if (isTextEntry(event.target)) return;
      const raw = event.clipboardData?.getData("text/plain") ?? null;
      if (!parsePayload(raw ?? "") && !clipboardRef.current) return;
      event.preventDefault();
      paste(raw, pointerRef.current);
    };
    window.addEventListener("copy", onCopy);
    window.addEventListener("cut", onCut);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("copy", onCopy);
      window.removeEventListener("cut", onCut);
      window.removeEventListener("paste", onPaste);
    };
  }, [copySelection, deleteSelection, paste]);

  // Editor shortcuts, matching what a user already has in their fingers.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey;
      const key = event.key.toLowerCase();
      if (meta && key === "s") {
        event.preventDefault();
        if (!saveMutation.isPending) saveMutation.mutate();
        return;
      }
      if (meta && key === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
        return;
      }
      if (meta && key === "y") {
        event.preventDefault();
        redo();
        return;
      }
      if (meta && key === "a" && !isTextEntry(event.target)) {
        event.preventDefault();
        selectAll();
        return;
      }
      if (meta && key === "d") {
        event.preventDefault();
        duplicateSelection();
        return;
      }
      if (event.key === "Escape") {
        // One layer per press, outermost first: the dialog, then the layer
        // panel, then the menu, then fullscreen.
        if (menu) setMenu(null);
        else if (settingsOpen) setSettingsOpen(false);
        else if (paletteIntent) setPaletteIntent(null);
        else if (fullscreen) setFullscreen(false);
        return;
      }
      if (meta || event.altKey) return;
      // Single-key shortcuts, but never while typing into a field — renaming a
      // node to "Gate" should not tidy the layout on the way through.
      if (isTextEntry(event.target)) return;
      if (key === "delete" || key === "backspace") {
        // Group frames are not React Flow's to delete, so the whole selection
        // is handled here rather than by `deleteKeyCode`.
        event.preventDefault();
        deleteSelection();
      } else if (key === "n") setPaletteIntent((current) => (current ? null : { kind: "canvas" }));
      else if (
        key === "enter" &&
        // Enter on a focused control activates that control; it only means
        // "open settings" when the keystroke belongs to nobody else.
        !isSpaceConsumer(event.target) &&
        (selectedId || selectedGroupIds.length > 0)
      ) {
        setSettingsOpen(true);
      } else if (key === "g") addGroup();
      else if (key === "t") tidy();
      else if (key === "f") fitView({ duration: 200, padding: 0.15 });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    saveMutation,
    undo,
    redo,
    selectAll,
    duplicateSelection,
    deleteSelection,
    fullscreen,
    menu,
    settingsOpen,
    paletteIntent,
    selectedId,
    selectedGroupIds,
    addGroup,
    tidy,
    fitView
  ]);

  // --- context menu --------------------------------------------------------

  const selectionCount = nodes.filter((node) => node.selected).length + selectedGroupIds.length;
  // Edges are deletable but not copyable — a wire without its endpoints is not
  // a thing that can be pasted — so they gate Delete only, not the clipboard.
  const selectedEdgeCount = edges.filter((edge) => edge.selected).length;
  const canPaste = clipboardRef.current !== null;

  const openMenu = useCallback(
    (event: React.MouseEvent, extra: ContextMenuItem[] = []) => {
      event.preventDefault();
      pointerRef.current = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const at = pointerRef.current;
      const hasSelection = selectionCount > 0;
      setMenu({
        x: event.clientX,
        y: event.clientY,
        items: [
          // Right-click adds a node where you right-clicked. That is the reason
          // most people open this menu, so it is the first row rather than the
          // one under six clipboard commands.
          {
            label: "Add node",
            shortcut: "N",
            icon: <Plus size={14} />,
            onSelect: () => setPaletteIntent({ kind: "at", position: at })
          },
          { kind: "separator" },
          ...extra,
          {
            label: "Cut",
            shortcut: accelerator("X"),
            icon: <Scissors size={14} />,
            disabled: !hasSelection,
            onSelect: cutSelection
          },
          {
            label: "Copy",
            shortcut: accelerator("C"),
            icon: <Clipboard size={14} />,
            disabled: !hasSelection,
            onSelect: () => copySelection()
          },
          {
            label: "Paste",
            shortcut: accelerator("V"),
            icon: <ClipboardPaste size={14} />,
            disabled: !canPaste,
            onSelect: () => paste(null, at)
          },
          {
            label: "Duplicate",
            shortcut: accelerator("D"),
            icon: <Copy size={14} />,
            disabled: !hasSelection,
            onSelect: duplicateSelection
          },
          {
            label: "Disconnect",
            icon: <Unlink size={14} />,
            disabled: !hasSelection,
            onSelect: disconnectSelection
          },
          { kind: "separator" },
          {
            label: "Select all",
            shortcut: accelerator("A"),
            icon: <SquareDashedMousePointer size={14} />,
            onSelect: selectAll
          },
          {
            label: "Deselect all",
            icon: <SquareDashedMousePointer size={14} />,
            disabled: !hasSelection && selectedEdgeCount === 0,
            onSelect: deselectAll
          },
          {
            label: hasSelection ? "Frame as group" : "Add group frame",
            shortcut: "G",
            icon: <Group size={14} />,
            onSelect: addGroup
          },
          { kind: "separator" },
          {
            label: "Tidy layout",
            shortcut: "T",
            icon: <Wand2 size={14} />,
            disabled: nodes.length === 0,
            onSelect: tidy
          },
          {
            label: "Fit to view",
            shortcut: "F",
            icon: <Maximize size={14} />,
            disabled: nodes.length === 0,
            onSelect: () => fitView({ duration: 200, padding: 0.15 })
          },
          { kind: "separator" },
          {
            label: "Undo",
            shortcut: accelerator("Z"),
            icon: <Undo2 size={14} />,
            disabled: !canUndo,
            onSelect: undo
          },
          {
            label: "Redo",
            shortcut: accelerator("⇧Z"),
            icon: <Redo2 size={14} />,
            disabled: !canRedo,
            onSelect: redo
          },
          { kind: "separator" },
          {
            label: "Delete",
            shortcut: "Del",
            icon: <Trash2 size={14} />,
            danger: true,
            disabled: !hasSelection && selectedEdgeCount === 0,
            onSelect: deleteSelection
          }
        ]
      });
    },
    [
      screenToFlowPosition,
      selectionCount,
      selectedEdgeCount,
      canPaste,
      cutSelection,
      copySelection,
      paste,
      duplicateSelection,
      disconnectSelection,
      selectAll,
      deselectAll,
      addGroup,
      nodes.length,
      tidy,
      fitView,
      canUndo,
      undo,
      canRedo,
      redo,
      deleteSelection
    ]
  );

  if (architectureQuery.isLoading || catalogQuery.isLoading) {
    return <PageSkeleton title="Architecture" />;
  }
  if (architectureQuery.isError) {
    return (
      <div className="panel arch-load-error">
        <p>That architecture could not be loaded.</p>
        <ButtonLink variant="secondary" href="/models/architectures">
          Back to architectures
        </ButtonLink>
      </div>
    );
  }

  const framedCount = nodes.filter((node) => node.selected).length;
  const graphIssues = nodeIssues.get("") ?? [];
  const errorCount = validation.issues.filter((issue) => issue.severity === "error").length;
  const warningCount = validation.issues.length - errorCount;

  return (
    <div className={`arch-studio${fullscreen ? " arch-studio-fullscreen" : ""}`}>
      <header className="arch-studio-head">
        <div className="arch-studio-identity">
          <Link className="arch-back" href="/models/architectures">
            <ArrowLeft size={15} aria-hidden /> Architectures
          </Link>
          <input
            className="arch-name-input"
            value={name}
            aria-label="Architecture name"
            onChange={(event) => {
              setName(event.target.value);
              markDirty();
            }}
          />
        </div>

        <div className="arch-studio-stats">
          <span>
            <strong>{validation.layers}</strong> layers
          </span>
          <span>
            <strong>{formatCount(validation.params)}</strong> params
          </span>
          <label className="arch-classes">
            Preview classes
            <NumberInput
              value={numClasses}
              min={1}
              max={2000000}
              step={1}
              onChange={(value) => {
                setNumClasses(Math.max(1, Math.round(value)));
                markDirty();
              }}
            />
          </label>
        </div>

        <div className="arch-studio-actions">
          <div className="arch-view-controls" role="group" aria-label="History">
            <IconButton
              aria-label="Undo"
              title={`Undo (${accelerator("Z")})`}
              disabled={!canUndo}
              onClick={undo}
            >
              <Undo2 size={18} />
            </IconButton>
            <IconButton
              aria-label="Redo"
              title={`Redo (${accelerator("⇧Z")})`}
              disabled={!canRedo}
              onClick={redo}
            >
              <Redo2 size={18} />
            </IconButton>
          </div>
          <div className="arch-view-controls" role="group" aria-label="Canvas layout">
            <IconButton
              aria-label="Open settings for the selection"
              title="Settings for the selected node or frame (↵) — or double-click it"
              disabled={!selected && !selectedGroup}
              onClick={() => setSettingsOpen(true)}
            >
              <SlidersHorizontal size={18} />
            </IconButton>
            <IconButton
              aria-label={fullscreen ? "Exit fullscreen" : "Expand the canvas to fullscreen"}
              title={
                fullscreen ? "Exit fullscreen (Esc)" : "Expand the canvas to fullscreen"
              }
              onClick={() => setFullscreen((value) => !value)}
            >
              {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
            </IconButton>
          </div>
          <ButtonLink
            variant="ghost"
            size="sm"
            href={api.architectureExportUrl(architectureId)}
          >
            <Download size={15} aria-hidden /> Export
          </ButtonLink>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setTab(tab === "code" ? "issues" : "code")}
          >
            <Code2 size={15} aria-hidden /> {tab === "code" ? "Hide code" : "View code"}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={!validation.ok || dirty}
            title={
              dirty
                ? "Save before training"
                : validation.ok
                  ? "Train this architecture"
                  : "Fix the errors first"
            }
            onClick={() =>
              router.push(
                // The task travels with the link so the training form can pick
                // the runner that matches this graph. Without it every graph
                // arrived as an image classifier.
                `/training?architecture_id=${encodeURIComponent(architectureId)}` +
                  `&task_type=${encodeURIComponent(architectureQuery.data?.task_type ?? "classification")}` +
                  `&project_id=${encodeURIComponent(projectId)}`
              )
            }
          >
            <Play size={15} aria-hidden /> Train
          </Button>
          <Button
            size="sm"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending || !dirty}
          >
            {dirty ? <Save size={15} aria-hidden /> : <Check size={15} aria-hidden />}
            {saveMutation.isPending ? "Saving…" : dirty ? "Save" : "Saved"}
          </Button>
        </div>
      </header>

      <MutationError mutations={[saveMutation]} />

      <div className="arch-studio-grid">
        <div
          ref={canvasRef}
          className={`arch-canvas${spacePan ? " arch-canvas-pan" : ""}`}
          onPointerMove={(event) => {
            pointerRef.current = screenToFlowPosition({ x: event.clientX, y: event.clientY });
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
          }}
          onDrop={(event) => {
            event.preventDefault();
            const type = event.dataTransfer.getData("application/x-arch-node");
            const spec = specs.get(type);
            if (!spec) return;
            addNode(
              spec,
              screenToFlowPosition({ x: event.clientX, y: event.clientY })
            );
          }}
        >
          <ReactFlow
            nodes={flowNodes as never}
            edges={routedEdges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDragStart={() => commit()}
            onEdgeMouseEnter={(_, edge) => setEdgeHover(edge.id)}
            onEdgeMouseLeave={() => setEdgeHover(null)}
            // Panning and zooming are exactly when a minimap earns its corner.
            onMove={wakeMinimap}
            onPaneClick={() => {
              setSelectedId("");
              setSelectedGroupIds([]);
              // Clicking the canvas is how you say "not that" to anything the
              // studio is showing, the layer panel included — reaching for its
              // close button to dismiss something you already dismissed is a
              // second gesture for one decision.
              setPaletteIntent(null);
            }}
            onNodeClick={() => setPaletteIntent(null)}
            // Double-click opens the thing you double-clicked. That gesture is
            // free only because pane double-click no longer zooms.
            onNodeDoubleClick={(_, node) => {
              const { id, type } = node as { id: string; type?: string };
              if (type === "group") {
                setSelectedGroupIds([id]);
                setSelectedId("");
              } else {
                setNodes((current) =>
                  current.map((item) => ({ ...item, selected: item.id === id }))
                );
                setSelectedId(id);
                setSelectedGroupIds([]);
              }
              setSettingsOpen(true);
            }}
            onPaneContextMenu={(event) => openMenu(event as unknown as React.MouseEvent)}
            onNodeContextMenu={(event, node) => {
              // Group frames ride in the same node array (hence the `as never`
              // on `nodes`), so the handler sees the union at runtime even
              // though the generic narrows it to a model node.
              const { id, type, selected: isSelected } = node as {
                id: string;
                type?: string;
                selected?: boolean;
              };
              // Right-clicking something outside the selection acts on that
              // thing, which is what every other editor does.
              if (type === "group") {
                if (!selectedGroupIds.includes(id)) {
                  setSelectedGroupIds([id]);
                  setSelectedId("");
                }
              } else if (!isSelected) {
                setNodes((current) =>
                  current.map((item) => ({ ...item, selected: item.id === id }))
                );
                setSelectedId(id);
                setSelectedGroupIds([]);
              }
              openMenu(event, [
                {
                  label: "Settings…",
                  shortcut: "↵",
                  icon: <SlidersHorizontal size={14} />,
                  onSelect: () => setSettingsOpen(true)
                },
                { kind: "separator" }
              ]);
            }}
            onEdgeContextMenu={(event, edge) => {
              // Right-clicking a wire acts on that wire, so it becomes the
              // selection first — the menu then matches what is highlighted.
              setEdges((current) =>
                current.map((item) => ({ ...item, selected: item.id === edge.id }))
              );
              setNodes((current) =>
                current.some((node) => node.selected)
                  ? current.map((node) => ({ ...node, selected: false }))
                  : current
              );
              setSelectedId("");
              setSelectedGroupIds([]);
              openMenu(event, [
                {
                  label: "Insert node here",
                  icon: <Plus size={14} />,
                  onSelect: () => setPaletteIntent({ kind: "edge", edgeId: edge.id })
                },
                {
                  label: "Delete connection",
                  icon: <Trash2 size={14} />,
                  danger: true,
                  onSelect: () => {
                    commit();
                    setEdges((current) => current.filter((item) => item.id !== edge.id));
                  }
                },
                { kind: "separator" }
              ]);
            }}
            fitView
            proOptions={{ hideAttribution: false }}
            // Delete is handled by the studio's own key handler so it can take
            // group frames — which React Flow does not own — with the selection.
            deleteKeyCode={null}
            // Drag on empty canvas draws a selection box — always, with no tool
            // to switch first. Panning is the trackpad's two fingers, the middle
            // mouse button, or Space; none of them cost a mode.
            selectionOnDrag={!spacePan}
            panOnDrag={spacePan ? true : [1]}
            selectionMode={SelectionMode.Partial}
            multiSelectionKeyCode={["Meta", "Shift", "Control"]}
            // Trackpad semantics, matching every other graph canvas: two-finger
            // scroll pans in both axes, pinch zooms. Wheel-zoom is off, so a
            // sideways flick no longer changes scale mid-gesture; ⌘/Ctrl+scroll
            // is still there for a mouse.
            panOnScroll
            panOnScrollMode={PanOnScrollMode.Free}
            zoomOnScroll={false}
            zoomOnPinch
            // Double-click belongs to "open this node's settings" now.
            zoomOnDoubleClick={false}
            // No `defaultEdgeOptions`: `routedEdges` is the only thing that
            // ever reaches this prop, and it sets the type and hit width on
            // every edge it emits.
            connectionLineType={ConnectionLineType.SmoothStep}
          >
            {/* Actions live on the canvas, not in the page header: they act on
                what is under the pointer, and a round trip to the top of the
                screen is the wrong cost for the most repeated motions in a node
                editor. The select/move pair that used to sit here is gone — the
                pointer always selects, so there was no mode left to show. */}
            <Panel
              position="top-left"
              className="arch-canvas-tools"
              // Both floating clusters step aside for the layer panel rather
              // than sitting under it — the + that opens the panel is also the
              // one that closes it.
              style={paletteIntent ? { marginLeft: palettePane.size + 10 } : undefined}
            >
              {/* Canvas controls carry `data-tip` rather than `title`: a native
                  tooltip waits about a second and then paints an OS chrome box
                  over the graph, which is the wrong latency and the wrong
                  surface for a control you hit dozens of times a session. */}
              <div className="arch-tool-group" role="group" aria-label="Add">
                <button
                  type="button"
                  aria-label="Add a node"
                  data-tip="Add node (N)"
                  aria-pressed={paletteIntent !== null}
                  className={`arch-tool arch-tip${paletteIntent ? " arch-tool-active" : ""}`}
                  onClick={() =>
                    setPaletteIntent((current) => (current ? null : { kind: "canvas" }))
                  }
                >
                  <Plus size={17} aria-hidden />
                </button>
              </div>
              <div className="arch-tool-group" role="group" aria-label="Canvas actions">
                <button
                  type="button"
                  className="arch-tool arch-tip"
                  aria-label="Frame the selection as a group"
                  data-tip={
                    framedCount > 0
                      ? `Frame ${framedCount} node${framedCount === 1 ? "" : "s"} (G)`
                      : "Add group frame (G)"
                  }
                  onClick={addGroup}
                >
                  <Group size={17} aria-hidden />
                </button>
                <button
                  type="button"
                  className="arch-tool arch-tip"
                  aria-label="Tidy the layout"
                  data-tip="Tidy layout (T)"
                  onClick={tidy}
                >
                  <Wand2 size={17} aria-hidden />
                </button>
              </div>
            </Panel>
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            {/* React Flow's stock zoom buttons are a bare + and −, which on a
                canvas whose other + adds a node is the wrong glyph entirely.
                These say magnifier. */}
            <Controls
              showZoom={false}
              showFitView={false}
              showInteractive={false}
              position="bottom-left"
              style={paletteIntent ? { marginLeft: palettePane.size + 15 } : undefined}
            >
              <ControlButton
                className="arch-tip arch-tip-right"
                aria-label="Zoom in"
                data-tip="Zoom in"
                onClick={() => zoomIn({ duration: 120 })}
              >
                {/* The lens-with-a-sign glyph, sized so the sign inside it is
                    actually legible: base.css's 12px cap rendered it a couple
                    of pixels wide and both buttons read as a plain circle. */}
                <ZoomIn size={22} strokeWidth={2.4} aria-hidden />
              </ControlButton>
              <ControlButton
                className="arch-tip arch-tip-right"
                aria-label="Zoom out"
                data-tip="Zoom out"
                onClick={() => zoomOut({ duration: 120 })}
              >
                <ZoomOut size={22} strokeWidth={2.4} aria-hidden />
              </ControlButton>
              {/* Fit lives here, with the other view controls, and no longer
                  also in the top-left toolbar — one action, one button. */}
              <ControlButton
                className="arch-tip arch-tip-right"
                aria-label="Fit the graph to the view"
                data-tip="Fit to view (F)"
                onClick={() => fitView({ duration: 200, padding: 0.15 })}
              >
                <Maximize size={20} strokeWidth={2.4} aria-hidden />
              </ControlButton>
            </Controls>
            <MiniMap
              pannable
              zoomable
              nodeColor={minimapColor}
              className={minimapAwake ? "arch-minimap-awake" : undefined}
              // Hovering it restarts the countdown, so it cannot fade out from
              // under a pointer that is using it to navigate.
              onPointerMove={wakeMinimap}
            />
          </ReactFlow>

          {/* The layer panel overlays the canvas rather than holding a column
              of it. It is closed by default, because for most of a session the
              answer to "what do I need on screen" is the graph. */}
          {paletteIntent && (
            <div
              className="arch-palette-drawer"
              style={{ width: `${palettePane.size}px` }}
              role="dialog"
              aria-label={PALETTE_TITLES[paletteIntent.kind]}
            >
              <div className="arch-palette-drawer-head">
                <p className="arch-palette-drawer-title">
                  {PALETTE_TITLES[paletteIntent.kind]}
                </p>
                <IconButton
                  aria-label="Close the layer panel"
                  title="Close the layer panel (Esc)"
                  onClick={() => setPaletteIntent(null)}
                >
                  <X size={18} />
                </IconButton>
              </div>
              <NodePalette
                specs={catalogQuery.data ?? []}
                categories={categoriesQuery.data ?? []}
                onAdd={handlePaletteAdd}
              />
              <div
                className={`arch-resize arch-resize-x arch-palette-drawer-resize${
                  palettePane.dragging ? " arch-resize-active" : ""
                }`}
                aria-label="Resize the layer panel"
                {...palettePane.handleProps}
              />
            </div>
          )}
        </div>
      </div>

      {showDrawer && (
        <div
          className={`arch-resize arch-resize-y${drawerPane.dragging ? " arch-resize-active" : ""}`}
          aria-label="Resize the issues and code panel"
          {...drawerPane.handleProps}
        />
      )}
      <section
        className={`arch-drawer${showDrawer ? "" : " arch-drawer-collapsed"}`}
        style={showDrawer ? { height: `${drawerPane.size}px`, maxHeight: "none" } : undefined}
      >
        <div className="arch-drawer-head">
          <div className="arch-drawer-tabs segmented-control">
            <button
              type="button"
              className={tab === "issues" ? "segmented-active" : ""}
              onClick={() => {
                setTab("issues");
                setShowDrawer(true);
              }}
            >
              Issues
              {errorCount + warningCount > 0 && (
                <span className="segmented-count">{errorCount + warningCount}</span>
              )}
            </button>
            <button
              type="button"
              className={tab === "code" ? "segmented-active" : ""}
              onClick={() => {
                setTab("code");
                setShowDrawer(true);
              }}
            >
              Generated code
            </button>
          </div>
          <IconButton
            aria-label={showDrawer ? "Collapse this panel" : "Expand this panel"}
            title={showDrawer ? "Collapse this panel" : "Expand this panel"}
            onClick={() => setShowDrawer((value) => !value)}
          >
            <ChevronDown
              size={18}
              className={showDrawer ? "arch-drawer-chevron" : "arch-drawer-chevron arch-drawer-chevron-up"}
            />
          </IconButton>
        </div>

        {!showDrawer ? null : tab === "issues" ? (
          <div className="arch-drawer-body">
            {validation.issues.length === 0 ? (
              <p className="arch-drawer-clean">
                <Check size={15} aria-hidden /> No issues. This graph compiles.
              </p>
            ) : (
              <ul className="arch-issue-list">
                {[...graphIssues, ...validation.issues.filter((issue) => issue.node_id)].map(
                  (issue, index) => (
                    <li key={index} className={`arch-issue arch-issue-${issue.severity}`}>
                      {issue.severity === "error" && <AlertTriangle size={14} aria-hidden />}
                      {issue.node_id && (
                        <button
                          type="button"
                          className="arch-issue-node"
                          onClick={() => {
                            setSelectedId(issue.node_id ?? "");
                            setSelectedGroupIds([]);
                          }}
                        >
                          {issue.node_id}
                        </button>
                      )}
                      <span>{issue.message}</span>
                    </li>
                  )
                )}
              </ul>
            )}
          </div>
        ) : (
          <CodePanel
            code={codeQuery.data?.code ?? ""}
            filename={codeQuery.data?.filename ?? "generated_model.py"}
            downloadUrl={api.architectureCodeDownloadUrl(architectureId, numClasses, framework)}
            framework={framework}
            onFrameworkChange={setFramework}
            loading={codeQuery.isFetching}
            error={
              codeQuery.isError
                ? codeQuery.error instanceof Error
                  ? codeQuery.error.message
                  : "Code generation failed"
                : dirty
                  ? "Save to regenerate code from your latest edits."
                  : null
            }
          />
        )}
      </section>

      {settingsOpen && selectedGroup && (
        <SettingsModal
          title={selectedGroup.title || "Untitled group"}
          subtitle="Group frame"
          onClose={() => setSettingsOpen(false)}
        >
          <GroupInspector
            group={selectedGroup}
            onChange={(patch) => patchGroup(selectedGroup.id, patch)}
            onDelete={() => {
              commit();
              setGroups((current) => current.filter((group) => group.id !== selectedGroup.id));
              setSelectedGroupIds([]);
              setSettingsOpen(false);
            }}
          />
        </SettingsModal>
      )}
      {settingsOpen && !selectedGroup && selected && (
        <SettingsModal
          title={selected.data.label || selected.data.spec.name}
          subtitle={`${selected.data.spec.name} · ${selected.id}`}
          onClose={() => setSettingsOpen(false)}
        >
          <NodeInspector
            node={selected}
            issues={nodeIssues.get(selected.id) ?? []}
            onRename={(label) =>
              updateSelected((node) => ({ ...node, data: { ...node.data, label } }))
            }
            onParamChange={(key, value) =>
              updateSelected((node) => ({
                ...node,
                data: { ...node.data, params: { ...node.data.params, [key]: value } }
              }))
            }
            onDuplicate={duplicateSelection}
            onDelete={() => {
              deleteSelection();
              setSettingsOpen(false);
            }}
          />
        </SettingsModal>
      )}

      <CanvasContextMenu state={menu} onClose={() => setMenu(null)} />
    </div>
  );
}
