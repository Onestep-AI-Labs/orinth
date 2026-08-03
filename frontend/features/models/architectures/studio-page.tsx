"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Background,
  BackgroundVariant,
  ConnectionLineType,
  Controls,
  MiniMap,
  Panel,
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
  Frame,
  Group,
  Hand,
  Maximize2,
  Minimize2,
  MousePointer2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Redo2,
  Save,
  Scissors,
  SquareDashedMousePointer,
  Trash2,
  Undo2,
  Wand2
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
import { autoLayout } from "./auto-layout";
import { useResizable } from "./use-resizable";
import { useHistory, type GraphSnapshot } from "./use-history";
import { instantiate, parsePayload, selectionPayload, serializePayload } from "./clipboard";
import { applyGroupChanges, isGroupEdit, resizePhase } from "./group-changes";
import { CanvasContextMenu, type ContextMenuItem, type ContextMenuState } from "./context-menu";
import { CanvasNode } from "./canvas-node";
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
const VALIDATE_DEBOUNCE_MS = 350;
/** Breathing room between a framed selection's nodes and the frame's edge. */
const GROUP_PADDING = 28;
const NODE_WIDTH = 176;
const NODE_HEIGHT = 78;
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
  const { screenToFlowPosition, fitView } = useReactFlow();

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
  // Canvas real estate is the scarce resource in a node editor, so every
  // surrounding panel can get out of the way independently.
  const [fullscreen, setFullscreen] = useState(false);
  const [showPalette, setShowPalette] = useState(true);
  const [showInspector, setShowInspector] = useState(true);
  const [showDrawer, setShowDrawer] = useState(true);
  // Select drags a marquee; Move pans the canvas. A node editor that only
  // does one of them forces a modifier key for the other.
  const [tool, setTool] = useState<"select" | "move">("select");
  const [groups, setGroups] = useState<StoredGroup[]>([]);
  const [framework, setFramework] = useState<Framework>("keras");
  const [menu, setMenu] = useState<ContextMenuState | null>(null);
  // Every pane edge is draggable and remembers where it was left. The canvas
  // is the work surface, so the rails start narrow and the drawer short.
  const palettePane = useResizable("palette", 208, { min: 160, max: 420 });
  const inspectorPane = useResizable("inspector", 288, { min: 240, max: 560, invert: true });
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
   */
  const routedEdges = useMemo(() => {
    const outDegree = new Map<string, number>();
    const inDegree = new Map<string, number>();
    for (const edge of edges) {
      outDegree.set(edge.source, (outDegree.get(edge.source) ?? 0) + 1);
      inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
    }
    return edges.map((edge) => {
      const branching =
        (outDegree.get(edge.source) ?? 0) > 1 || (inDegree.get(edge.target) ?? 0) > 1;
      return {
        ...edge,
        type: branching ? "default" : "smoothstep",
        className: branching ? "arch-edge-branch" : undefined,
        // A 1.5px wire is a 1.5px click target without this.
        interactionWidth: EDGE_INTERACTION_WIDTH,
        animated: false
      };
    });
  }, [edges]);

  const decoratedNodes = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          shape: validation.shapes[node.id] ?? null,
          issues: nodeIssues.get(node.id) ?? []
        }
      })),
    [nodes, validation.shapes, nodeIssues]
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
      setEdges((current) =>
        addEdge(
          { ...connection, id: edgeId(connection.source, connection.target), type: "smoothstep" },
          current
        )
      );
    },
    [commit]
  );

  const addNode = useCallback(
    (spec: NodeSpec, position?: { x: number; y: number }) => {
      // The id and position are derived before the updater runs: a state
      // updater must stay pure, and React invokes it twice under StrictMode.
      const id = nextNodeId(spec.type, nodes.map((node) => node.id));
      const spot =
        position ??
        // Stagger click-added nodes so they do not stack on one point.
        { x: 320 + (nodes.length % 5) * 40, y: 120 + (nodes.length % 7) * 60 };
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
    [nodes, commit]
  );

  const selected = decoratedNodes.find((node) => node.id === selectedId) ?? null;
  const selectedGroup =
    groups.find((group) => group.id === selectedGroupIds.at(-1)) ?? null;

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
        setMenu(null);
        if (fullscreen) setFullscreen(false);
      }
      if (meta || event.altKey) return;
      // Single-key tool shortcuts, but never while typing into a field —
      // renaming a node to "Vision" should not switch tools five times.
      if (isTextEntry(event.target)) return;
      if (key === "delete" || key === "backspace") {
        // Group frames are not React Flow's to delete, so the whole selection
        // is handled here rather than by `deleteKeyCode`.
        event.preventDefault();
        deleteSelection();
      } else if (key === "v") setTool("select");
      else if (key === "h") setTool("move");
      else if (key === "g") addGroup();
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
          { kind: "separator" },
          {
            label: "Select all",
            shortcut: accelerator("A"),
            icon: <SquareDashedMousePointer size={14} />,
            onSelect: selectAll
          },
          {
            label: hasSelection ? "Frame as group" : "Add group frame",
            shortcut: "G",
            icon: <Group size={14} />,
            onSelect: addGroup
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
      selectAll,
      addGroup,
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
              aria-label={showPalette ? "Hide the layer palette" : "Show the layer palette"}
              onClick={() => setShowPalette((value) => !value)}
            >
              {showPalette ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
            </IconButton>
            <IconButton
              aria-label={showInspector ? "Hide node settings" : "Show node settings"}
              onClick={() => setShowInspector((value) => !value)}
            >
              {showInspector ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </IconButton>
            <IconButton
              aria-label={fullscreen ? "Exit fullscreen" : "Expand the canvas to fullscreen"}
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

      <div
        className={[
          "arch-studio-grid",
          showPalette ? "" : "arch-no-palette",
          showInspector ? "" : "arch-no-inspector"
        ]
          .filter(Boolean)
          .join(" ")}
        style={{
          gridTemplateColumns: [
            showPalette ? `${palettePane.size}px` : null,
            "minmax(0, 1fr)",
            showInspector ? `${inspectorPane.size}px` : null
          ]
            .filter(Boolean)
            .join(" 6px ")
        }}
      >
        {showPalette && (
          <NodePalette
            specs={catalogQuery.data ?? []}
            categories={categoriesQuery.data ?? []}
            onAdd={(spec) => addNode(spec)}
          />
        )}
        {showPalette && (
          <div
            className={`arch-resize arch-resize-x${palettePane.dragging ? " arch-resize-active" : ""}`}
            aria-label="Resize the layer palette"
            {...palettePane.handleProps}
          />
        )}

        <div
          className="arch-canvas"
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
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDragStart={() => commit()}
            onPaneClick={() => {
              setSelectedId("");
              setSelectedGroupIds([]);
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
              openMenu(event);
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
            // Drag on empty canvas draws a selection box; the inspector still
            // edits one node, but delete and drag act on the whole selection.
            selectionOnDrag={tool === "select"}
            panOnDrag={tool === "move" ? true : [1, 2]}
            selectionMode={SelectionMode.Partial}
            multiSelectionKeyCode={["Meta", "Shift", "Control"]}
            // Curved edges: a straight line through a branch reads as one path
            // rather than two, which is exactly where a graph needs clarity.
            defaultEdgeOptions={{ type: "smoothstep", interactionWidth: EDGE_INTERACTION_WIDTH }}
            connectionLineType={ConnectionLineType.SmoothStep}
          >
            {/* Tools live on the canvas, not in the page header: they act on
                what is under the pointer, and a round trip to the top of the
                screen to switch between selecting and panning is the single
                most repeated motion in a node editor. */}
            <Panel position="top-left" className="arch-canvas-tools">
              <div className="arch-tool-group" role="group" aria-label="Canvas tool">
                <button
                  type="button"
                  aria-label="Select — drag to marquee-select"
                  title="Select (V) — drag to marquee-select"
                  aria-pressed={tool === "select"}
                  className={tool === "select" ? "arch-tool arch-tool-active" : "arch-tool"}
                  onClick={() => setTool("select")}
                >
                  <MousePointer2 size={17} aria-hidden />
                </button>
                <button
                  type="button"
                  aria-label="Move — drag to pan the canvas"
                  title="Move (H) — drag to pan the canvas"
                  aria-pressed={tool === "move"}
                  className={tool === "move" ? "arch-tool arch-tool-active" : "arch-tool"}
                  onClick={() => setTool("move")}
                >
                  <Hand size={17} aria-hidden />
                </button>
              </div>
              <div className="arch-tool-group" role="group" aria-label="Canvas actions">
                <button
                  type="button"
                  className="arch-tool"
                  aria-label="Frame the selection as a group"
                  title={
                    framedCount > 0
                      ? `Frame ${framedCount} selected node${framedCount === 1 ? "" : "s"} as a group (G)`
                      : "Add a group frame (G)"
                  }
                  onClick={addGroup}
                >
                  <Group size={17} aria-hidden />
                </button>
                <button
                  type="button"
                  className="arch-tool"
                  aria-label="Tidy the layout"
                  title="Tidy — lay the graph out left to right (T)"
                  onClick={tidy}
                >
                  <Wand2 size={17} aria-hidden />
                </button>
                <button
                  type="button"
                  className="arch-tool"
                  aria-label="Fit the graph to the view"
                  title="Fit to view (F)"
                  onClick={() => fitView({ duration: 200, padding: 0.15 })}
                >
                  <Frame size={17} aria-hidden />
                </button>
              </div>
            </Panel>
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls showInteractive={false} position="bottom-left" />
            <MiniMap pannable zoomable nodeColor={minimapColor} />
          </ReactFlow>
        </div>

        {showInspector && (
          <div
            className={`arch-resize arch-resize-x${inspectorPane.dragging ? " arch-resize-active" : ""}`}
            aria-label="Resize the node settings panel"
            {...inspectorPane.handleProps}
          />
        )}
        {showInspector &&
          (selectedGroup ? (
            <GroupInspector
              group={selectedGroup}
              onChange={(patch) => patchGroup(selectedGroup.id, patch)}
              onDelete={() => {
                commit();
                setGroups((current) =>
                  current.filter((group) => group.id !== selectedGroup.id)
                );
                setSelectedGroupIds([]);
              }}
            />
          ) : (
            <NodeInspector
              node={selected}
              issues={selected ? (nodeIssues.get(selected.id) ?? []) : []}
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
              onDelete={deleteSelection}
            />
          ))}
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

      <CanvasContextMenu state={menu} onClose={() => setMenu(null)} />
    </div>
  );
}
