"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Background,
  BackgroundVariant,
  ConnectionLineType,
  Controls,
  MiniMap,
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
  Code2,
  Copy,
  Download,
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
  Save,
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
import { CanvasNode } from "./canvas-node";
import { GroupFrame, type GroupFlowNode } from "./group-frame";
import { CodePanel } from "./code-panel";
import { NodeInspector } from "./node-inspector";
import { NodePalette } from "./node-palette";
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
  const { screenToFlowPosition } = useReactFlow();

  const [nodes, setNodes] = useState<ArchFlowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
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
  // Every pane edge is draggable and remembers where it was left.
  const palettePane = useResizable("palette", 232, { min: 160, max: 420 });
  const inspectorPane = useResizable("inspector", 320, { min: 240, max: 560, invert: true });
  const drawerPane = useResizable("drawer", 240, { min: 132, max: 560, axis: "vertical", invert: true });

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
  }, [architectureQuery.data, catalogQuery.data, specs, loaded]);

  const graph = useMemo(
    () => toGraph(nodes, edges, { num_classes_preview: numClasses, groups }),
    [nodes, edges, numClasses, groups]
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
  // `nodes`, so no validator or emitter has to know they exist.
  const groupNodes = useMemo(
    () =>
      groups.map((group) => ({
        id: group.id,
        type: "group" as const,
        position: { x: group.x, y: group.y },
        width: group.width,
        height: group.height,
        selectable: true,
        draggable: true,
        data: { title: group.title, tint: group.tint }
      })),
    [groups]
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

  const saveMutation = useMutation({
    mutationFn: () =>
      api.updateArchitecture(architectureId, { name, graph, version }),
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

  const markDirty = useCallback(() => setDirty(true), []);

  const addGroup = useCallback(() => {
    const id = `group_${Date.now().toString(36)}`;
    setGroups((current) => [
      ...current,
      {
        id,
        title: "New group",
        tint: current.length % 8,
        x: 40 + current.length * 24,
        y: 40 + current.length * 24,
        width: 420,
        height: 280
      }
    ]);
    markDirty();
  }, [markDirty]);

  // The frame renders inside React Flow and cannot reach this state directly,
  // so title and colour edits come back as events.
  useEffect(() => {
    const onTitle = (event: Event) => {
      const { id, title } = (event as CustomEvent).detail;
      setGroups((current) =>
        current.map((group) => (group.id === id ? { ...group, title } : group))
      );
      markDirty();
    };
    const onTint = (event: Event) => {
      const { id, tint } = (event as CustomEvent).detail;
      setGroups((current) =>
        current.map((group) => (group.id === id ? { ...group, tint } : group))
      );
      markDirty();
    };
    window.addEventListener("arch-group-title", onTitle);
    window.addEventListener("arch-group-tint", onTint);
    return () => {
      window.removeEventListener("arch-group-title", onTitle);
      window.removeEventListener("arch-group-tint", onTint);
    };
  }, [markDirty]);

  const onNodesChange = useCallback(
    (changes: NodeChange<ArchFlowNode>[]) => {
      // Group frames share the change stream; route theirs to `groups` and
      // let the rest fall through to the model nodes.
      const groupIds = new Set(groups.map((group) => group.id));
      const groupChanges = changes.filter(
        (change) => "id" in change && groupIds.has(change.id)
      );
      if (groupChanges.length > 0) {
        setGroups((current) =>
          current.map((group) => {
            let next = group;
            for (const change of groupChanges) {
              if (!("id" in change) || change.id !== group.id) continue;
              if (change.type === "position" && change.position) {
                next = { ...next, x: change.position.x, y: change.position.y };
              }
              if (change.type === "dimensions" && change.dimensions) {
                next = {
                  ...next,
                  width: change.dimensions.width,
                  height: change.dimensions.height
                };
              }
              if (change.type === "remove") return null as never;
            }
            return next;
          }).filter(Boolean)
        );
        if (groupChanges.some((change) => change.type !== "select")) markDirty();
      }
      const nodeChanges = changes.filter(
        (change) => !("id" in change) || !groupIds.has(change.id)
      );
      setNodes((current) => applyNodeChanges(nodeChanges, current));
      if (nodeChanges.some((change) => change.type !== "select" && change.type !== "dimensions")) {
        markDirty();
      }
      const selection = nodeChanges.find((change) => change.type === "select" && change.selected);
      if (selection && "id" in selection) setSelectedId(selection.id);
    },
    [groups, markDirty]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((current) => applyEdgeChanges(changes, current));
      if (changes.some((change) => change.type !== "select")) markDirty();
    },
    [markDirty]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((current) =>
        addEdge(
          { ...connection, id: edgeId(connection.source, connection.target), type: "smoothstep" },
          current
        )
      );
      markDirty();
    },
    [markDirty]
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
      markDirty();
    },
    [nodes, markDirty]
  );

  const selected = decoratedNodes.find((node) => node.id === selectedId) ?? null;

  const updateSelected = useCallback(
    (mutate: (node: ArchFlowNode) => ArchFlowNode) => {
      setNodes((current) => current.map((node) => (node.id === selectedId ? mutate(node) : node)));
      markDirty();
    },
    [selectedId, markDirty]
  );

  const tidy = useCallback(() => {
    setNodes((current) => autoLayout(current, edges));
    markDirty();
  }, [edges, markDirty]);

  const duplicateSelected = useCallback(() => {
    const source = nodes.find((node) => node.id === selectedId);
    if (!source) return;
    const id = nextNodeId(source.data.spec.type, nodes.map((node) => node.id));
    setNodes((current) => [
      ...current.map((node) => ({ ...node, selected: false })),
      {
        ...source,
        id,
        selected: true,
        // Offset so the copy is visibly a second node, not a perfect overlap.
        position: { x: source.position.x + 40, y: source.position.y + 40 },
        data: { ...source.data, params: { ...source.data.params } }
      }
    ]);
    setSelectedId(id);
    markDirty();
  }, [nodes, selectedId, markDirty]);

  const deleteSelected = useCallback(() => {
    // Whatever the selection box caught, plus the inspected node.
    const doomed = new Set(nodes.filter((node) => node.selected).map((node) => node.id));
    if (selectedId) doomed.add(selectedId);
    if (doomed.size === 0) return;
    setNodes((current) => current.filter((node) => !doomed.has(node.id)));
    setEdges((current) =>
      current.filter((edge) => !doomed.has(edge.source) && !doomed.has(edge.target))
    );
    setSelectedId("");
    markDirty();
  }, [nodes, selectedId, markDirty]);

  // Editor shortcuts, matching what a user already has in their fingers.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey;
      if (meta && event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (!saveMutation.isPending) saveMutation.mutate();
        return;
      }
      if (meta && event.key.toLowerCase() === "d") {
        event.preventDefault();
        duplicateSelected();
        return;
      }
      if (event.key === "Escape" && fullscreen) setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [saveMutation, duplicateSelected, fullscreen]);

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
          <div className="arch-view-controls" role="group" aria-label="Canvas tool">
            <IconButton
              aria-label="Select tool — drag to marquee-select"
              aria-pressed={tool === "select"}
              className={tool === "select" ? "arch-tool-active" : undefined}
              onClick={() => setTool("select")}
            >
              <MousePointer2 size={16} />
            </IconButton>
            <IconButton
              aria-label="Move tool — drag to pan the canvas"
              aria-pressed={tool === "move"}
              className={tool === "move" ? "arch-tool-active" : undefined}
              onClick={() => setTool("move")}
            >
              <Hand size={16} />
            </IconButton>
            <IconButton aria-label="Add a group frame" onClick={addGroup}>
              <Group size={16} />
            </IconButton>
          </div>
          <div className="arch-view-controls" role="group" aria-label="Canvas layout">
            <IconButton
              aria-label={showPalette ? "Hide the layer palette" : "Show the layer palette"}
              onClick={() => setShowPalette((value) => !value)}
            >
              {showPalette ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
            </IconButton>
            <IconButton
              aria-label={showInspector ? "Hide node settings" : "Show node settings"}
              onClick={() => setShowInspector((value) => !value)}
            >
              {showInspector ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
            </IconButton>
            <IconButton
              aria-label={fullscreen ? "Exit fullscreen" : "Expand the canvas to fullscreen"}
              onClick={() => setFullscreen((value) => !value)}
            >
              {fullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </IconButton>
          </div>
          <Button variant="ghost" size="sm" onClick={tidy} title="Lay the graph out left to right">
            <Wand2 size={15} aria-hidden /> Tidy
          </Button>
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
                `/training?architecture_id=${encodeURIComponent(architectureId)}&project_id=${encodeURIComponent(projectId)}`
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
            nodes={[...groupNodes, ...decoratedNodes] as never}
            edges={routedEdges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onPaneClick={() => setSelectedId("")}
            fitView
            proOptions={{ hideAttribution: false }}
            deleteKeyCode={["Backspace", "Delete"]}
            // Drag on empty canvas draws a selection box; the inspector still
            // edits one node, but delete and drag act on the whole selection.
            selectionOnDrag={tool === "select"}
            panOnDrag={tool === "move" ? true : [1, 2]}
            selectionMode={SelectionMode.Partial}
            multiSelectionKeyCode={["Meta", "Shift", "Control"]}
            // Curved edges: a straight line through a branch reads as one path
            // rather than two, which is exactly where a graph needs clarity.
            defaultEdgeOptions={{ type: "smoothstep" }}
            connectionLineType={ConnectionLineType.SmoothStep}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap pannable zoomable />
          </ReactFlow>
        </div>

        {showInspector && (
          <div
            className={`arch-resize arch-resize-x${inspectorPane.dragging ? " arch-resize-active" : ""}`}
            aria-label="Resize the node settings panel"
            {...inspectorPane.handleProps}
          />
        )}
        {showInspector && (
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
            onDuplicate={duplicateSelected}
            onDelete={deleteSelected}
          />
        )}
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
              size={16}
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
                          onClick={() => setSelectedId(issue.node_id ?? "")}
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
    </div>
  );
}
