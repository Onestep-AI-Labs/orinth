"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState } from "react";
import { CheckCircle2, MousePointer2, Pentagon, Save, Square, Trash2, Undo2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api, apiAssetUrl } from "@/lib/api";
import { isNlpTask, labelColor, pointsAttr } from "@/features/platform/utils";
import type { DatasetAnnotation, DatasetItemDetail, DatasetSummary } from "@/types/api";

type AnnotationTool = "select" | "box" | "polygon";
type AnnotationBox = NonNullable<DatasetAnnotation["bbox"]>;
type BoxDraft = { startX: number; startY: number; x: number; y: number; width: number; height: number };
type DragState = {
  index: number;
  startX: number;
  startY: number;
  original: DatasetAnnotation;
};

export function AnnotationEditor({
  dataset,
  item,
  onSaved
}: {
  dataset: DatasetSummary;
  item: DatasetItemDetail;
  onSaved: () => void | Promise<void>;
}) {
  if (isNlpTask(dataset.task_type)) {
    return <NlpAnnotationEditor dataset={dataset} item={item} onSaved={onSaved} />;
  }
  const [annotations, setAnnotations] = useState<DatasetAnnotation[]>(item.annotations);
  const [draft, setDraft] = useState<number[][]>([]);
  const [classId, setClassId] = useState(0);
  const [box, setBox] = useState({ x: 0, y: 0, width: 80, height: 80 });
  const [boxDraft, setBoxDraft] = useState<BoxDraft | null>(null);
  const [activeTool, setActiveTool] = useState<AnnotationTool>(
    dataset.task_type === "segmentation" ? "polygon" : dataset.task_type === "object_detection" ? "box" : "select"
  );
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const editable = dataset.editable;
  const drawable = dataset.task_type !== "classification";
  const imageUrl = apiAssetUrl(item.image_url);
  const saveMutation = useMutation({
    mutationFn: () => api.saveDatasetAnnotations(dataset.id, item.split, item.id, annotations),
    onSuccess: async (data) => {
      setAnnotations(data.annotations);
      setDraft([]);
      await onSaved();
    }
  });
  const labelMutation = useMutation({
    mutationFn: () => api.updateDatasetItemLabel(dataset.id, item.split, item.id, { class_id: classId }),
    onSuccess: async (data) => {
      setAnnotations(data.annotations);
      await onSaved();
    }
  });

  useEffect(() => {
    setAnnotations(item.annotations);
    setClassId(item.annotations[0]?.class_id ?? 0);
    setDraft([]);
    setBoxDraft(null);
    setSelectedIndex(null);
    setDragState(null);
  }, [item.annotations, item.id]);

  useEffect(() => {
    setActiveTool(dataset.task_type === "segmentation" ? "polygon" : dataset.task_type === "object_detection" ? "box" : "select");
  }, [dataset.task_type, item.id]);

  useEffect(() => {
    if (activeTool !== "polygon") setDraft([]);
    if (activeTool !== "box") setBoxDraft(null);
  }, [activeTool]);

  function svgPoint(event: React.PointerEvent<SVGSVGElement> | React.MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * item.width;
    const y = ((event.clientY - rect.top) / rect.height) * item.height;
    return {
      x: Math.max(0, Math.min(item.width, x)),
      y: Math.max(0, Math.min(item.height, y))
    };
  }

  function addPoint(event: React.MouseEvent<SVGSVGElement>) {
    if (!editable || !drawable || activeTool !== "polygon") return;
    if (event.target !== event.currentTarget) return;
    const { x, y } = svgPoint(event);
    setDraft((points) => [...points, [x, y]]);
  }

  function beginBox(event: React.PointerEvent<SVGSVGElement>) {
    if (!editable || !drawable || activeTool !== "box" || event.target !== event.currentTarget) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = svgPoint(event);
    setBoxDraft({ startX: point.x, startY: point.y, x: point.x, y: point.y, width: 0, height: 0 });
  }

  function updateBox(event: React.PointerEvent<SVGSVGElement>) {
    if (dragState) {
      const point = svgPoint(event);
      const dx = point.x - dragState.startX;
      const dy = point.y - dragState.startY;
      setAnnotations((rows) =>
        rows.map((annotation, index) => (index === dragState.index ? moveAnnotation(dragState.original, dx, dy, item.width, item.height) : annotation))
      );
      return;
    }
    if (!boxDraft || !drawable || activeTool !== "box") return;
    const point = svgPoint(event);
    setBoxDraft({
      ...boxDraft,
      x: Math.min(boxDraft.startX, point.x),
      y: Math.min(boxDraft.startY, point.y),
      width: Math.abs(point.x - boxDraft.startX),
      height: Math.abs(point.y - boxDraft.startY)
    });
  }

  function finishBox(event: React.PointerEvent<SVGSVGElement>) {
    if (dragState) {
      setDragState(null);
      return;
    }
    if (!boxDraft || !drawable || activeTool !== "box") return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    const nextBox = {
      x: Math.round(boxDraft.x),
      y: Math.round(boxDraft.y),
      width: Math.round(boxDraft.width),
      height: Math.round(boxDraft.height)
    };
    setBoxDraft(null);
    if (nextBox.width < 2 || nextBox.height < 2) return;
    setBox(nextBox);
    addBox(nextBox);
  }

  function addBox(nextBox = box) {
    setAnnotations((rows) => [
      ...rows,
      {
        class_id: classId,
        class_name: dataset.labels[classId],
        kind: "box",
        bbox: nextBox,
        polygon: [],
        text: null,
        question: null,
        answer: null
      }
    ]);
  }

  function finishDraft() {
    if (draft.length < 3) return;
    setAnnotations((rows) => [
      ...rows,
      {
        class_id: classId,
        class_name: dataset.labels[classId],
        kind: "polygon",
        bbox: null,
        polygon: draft,
        text: null,
        question: null,
        answer: null
      }
    ]);
    setDraft([]);
  }

  function beginMove(event: React.PointerEvent<SVGElement>, annotation: DatasetAnnotation, index: number) {
    event.stopPropagation();
    setSelectedIndex(index);
    if (!editable || activeTool !== "select") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = svgPointFromElement(event, item.width, item.height);
    setDragState({ index, startX: point.x, startY: point.y, original: annotation });
  }

  function selectAnnotation(index: number) {
    setSelectedIndex(index);
  }

  function clearSelection(event: React.PointerEvent<SVGSVGElement>) {
    if (activeTool === "select" && event.target === event.currentTarget) setSelectedIndex(null);
    beginBox(event);
  }

  function selectTool(tool: AnnotationTool) {
    setActiveTool(tool);
    setDragState(null);
    setBoxDraft(null);
  }

  return (
    <div className="annotation-panel">
      <div className="annotation-toolbar" aria-label="Annotation tools">
        <button
          type="button"
          className={`icon-button annotation-tool-button ${activeTool === "select" ? "annotation-tool-active" : ""}`}
          onClick={() => selectTool("select")}
          disabled={!drawable}
          title="Select and move annotations"
          aria-pressed={activeTool === "select"}
        >
          <MousePointer2 size={16} />
        </button>
        <button
          type="button"
          className={`icon-button annotation-tool-button ${activeTool === "box" ? "annotation-tool-active" : ""}`}
          onClick={() => selectTool("box")}
          disabled={!drawable || !editable}
          title="Bounding box tool"
          aria-pressed={activeTool === "box"}
        >
          <Square size={16} />
        </button>
        <button
          type="button"
          className={`icon-button annotation-tool-button ${activeTool === "polygon" ? "annotation-tool-active" : ""}`}
          onClick={() => selectTool("polygon")}
          disabled={!drawable || !editable}
          title="Polygon tool"
          aria-pressed={activeTool === "polygon"}
        >
          <Pentagon size={16} />
        </button>
      </div>
      <div className="annotation-stage">
        {imageUrl && <img src={imageUrl} alt={item.filename} />}
        <svg
          className={`annotation-svg annotation-svg-${activeTool}`}
          viewBox={`0 0 ${item.width} ${item.height}`}
          preserveAspectRatio="none"
          onClick={addPoint}
          onPointerDown={clearSelection}
          onPointerMove={updateBox}
          onPointerUp={finishBox}
        >
          {annotations.map((annotation, index) => {
            const selected = selectedIndex === index;
            return annotation.kind === "box" && annotation.bbox ? (
              <rect
                key={`${annotation.class_name}-${index}`}
                x={annotation.bbox.x}
                y={annotation.bbox.y}
                width={annotation.bbox.width}
                height={annotation.bbox.height}
                className={`annotation-poly ${selected ? "annotation-selected" : ""}`}
                style={{ stroke: labelColor(annotation.class_id), fill: `${labelColor(annotation.class_id)}33` }}
                onPointerDown={(event) => beginMove(event, annotation, index)}
                onClick={(event) => {
                  event.stopPropagation();
                  selectAnnotation(index);
                }}
              />
            ) : annotation.kind === "polygon" ? (
              <polygon
                key={`${annotation.class_name}-${index}`}
                points={pointsAttr(annotation.polygon)}
                className={`annotation-poly ${selected ? "annotation-selected" : ""}`}
                style={{ stroke: labelColor(annotation.class_id), fill: `${labelColor(annotation.class_id)}33` }}
                onPointerDown={(event) => beginMove(event, annotation, index)}
                onClick={(event) => {
                  event.stopPropagation();
                  selectAnnotation(index);
                }}
              />
            ) : null;
          })}
          {draft.length > 1 && <polyline points={pointsAttr(draft)} className="annotation-draft" />}
          {draft.map((point, index) => (
            <circle cx={point[0]} cy={point[1]} r={5} className="annotation-point" key={index} />
          ))}
          {boxDraft && (
            <rect
              x={boxDraft.x}
              y={boxDraft.y}
              width={boxDraft.width}
              height={boxDraft.height}
              className="annotation-draft-box"
            />
          )}
        </svg>
      </div>

      <div className="annotation-actions">
        <select value={classId} onChange={(event) => setClassId(Number(event.target.value))} disabled={!editable}>
          {dataset.labels.map((label, index) => (
            <option value={index} key={label}>
              {label}
            </option>
          ))}
        </select>
        {dataset.task_type === "classification" && (
          <button className="primary-button" onClick={() => labelMutation.mutate()} disabled={!editable || labelMutation.isPending}>
            <CheckCircle2 size={16} /> Save label
          </button>
        )}
        {drawable && activeTool === "box" && (
          <button className="secondary-button" onClick={() => addBox()} disabled={!editable}>
            <CheckCircle2 size={16} /> Add box
          </button>
        )}
        {drawable && activeTool === "polygon" && (
          <button className="secondary-button" onClick={finishDraft} disabled={!editable || draft.length < 3}>
            <CheckCircle2 size={16} /> Finish
          </button>
        )}
        <button className="icon-button" onClick={() => setDraft((points) => points.slice(0, -1))} disabled={!editable || activeTool !== "polygon" || draft.length === 0} title="Undo">
          <Undo2 size={16} />
        </button>
        <button className="icon-button" onClick={() => setDraft([])} disabled={!editable || activeTool !== "polygon" || draft.length === 0} title="Clear">
          <Trash2 size={16} />
        </button>
        {drawable && (
          <button className="primary-button" onClick={() => saveMutation.mutate()} disabled={!editable || saveMutation.isPending}>
            <Save size={16} /> Save
          </button>
        )}
      </div>

      {drawable && activeTool === "box" && (
        <div className="grid grid-cols-4 gap-2">
          {(["x", "y", "width", "height"] as const).map((key) => (
            <input
              key={key}
              type="number"
              value={box[key]}
              min={0}
              onChange={(event) => setBox((value) => ({ ...value, [key]: Number(event.target.value) }))}
            />
          ))}
        </div>
      )}
      <AnnotationTable
        annotations={annotations}
        setAnnotations={setAnnotations}
        editable={editable}
        selectedIndex={selectedIndex}
        setSelectedIndex={setSelectedIndex}
      />
      <MutationError mutations={[saveMutation, labelMutation]} />
    </div>
  );
}

function NlpAnnotationEditor({
  dataset,
  item,
  onSaved
}: {
  dataset: DatasetSummary;
  item: DatasetItemDetail;
  onSaved: () => void | Promise<void>;
}) {
  const initial = item.annotations[0];
  const [classId, setClassId] = useState(initial?.class_id ?? 0);
  const [summary, setSummary] = useState(initial?.text ?? initial?.answer ?? "");
  const [question, setQuestion] = useState(initial?.question ?? "");
  const [answer, setAnswer] = useState(initial?.answer ?? initial?.text ?? "");
  const editable = dataset.editable;
  const saveMutation = useMutation({
    mutationFn: () => {
      let annotations: DatasetAnnotation[] = [];
      if (dataset.task_type === "text_classification") {
        annotations = [
          {
            class_id: classId,
            class_name: dataset.labels[classId] ?? String(classId),
            kind: "classification",
            bbox: null,
            polygon: [],
            text: null,
            question: null,
            answer: null
          }
        ];
      } else if (dataset.task_type === "summarization") {
        annotations = summary.trim()
          ? [
              {
                class_id: 0,
                class_name: dataset.labels[0] ?? "summary",
                kind: "summary",
                bbox: null,
                polygon: [],
                text: summary.trim(),
                question: null,
                answer: summary.trim()
              }
            ]
          : [];
      } else {
        annotations = question.trim() && answer.trim()
          ? [
              {
                class_id: 0,
                class_name: dataset.labels[0] ?? "answer",
                kind: "qa",
                bbox: null,
                polygon: [],
                text: answer.trim(),
                question: question.trim(),
                answer: answer.trim()
              }
            ]
          : [];
      }
      return api.saveDatasetAnnotations(dataset.id, item.split, item.id, annotations);
    },
    onSuccess: async () => {
      await onSaved();
    }
  });

  useEffect(() => {
    const next = item.annotations[0];
    setClassId(next?.class_id ?? 0);
    setSummary(next?.text ?? next?.answer ?? "");
    setQuestion(next?.question ?? "");
    setAnswer(next?.answer ?? next?.text ?? "");
  }, [item.annotations, item.id]);

  return (
    <div className="annotation-panel nlp-annotation-panel">
      <div className="text-item-preview">
        {item.text_content || item.text_preview || ""}
      </div>
      {dataset.task_type === "text_classification" ? (
        <select value={classId} onChange={(event) => setClassId(Number(event.target.value))} disabled={!editable}>
          {dataset.labels.map((label, index) => (
            <option value={index} key={label}>{label}</option>
          ))}
        </select>
      ) : dataset.task_type === "summarization" ? (
        <textarea value={summary} onChange={(event) => setSummary(event.target.value)} rows={5} disabled={!editable} />
      ) : (
        <>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} disabled={!editable} placeholder="Question" />
          <textarea value={answer} onChange={(event) => setAnswer(event.target.value)} rows={4} disabled={!editable} placeholder="Answer" />
        </>
      )}
      <button className="primary-button" onClick={() => saveMutation.mutate()} disabled={!editable || saveMutation.isPending}>
        <Save size={16} /> Save
      </button>
      <MutationError mutations={[saveMutation]} />
    </div>
  );
}

function svgPointFromElement(event: React.PointerEvent<SVGElement>, width: number, height: number) {
  const svg = event.currentTarget.ownerSVGElement;
  const rect = (svg ?? event.currentTarget).getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * width;
  const y = ((event.clientY - rect.top) / rect.height) * height;
  return {
    x: Math.max(0, Math.min(width, x)),
    y: Math.max(0, Math.min(height, y))
  };
}

function moveAnnotation(annotation: DatasetAnnotation, dx: number, dy: number, imageWidth: number, imageHeight: number): DatasetAnnotation {
  if (annotation.kind === "box" && annotation.bbox) {
    return { ...annotation, bbox: moveBox(annotation.bbox, dx, dy, imageWidth, imageHeight) };
  }
  if (annotation.kind === "polygon" && annotation.polygon.length > 0) {
    return { ...annotation, polygon: movePolygon(annotation.polygon, dx, dy, imageWidth, imageHeight) };
  }
  return annotation;
}

function moveBox(box: AnnotationBox, dx: number, dy: number, imageWidth: number, imageHeight: number): AnnotationBox {
  return {
    ...box,
    x: clamp(box.x + dx, 0, Math.max(0, imageWidth - box.width)),
    y: clamp(box.y + dy, 0, Math.max(0, imageHeight - box.height))
  };
}

function movePolygon(polygon: number[][], dx: number, dy: number, imageWidth: number, imageHeight: number): number[][] {
  const xs = polygon.map((point) => point[0]);
  const ys = polygon.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const clampedDx = clamp(dx, -minX, imageWidth - maxX);
  const clampedDy = clamp(dy, -minY, imageHeight - maxY);
  return polygon.map(([x, y]) => [x + clampedDx, y + clampedDy]);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function AnnotationTable({
  annotations,
  setAnnotations,
  editable,
  selectedIndex,
  setSelectedIndex
}: {
  annotations: DatasetAnnotation[];
  setAnnotations: React.Dispatch<React.SetStateAction<DatasetAnnotation[]>>;
  editable: boolean;
  selectedIndex: number | null;
  setSelectedIndex: (index: number | null) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Kind</th>
            <th>Points</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {annotations.map((annotation, index) => (
            <tr
              className={selectedIndex === index ? "annotation-row-selected" : ""}
              key={`${annotation.class_name}-${index}`}
              onClick={() => setSelectedIndex(index)}
            >
              <td>
                <span className="class-dot" style={{ backgroundColor: labelColor(annotation.class_id) }} />
                {annotation.class_name}
              </td>
              <td>{annotation.kind}</td>
              <td>{annotation.kind === "box" ? "box" : annotation.polygon.length}</td>
              <td>
                <button
                  className="icon-button"
                  disabled={!editable}
                  onClick={(event) => {
                    event.stopPropagation();
                    setAnnotations((rows) => rows.filter((_row, rowIndex) => rowIndex !== index));
                    setSelectedIndex(null);
                  }}
                  title="Remove annotation"
                >
                  <Trash2 size={14} />
                </button>
              </td>
            </tr>
          ))}
          {annotations.length === 0 && (
            <tr>
              <td colSpan={4}>No annotations</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function MutationError({ mutations }: { mutations: Array<{ error: Error | null }> }) {
  const error = mutations.find((mutation) => mutation.error)?.error;
  return error ? <p className="error-text">{error.message}</p> : null;
}
