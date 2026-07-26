"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { labelColor } from "@/features/platform/utils";

export type ChartSeries = { key: string; label: string };
export type ChartRow = Record<string, number | string | undefined>;

type Point = { x: number; y: number; value: number };

// The SVG is rendered at 1:1 with its container (viewBox = measured pixel box),
// so axis text and marks keep their real px size instead of being magnified
// when the panel stretches full-width. Height is fixed and bounded.
const HEIGHT = 260;
const PAD = { top: 14, right: 18, bottom: 40, left: 46 };
const MIN_WIDTH = 320;

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return [Number.isFinite(min) ? min : 0];
  }
  const span = max - min;
  const rawStep = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  const step = (normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1) * magnitude;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= max + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function formatTick(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 100000)) return value.toExponential(1);
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(abs < 1 ? 2 : 1);
}

function formatValue(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 100000)) return value.toExponential(2);
  return value.toFixed(abs < 1 ? 4 : 3);
}

/**
 * Interactive multi-series line chart — axes, gridlines, legend, and a hover
 * crosshair with a per-series tooltip. Self-contained SVG (no chart library);
 * one shared Y scale per panel so overlaid series stay comparable. Renders at
 * the container's measured width so text is never scaled up.
 */
export function TrainingChart({
  title,
  rows,
  series,
  xKey,
  xLabel,
  colorOffset = 0
}: {
  title: string;
  rows: ChartRow[];
  series: ChartSeries[];
  xKey?: string;
  xLabel: string;
  colorOffset?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const measure = () => setWidth(Math.max(MIN_WIDTH, Math.round(node.clientWidth)));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;

  const model = useMemo(() => {
    const active = series.filter((entry) => rows.some((row) => Number.isFinite(Number(row[entry.key]))));
    if (active.length === 0) return null;
    const xs = rows.map((row, index) => (xKey ? Number(row[xKey]) : index));
    const xValues = xs.every((value) => Number.isFinite(value)) ? xs : rows.map((_, index) => index);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const xSpan = Math.max(xMax - xMin, 1e-9);

    let yMin = Infinity;
    let yMax = -Infinity;
    for (const entry of active) {
      for (const row of rows) {
        const value = Number(row[entry.key]);
        if (Number.isFinite(value)) {
          yMin = Math.min(yMin, value);
          yMax = Math.max(yMax, value);
        }
      }
    }
    if (yMin === yMax) {
      yMin -= 1;
      yMax += 1;
    }
    const yPad = (yMax - yMin) * 0.08;
    const lo = yMin - yPad;
    const hi = yMax + yPad;
    const ySpan = Math.max(hi - lo, 1e-9);

    const sx = (value: number) => PAD.left + ((value - xMin) / xSpan) * plotW;
    const sy = (value: number) => PAD.top + (1 - (value - lo) / ySpan) * plotH;

    const lines = active.map((entry, seriesIndex) => {
      // cells stays row-aligned (null where a series has no value that row, e.g.
      // val_loss only logged per epoch) so hover-by-row-index lines up; the path
      // is drawn only through the points that exist.
      const cells: (Point | null)[] = rows.map((row, index) => {
        const value = Number(row[entry.key]);
        return Number.isFinite(value) ? { x: sx(xValues[index]), y: sy(value), value } : null;
      });
      const present = cells.filter((cell): cell is Point => cell !== null);
      return {
        ...entry,
        color: labelColor(colorOffset + seriesIndex),
        cells,
        path: present.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ")
      };
    });

    return {
      lines,
      xValues,
      yTicks: niceTicks(lo, hi, 4).filter((tick) => tick >= lo && tick <= hi),
      xTicks: xTickIndices(rows.length).map((index) => ({ index, x: sx(xValues[index]), label: formatTick(xValues[index]) })),
      sx,
      sy,
      count: rows.length
    };
  }, [rows, series, xKey, colorOffset, plotW, plotH]);

  useEffect(() => {
    if (model && hover !== null && hover > model.count - 1) setHover(null);
  }, [model, hover]);

  if (!model) return null;

  function handleMove(event: React.MouseEvent<HTMLDivElement>) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || !model) return;
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - PAD.left) / plotW));
    setHover(Math.round(ratio * (model.count - 1)));
  }

  const hoverX = hover !== null ? model.sx(model.xValues[hover]) : null;
  const tooltipLeft = hoverX !== null ? (hoverX / width) * 100 : 0;
  const flip = tooltipLeft > 62;

  return (
    <figure className="training-chart">
      <figcaption className="training-chart-head">
        <span className="training-chart-title">{title}</span>
        <span className="training-chart-legend">
          {model.lines.map((line) => (
            <span key={line.key} className="training-chart-legend-item">
              <span className="training-chart-swatch" style={{ backgroundColor: line.color }} />
              {line.label}
            </span>
          ))}
        </span>
      </figcaption>
      <div
        ref={containerRef}
        className="training-chart-plot"
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
      >
        <svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label={title}>
          {model.yTicks.map((tick) => {
            const y = model.sy(tick);
            return (
              <g key={`y-${tick}`}>
                <line x1={PAD.left} y1={y} x2={width - PAD.right} y2={y} className="training-chart-gridline" />
                <text x={PAD.left - 8} y={y + 3.5} textAnchor="end" className="training-chart-axis-text">
                  {formatTick(tick)}
                </text>
              </g>
            );
          })}
          {model.xTicks.map((tick) => (
            <text key={`x-${tick.index}`} x={tick.x} y={HEIGHT - PAD.bottom + 18} textAnchor="middle" className="training-chart-axis-text">
              {tick.label}
            </text>
          ))}
          <text x={PAD.left + plotW / 2} y={HEIGHT - 6} textAnchor="middle" className="training-chart-axis-label">
            {xLabel}
          </text>
          {model.lines.map((line) => (
            <path key={line.key} d={line.path} fill="none" stroke={line.color} strokeWidth={1.75} vectorEffect="non-scaling-stroke" />
          ))}
          {hoverX !== null && (
            <line x1={hoverX} y1={PAD.top} x2={hoverX} y2={PAD.top + plotH} className="training-chart-crosshair" />
          )}
          {hover !== null &&
            model.lines.map((line) => {
              const point = line.cells[hover];
              if (!point) return null;
              return <circle key={`dot-${line.key}`} cx={point.x} cy={point.y} r={3} fill={line.color} className="training-chart-dot" />;
            })}
        </svg>
        {hover !== null && (
          <div className={`training-chart-tip${flip ? " flip" : ""}`} style={{ left: `${tooltipLeft}%` }}>
            <span className="training-chart-tip-x">
              {xLabel} {formatTick(model.xValues[hover])}
            </span>
            {model.lines.map((line) => {
              const value = line.cells[hover]?.value;
              if (value === undefined) return null;
              return (
                <span key={`tip-${line.key}`} className="training-chart-tip-row">
                  <span className="training-chart-swatch" style={{ backgroundColor: line.color }} />
                  <span className="training-chart-tip-label">{line.label}</span>
                  <span className="training-chart-tip-value">{formatValue(value)}</span>
                </span>
              );
            })}
          </div>
        )}
      </div>
    </figure>
  );
}

function xTickIndices(count: number, max = 7): number[] {
  if (count <= 1) return [0];
  const target = Math.min(max, count);
  const step = (count - 1) / (target - 1);
  const indices = new Set<number>();
  for (let i = 0; i < target; i += 1) indices.add(Math.round(i * step));
  return [...indices].sort((a, b) => a - b);
}
