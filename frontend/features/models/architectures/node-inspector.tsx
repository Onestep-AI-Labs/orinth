"use client";

import { BookMarked, Copy, Trash2 } from "lucide-react";
import { CodeEditor, lintCustomFunction, lintCustomLayer } from "./code-editor";
import { CategoryIcon, categoryColorVar, categoryTint } from "./node-visuals";
import { AdvancedField } from "@/features/training/advanced-settings";
import { Button, Field } from "@/features/platform/ui";
import type { ArchitectureIssue } from "@/types/api";
import type { ArchFlowNode } from "./graph-state";
import { blockRepeat, formatShape } from "./graph-state";

/**
 * Settings for the selected node.
 *
 * Fields render from `NodeSpec.params` through `AdvancedField` — the same
 * catalog-driven renderer the training form's advanced accordion uses. That is
 * the point of reusing `AdvancedParameterSpec` on the backend: a new node type
 * gets its inspector without a line of frontend code.
 */
export function NodeInspector({
  node,
  issues,
  onRename,
  onParamChange,
  onDuplicate,
  onDelete
}: {
  node: ArchFlowNode | null;
  issues: ArchitectureIssue[];
  onRename: (label: string) => void;
  onParamChange: (key: string, value: unknown) => void;
  onDuplicate: () => void;
  onDelete: () => void;
}) {
  if (!node) {
    return (
      <aside className="arch-inspector" aria-label="Node settings">
        <p className="arch-inspector-hint">
          Select a node to edit its settings, or add one from the palette.
        </p>
      </aside>
    );
  }

  const { spec, params } = node.data;
  const grouped = groupParams(spec.params);
  const repeat = blockRepeat(node.data);

  return (
    <aside className="arch-inspector" aria-label="Node settings">
      {/* One scroll region from the header down, not just over the fields. A
          block's description and source line are often the longest thing in the
          panel, and pinning them meant a tall node's settings scrolled inside a
          sliver while most of the rail sat frozen. */}
      <div className="arch-inspector-scroll">
        <div
          className="arch-inspector-head"
          style={{ background: categoryTint(spec.category, 10) }}
        >
          <p className="arch-inspector-eyebrow" style={{ color: categoryColorVar(spec.category) }}>
            <span
              className="arch-palette-dot"
              style={{ background: categoryColorVar(spec.category) }}
              aria-hidden
            />
            <CategoryIcon category={spec.category} size={12} />
            {spec.category}
            {spec.kind === "block" && <span className="arch-inspector-kind">Block</span>}
          </p>
          <p className="arch-inspector-title">{spec.name}</p>
          <p className="arch-inspector-desc">{spec.description}</p>
          {/* Where a block's defaults came from. A number on this canvas should
              be traceable to the config it was read out of, not taken on faith. */}
          {spec.source && (
            <p className="arch-inspector-source">
              <BookMarked size={12} aria-hidden /> {spec.source}
            </p>
          )}
        </div>

        <div className="arch-inspector-facts">
          <div>
            <p className="arch-inspector-fact-label">Output shape</p>
            <p className="arch-inspector-fact-value">{formatShape(node.data.shape)}</p>
          </div>
          <div>
            <p className="arch-inspector-fact-label">{repeat > 1 ? "Expands to" : "Node id"}</p>
            <p className="arch-inspector-fact-value">
              {repeat > 1 ? `${repeat} layers` : node.id}
            </p>
          </div>
        </div>

        {issues.length > 0 && (
          <ul className="arch-inspector-issues">
            {issues.map((issue, index) => (
              <li key={index} className={`arch-issue arch-issue-${issue.severity}`}>
                {issue.message}
              </li>
            ))}
          </ul>
        )}

        <div className="arch-inspector-body">
          <Field label="Label">
            <input
              value={node.data.label}
              onChange={(event) => onRename(event.target.value)}
              placeholder={spec.name}
            />
          </Field>

          {grouped.map(([group, specs]) => (
            <div className="arch-inspector-group" key={group}>
              <p className="advanced-group-label">{group}</p>
              <div className="form-grid">
                {specs.map((param) =>
                  // A `code` param on a custom node gets the editor: syntax
                  // highlighting and lint feedback matter far more here than
                  // the generic textarea the shared renderer would give it.
                  param.type === "code" && param.key !== "class_name" ? (
                    <CodeEditor
                      key={param.key}
                      ariaLabel={param.label}
                      value={String(params[param.key] ?? "")}
                      minRows={node.data.spec.type === "custom_function" ? 3 : 10}
                      findings={
                        node.data.spec.type === "custom_function"
                          ? lintCustomFunction(String(params[param.key] ?? ""))
                          : lintCustomLayer(
                              String(params[param.key] ?? ""),
                              String(params.class_name ?? "")
                            )
                      }
                      onChange={(value) => onParamChange(param.key, value)}
                    />
                  ) : (
                    <AdvancedField
                      key={param.key}
                      spec={param}
                      value={params[param.key]}
                      onChange={(value) => onParamChange(param.key, value)}
                    />
                  )
                )}
              </div>
            </div>
          ))}

          {spec.params.length === 0 && (
            <p className="arch-inspector-hint">This layer has no settings.</p>
          )}
        </div>
      </div>

      <div className="arch-inspector-foot">
        <Button variant="secondary" size="sm" onClick={onDuplicate}>
          <Copy size={15} aria-hidden /> Duplicate
        </Button>
        <Button variant="danger" size="sm" onClick={onDelete}>
          <Trash2 size={15} aria-hidden /> Remove
        </Button>
      </div>
    </aside>
  );
}

function groupParams(params: ArchFlowNode["data"]["spec"]["params"]) {
  const byGroup = new Map<string, typeof params>();
  for (const param of params) {
    byGroup.set(param.group, [...(byGroup.get(param.group) ?? []), param]);
  }
  return [...byGroup.entries()];
}
