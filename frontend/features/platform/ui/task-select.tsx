"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  Check,
  ChevronDown,
  FileImage,
  FileText,
  MessageSquareText,
  ScanSearch,
  Search,
  Shapes,
  Sparkles
} from "lucide-react";
import {
  LLM_TASK_TYPES,
  NLP_TASK_TYPES,
  VISION_TASK_TYPES,
  formatDatasetTask,
  taskDescription
} from "@/features/platform/utils";
import type { TaskType } from "@/types/api";

type TaskDomain = "vision" | "nlp" | "llm";

const DOMAIN_ORDER: TaskDomain[] = ["vision", "nlp", "llm"];
const DOMAIN_LABEL: Record<TaskDomain, string> = {
  vision: "Vision",
  nlp: "NLP",
  llm: "LLM"
};
const DOMAIN_TASKS: Record<TaskDomain, TaskType[]> = {
  vision: VISION_TASK_TYPES,
  nlp: NLP_TASK_TYPES,
  llm: LLM_TASK_TYPES
};

function domainForTask(task: TaskType): TaskDomain {
  if (NLP_TASK_TYPES.includes(task)) return "nlp";
  if (LLM_TASK_TYPES.includes(task)) return "llm";
  return "vision";
}

function taskIcon(task: TaskType) {
  if (task === "classification") return <FileImage size={16} />;
  if (task === "object_detection") return <ScanSearch size={16} />;
  if (task === "segmentation") return <Shapes size={16} />;
  if (task === "text_classification") return <FileText size={16} />;
  if (task === "summarization") return <BarChart3 size={16} />;
  if (task === "question_answering") return <MessageSquareText size={16} />;
  if (task === "llm_finetune") return <Sparkles size={16} />;
  return <BarChart3 size={16} />;
}

function matchesQuery(task: TaskType, query: string): boolean {
  if (!query) return true;
  return (
    formatDatasetTask(task).toLowerCase().includes(query) ||
    taskDescription(task).toLowerCase().includes(query)
  );
}

/**
 * Single-select task picker that splits its options into Vision / NLP / LLM
 * tabs and lets you search across all of them. It is the shared control for the
 * training, testing, inference, and serving forms, which each pass the project's
 * allowed task types — mirroring the grouped creation picker so a task means the
 * same thing everywhere.
 *
 * A native `<select>` cannot render the domain tabs, search, icons, or per-task
 * descriptions, so this keeps the closed, summarised affordance of a dropdown
 * (mirroring {@link MultiSelect}) and opens a tabbed, searchable panel instead.
 */
export function TaskSelect({
  value,
  onChange,
  options,
  disabled,
  id
}: {
  value: TaskType;
  onChange: (task: TaskType) => void;
  /** The project's allowed task types, in the order the caller supplies them. */
  options: TaskType[];
  disabled?: boolean;
  id?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // The domains this project actually uses, in canonical order. A single-domain
  // project shows no tab bar — one tab is noise.
  const presentDomains = useMemo(
    () => DOMAIN_ORDER.filter((domain) => DOMAIN_TASKS[domain].some((task) => options.includes(task))),
    [options]
  );
  const showTabs = presentDomains.length > 1;

  const [activeDomain, setActiveDomain] = useState<TaskDomain>(() => domainForTask(value));
  // Keep the active tab within the domains that exist, without an effect.
  const effectiveDomain = presentDomains.includes(activeDomain)
    ? activeDomain
    : presentDomains[0] ?? "vision";

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const trimmed = query.trim().toLowerCase();
  const searching = trimmed.length > 0;

  // While searching, look across every domain and keep the group headers so the
  // matches stay grouped; otherwise show just the active tab's tasks.
  const groups = useMemo(() => {
    const domains = searching ? presentDomains : [effectiveDomain];
    return domains
      .map((domain) => ({
        domain,
        tasks: DOMAIN_TASKS[domain].filter((task) => options.includes(task) && matchesQuery(task, trimmed))
      }))
      .filter((group) => group.tasks.length > 0);
  }, [searching, presentDomains, effectiveDomain, options, trimmed]);

  const firstMatch = groups[0]?.tasks[0];
  const selected = options.includes(value) ? value : null;

  function openPanel() {
    setQuery("");
    setActiveDomain(presentDomains.includes(domainForTask(value)) ? domainForTask(value) : presentDomains[0] ?? "vision");
    setOpen(true);
    requestAnimationFrame(() => searchRef.current?.focus());
  }

  function pick(task: TaskType) {
    if (task !== value) onChange(task);
    setOpen(false);
  }

  return (
    <div className="task-select" ref={rootRef}>
      <button
        type="button"
        id={id}
        className="task-select-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled || options.length === 0}
        onClick={() => (open ? setOpen(false) : openPanel())}
      >
        {selected ? (
          <span className="task-select-value">
            <span className="task-select-icon">{taskIcon(selected)}</span>
            <span className="task-select-value-copy">{formatDatasetTask(selected)}</span>
            <span className="task-select-domain-tag">{DOMAIN_LABEL[domainForTask(selected)]}</span>
          </span>
        ) : (
          <span className="task-select-placeholder">Choose a task…</span>
        )}
        <ChevronDown size={16} className="task-select-chevron" />
      </button>
      {open && (
        <div className="task-select-panel">
          <div className="task-select-head">
            {showTabs ? (
              <div className="segmented-control task-select-tabs" role="tablist">
                {presentDomains.map((domain) => (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={!searching && domain === effectiveDomain}
                    className={!searching && domain === effectiveDomain ? "segmented-active" : ""}
                    key={domain}
                    onClick={() => {
                      setActiveDomain(domain);
                      setQuery("");
                      searchRef.current?.focus();
                    }}
                  >
                    {DOMAIN_LABEL[domain]}
                  </button>
                ))}
              </div>
            ) : null}
            <div className="task-select-search">
              <Search size={14} />
              <input
                ref={searchRef}
                type="text"
                value={query}
                placeholder="Search tasks…"
                aria-label="Search tasks"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && firstMatch) {
                    event.preventDefault();
                    pick(firstMatch);
                  }
                }}
              />
            </div>
          </div>
          <div className="task-select-list" role="listbox" aria-activedescendant={selected ?? undefined}>
            {groups.length === 0 ? (
              <p className="task-select-empty">No tasks match “{query.trim()}”.</p>
            ) : (
              groups.map((group) => (
                <div className="task-select-group" key={group.domain}>
                  {searching ? <p className="task-select-group-label">{DOMAIN_LABEL[group.domain]}</p> : null}
                  {group.tasks.map((task) => {
                    const active = task === value;
                    return (
                      <button
                        type="button"
                        role="option"
                        aria-selected={active}
                        id={active ? task : undefined}
                        key={task}
                        className={`task-select-option${active ? " task-select-option-active" : ""}`}
                        onClick={() => pick(task)}
                      >
                        <span className="task-select-icon">{taskIcon(task)}</span>
                        <span className="task-select-option-copy">
                          <strong>{formatDatasetTask(task)}</strong>
                          <small>{taskDescription(task)}</small>
                        </span>
                        {active ? <Check size={16} className="task-select-check" /> : null}
                      </button>
                    );
                  })}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
