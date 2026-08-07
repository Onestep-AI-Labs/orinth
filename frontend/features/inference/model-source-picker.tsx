"use client";

import { useEffect, useState } from "react";
import { Cpu, Download, FolderOpen, Play, Sparkles } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { compactNumber, formatBytes } from "@/features/platform/utils";
import { Badge, Button, InlineSpinner, Select } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import type { ModelInfo } from "@/types/api";

export type StartTarget = { model_id?: string; model_path?: string };
type Source = "trained" | "local" | "hub";

/**
 * Shared, tabbed GGUF model picker — Trained (registered) · Local (a folder on
 * this machine) · Hugging Face (recommended-for-hardware + search) — used by the
 * chat rail and the inference serve view so model selection is one consistent
 * surface. Every source resolves to a `ServingStartRequest` via `onServe`.
 */
export function ModelSourcePicker({
  registeredModels,
  onServe,
  disabled = false,
  initialSource
}: {
  registeredModels: ModelInfo[];
  onServe: (target: StartTarget) => void;
  disabled?: boolean;
  initialSource?: Source;
}) {
  const [source, setSource] = useState<Source>(
    initialSource ?? (registeredModels.length ? "trained" : "hub")
  );
  return (
    <div className="model-source-picker">
      <div className="serve-source-toggle">
        {(["trained", "local", "hub"] as Source[]).map((key) => (
          <button
            key={key}
            type="button"
            className={`serve-source-chip${source === key ? " serve-source-chip-active" : ""}`}
            onClick={() => setSource(key)}
          >
            {key === "trained" ? "Trained" : key === "local" ? "Local" : "Hugging Face"}
          </button>
        ))}
      </div>
      {source === "trained" && <TrainedTab models={registeredModels} onServe={onServe} disabled={disabled} />}
      {source === "local" && <LocalTab onServe={onServe} disabled={disabled} />}
      {source === "hub" && <HubTab onServe={onServe} disabled={disabled} />}
    </div>
  );
}

function TrainedTab({
  models,
  onServe,
  disabled
}: {
  models: ModelInfo[];
  onServe: (t: StartTarget) => void;
  disabled: boolean;
}) {
  if (models.length === 0) {
    return (
      <p className="field-hint model-source-empty">
        No trained GGUF models yet. Export a fine-tuned model to GGUF on its detail page, or use
        Local / Hugging Face.
      </p>
    );
  }
  return (
    <div className="model-source-list">
      {models.map((model) => (
        <button
          key={model.id}
          type="button"
          className="model-source-row"
          onClick={() => onServe({ model_id: model.id })}
          disabled={disabled}
        >
          <Play size={14} />
          <span className="model-source-name" title={model.name}>{model.name}</span>
          {model.size_bytes != null ? <span className="model-source-size">{formatBytes(model.size_bytes)}</span> : null}
        </button>
      ))}
    </div>
  );
}

function LocalTab({ onServe, disabled }: { onServe: (t: StartTarget) => void; disabled: boolean }) {
  const [path, setPath] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [scan, setScan] = useState<Awaited<ReturnType<typeof api.scanServingModels>> | null>(null);

  const scanMutation = useMutation({
    mutationFn: (target: string) => api.scanServingModels(target),
    onSuccess: (result, target) => {
      setScan(result);
      if (!target.toLowerCase().endsWith(".gguf")) {
        void api.saveServingConfig(result.root).catch(() => undefined);
      }
    }
  });
  const { mutate: runScan } = scanMutation;
  const pick = useMutation({
    mutationFn: (kind: "folder" | "file") => api.pickServingPath(kind),
    onSuccess: (result) => {
      if (result.canceled) return;
      if (result.unavailable || !result.path) {
        toast.error(result.message ?? "Native dialog unavailable — paste a path instead.");
        return;
      }
      setPath(result.path);
    }
  });

  useEffect(() => {
    let active = true;
    api
      .servingConfig()
      .then((config) => {
        if (active && config.models_dir) setPath(config.models_dir);
      })
      .catch(() => undefined)
      .finally(() => {
        if (active) setLoaded(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!loaded) return;
    const target = path.trim();
    if (!target) {
      setScan(null);
      return;
    }
    const handle = setTimeout(() => runScan(target), 500);
    return () => clearTimeout(handle);
  }, [path, loaded, runScan]);

  return (
    <div className="model-source-local">
      <input
        value={path}
        onChange={(event) => setPath(event.target.value)}
        placeholder="/path/to/models  or  /path/to/model.gguf"
      />
      <div className="model-source-local-actions">
        <Button variant="secondary" size="sm" onClick={() => pick.mutate("folder")} disabled={pick.isPending}>
          <FolderOpen size={14} /> Folder…
        </Button>
        <Button variant="secondary" size="sm" onClick={() => pick.mutate("file")} disabled={pick.isPending}>
          <FolderOpen size={14} /> .gguf…
        </Button>
      </div>
      {scanMutation.isPending && <InlineSpinner label="Scanning" />}
      {scan && scan.entries.length === 0 && !scanMutation.isPending ? (
        <p className="field-hint">No GGUF files found under this path.</p>
      ) : null}
      {scan && scan.entries.length > 0 && (
        <div className="model-source-list">
          {scan.entries.map((entry) => (
            <button
              key={entry.path}
              type="button"
              className="model-source-row"
              onClick={() => onServe(entry.model_id ? { model_id: entry.model_id } : { model_path: entry.path })}
              disabled={disabled}
            >
              <Play size={14} />
              <span className="model-source-name" title={entry.path}>{entry.name}</span>
              <span className="model-source-size">{formatBytes(entry.size_bytes)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function HubTab({ onServe, disabled }: { onServe: (t: StartTarget) => void; disabled: boolean }) {
  const [queryText, setQueryText] = useState("");
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [format, setFormat] = useState<"gguf" | "mlx">("gguf");
  const [openRepo, setOpenRepo] = useState<string | null>(null);

  const recommended = useQuery({
    queryKey: ["hub-recommended"],
    queryFn: () => api.recommendedHubModels()
  });
  const search = useQuery({
    queryKey: ["hub-search", submitted, format],
    queryFn: () => api.searchHubModels(submitted ?? "", format),
    enabled: submitted !== null
  });
  const files = useQuery({
    queryKey: ["hub-files", openRepo, format],
    queryFn: () => api.hubModelFiles(openRepo as string, format),
    enabled: Boolean(openRepo)
  });
  const download = useMutation({
    mutationFn: ({ repoId, filename }: { repoId: string; filename: string }) =>
      api.downloadHubModel(repoId, filename),
    onSuccess: (result) => {
      if (result.servable) onServe({ model_path: result.path });
      else toast.success(result.message ?? "Downloaded");
    },
    onError: (error: Error) => toast.error(error.message)
  });

  return (
    <div className="model-source-hub">
      <div className="model-source-hub-search">
        <input
          value={queryText}
          onChange={(event) => setQueryText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") setSubmitted(queryText.trim());
          }}
          placeholder="Search Hugging Face…"
        />
        <Select value={format} onChange={(event) => setFormat(event.target.value as "gguf" | "mlx")}>
          <option value="gguf">GGUF</option>
          <option value="mlx">MLX</option>
        </Select>
      </div>
      {format === "mlx" && (
        <p className="field-hint">MLX is downloadable for reference; only GGUF is servable here.</p>
      )}

      {submitted === null ? (
        <RecommendedList
          recommended={recommended.data}
          loading={recommended.isFetching}
          openRepo={openRepo}
          onToggle={(repo) => setOpenRepo((cur) => (cur === repo ? null : repo))}
          files={openRepo ? files.data?.files ?? [] : []}
          filesLoading={files.isFetching}
          onDownload={(repoId, filename) => download.mutate({ repoId, filename })}
          downloading={download.isPending ? download.variables?.filename : undefined}
          disabled={disabled}
        />
      ) : search.isFetching ? (
        <InlineSpinner label="Searching" />
      ) : search.data?.error ? (
        <p className="error-text">{search.data.error}</p>
      ) : (
        <div className="model-source-list">
          {(search.data?.results ?? []).map((repo) => (
            <HubRepoRow
              key={repo.repo_id}
              repoId={repo.repo_id}
              meta={`${compactNumber(repo.downloads)} ↓`}
              open={openRepo === repo.repo_id}
              onToggle={() => setOpenRepo((cur) => (cur === repo.repo_id ? null : repo.repo_id))}
              files={openRepo === repo.repo_id ? files.data?.files ?? [] : []}
              filesLoading={openRepo === repo.repo_id && files.isFetching}
              onDownload={(filename) => download.mutate({ repoId: repo.repo_id, filename })}
              downloading={download.isPending ? download.variables?.filename : undefined}
              disabled={disabled}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function RecommendedList({
  recommended,
  loading,
  openRepo,
  onToggle,
  files,
  filesLoading,
  onDownload,
  downloading,
  disabled
}: {
  recommended: Awaited<ReturnType<typeof api.recommendedHubModels>> | undefined;
  loading: boolean;
  openRepo: string | null;
  onToggle: (repo: string) => void;
  files: Awaited<ReturnType<typeof api.hubModelFiles>>["files"];
  filesLoading: boolean;
  onDownload: (repoId: string, filename: string) => void;
  downloading?: string;
  disabled: boolean;
}) {
  if (loading) return <InlineSpinner label="Checking your hardware" />;
  if (!recommended) return null;
  return (
    <div className="model-source-recommended">
      <p className="model-source-reco-head">
        <Sparkles size={13} /> Recommended
        <span className="model-source-reco-hw">
          <Cpu size={12} /> {recommended.device.toUpperCase()}
          {recommended.total_memory_gb ? ` · ${recommended.total_memory_gb} GB` : ""}
        </span>
      </p>
      <div className="model-source-list">
        {recommended.results.map((rec) => (
          <HubRepoRow
            key={rec.repo_id}
            repoId={rec.title}
            meta={`~${rec.approx_size_gb} GB`}
            fits={rec.fits}
            open={openRepo === rec.repo_id}
            onToggle={() => onToggle(rec.repo_id)}
            files={openRepo === rec.repo_id ? files : []}
            filesLoading={openRepo === rec.repo_id && filesLoading}
            onDownload={(filename) => onDownload(rec.repo_id, filename)}
            downloading={downloading}
            disabled={disabled}
          />
        ))}
      </div>
    </div>
  );
}

function HubRepoRow({
  repoId,
  meta,
  fits,
  open,
  onToggle,
  files,
  filesLoading,
  onDownload,
  downloading,
  disabled
}: {
  repoId: string;
  meta: string;
  fits?: boolean;
  open: boolean;
  onToggle: () => void;
  files: Awaited<ReturnType<typeof api.hubModelFiles>>["files"];
  filesLoading: boolean;
  onDownload: (filename: string) => void;
  downloading?: string;
  disabled: boolean;
}) {
  return (
    <div className="model-source-repo">
      <button type="button" className="model-source-row" onClick={onToggle}>
        <span className="model-source-name" title={repoId}>{repoId}</span>
        {fits === false ? <Badge tone="warn">tight</Badge> : fits === true ? <Badge tone="ok">fits</Badge> : null}
        <span className="model-source-size">{meta}</span>
      </button>
      {open && (
        <div className="model-source-files">
          {filesLoading ? (
            <InlineSpinner label="Files" />
          ) : files.length === 0 ? (
            <p className="field-hint">No files.</p>
          ) : (
            files.map((file) => (
              <button
                key={file.filename}
                type="button"
                className="model-source-row"
                disabled={disabled || downloading === file.filename}
                onClick={() => onDownload(file.filename)}
              >
                <Download size={13} />
                <span className="model-source-name" title={file.filename}>{file.quantization ?? file.filename}</span>
                <span className="model-source-size">
                  {downloading === file.filename ? "Downloading…" : formatBytes(file.size_bytes)}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
