#!/usr/bin/env node
/**
 * Assemble everything the `.app` bundle ships.
 *
 * The bundle stays small on purpose: it carries source and a pinned `uv`, and
 * the 2.7 GB ML runtime is provisioned into the user's data root on first
 * launch (see specs/phase-18-macos-desktop-app.md). What lands here is:
 *
 *   resources/backend/            FastAPI source + pyproject.toml + uv.lock
 *   resources/frontend.tar.gz     Next standalone server
 *   resources/sample_data/        starter datasets (trash classification + NLP)
 *   resources/bin/uv              pinned uv binary for the host arch
 */

import { createHash } from "node:crypto";
import { execFileSync, execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const desktopDir = path.resolve(here, "..");
const repoRoot = path.resolve(desktopDir, "..");
const frontendDir = path.join(repoRoot, "frontend");
const backendDir = path.join(repoRoot, "backend");
const resourcesDir = path.join(desktopDir, "src-tauri", "resources");
const cacheDir = path.join(desktopDir, ".cache");

// Must match PLACEHOLDER_ORIGIN in src-tauri/src/origin.rs. `.invalid` is
// reserved by RFC 2606 and can never resolve, so a missed substitution fails
// loudly instead of quietly reaching some real host.
const BACKEND_ORIGIN_PLACEHOLDER = "http://onestep-backend.invalid";

// Keep in sync with the pinned versions table in
// specs/phase-18-macos-desktop-app.md.
const UV_VERSION = "0.12.1";
const UV_ASSETS = {
  arm64: {
    name: "uv-aarch64-apple-darwin",
    sha256: "77d2906988e8074fd43f2f329ec452ebbf9b0c257ba1c66451c71de70a6baf42"
  },
  x64: {
    name: "uv-x86_64-apple-darwin",
    sha256: "69d9f9a00337f25a50dcb13882052da08b8469bac11091c98c5694c3c6721467"
  }
};

// Source files that must never reach a user's machine or would break the
// relocated runtime (a build-machine venv, compiled caches, the test suite).
const BACKEND_EXCLUDES = new Set([
  "__pycache__",
  ".venv",
  ".pytest_cache",
  ".ruff_cache",
  ".mypy_cache",
  ".env",
  "openapi.json"
]);

function log(message) {
  process.stdout.write(`[prepare-resources] ${message}\n`);
}

function fail(message) {
  process.stderr.write(`[prepare-resources] ERROR: ${message}\n`);
  process.exit(1);
}

function run(command, args, options = {}) {
  execFileSync(command, args, { stdio: "inherit", ...options });
}

function resetDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
}

function copyFiltered(src, dest) {
  fs.cpSync(src, dest, {
    recursive: true,
    dereference: false,
    filter: (source) => !BACKEND_EXCLUDES.has(path.basename(source))
  });
}

function sha256(file) {
  return createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

/**
 * Build the Next.js app in standalone mode and stage it as a tarball.
 *
 * It ships as a tarball rather than a directory because pnpm's standalone
 * output is a symlink farm: `node_modules/next` points into
 * `node_modules/.pnpm/next@…/node_modules/next`, and Node resolves a package's
 * dependencies from its *real* path. Flatten those links and `next` can no
 * longer see `styled-jsx`, so the server dies on boot with MODULE_NOT_FOUND.
 *
 * Both copy steps between here and a user's disk destroy that structure:
 * `fs.cpSync` rewrites relative symlinks to absolute paths (which point at the
 * build machine), and the Tauri bundler dereferences them into real
 * directories. `tar` is the one tool in the chain that reproduces the farm
 * byte-for-byte, and a single opaque file is immune to whatever the bundler
 * does to it. The bootstrap unpacks it with `tar` on first launch.
 */
function buildFrontend() {
  log("Building the Next.js standalone server…");
  run("pnpm", ["build"], {
    cwd: frontendDir,
    env: {
      ...process.env,
      // Gated so a plain `pnpm build` / `make build` keeps its existing output.
      DESKTOP_BUILD: "1",
      // Next resolves rewrite destinations at build time and freezes them into
      // routes-manifest.json, so the runtime env var cannot steer them. Bake an
      // unroutable placeholder that the app substitutes once it knows the
      // backend's port (see src-tauri/src/origin.rs). Keep this string in sync
      // with PLACEHOLDER_ORIGIN there.
      BACKEND_PROXY_ORIGIN: BACKEND_ORIGIN_PLACEHOLDER
    }
  });

  const standalone = path.join(frontendDir, ".next", "standalone");
  if (!fs.existsSync(path.join(standalone, "server.js"))) {
    fail(
      `Next did not emit ${path.join(standalone, "server.js")}. ` +
        "Confirm next.config.mjs sets output: 'standalone' when DESKTOP_BUILD=1."
    );
  }

  const staging = path.join(cacheDir, "frontend-staging");
  resetDir(staging);

  // tar-to-tar rather than cp: this is the copy that must not touch symlinks.
  execSync(
    `/usr/bin/tar -cf - -C ${JSON.stringify(standalone)} . | ` +
      `/usr/bin/tar -xf - -C ${JSON.stringify(staging)}`,
    { stdio: "inherit", shell: "/bin/bash" }
  );

  // Static assets and public/ are deliberately not traced into the standalone
  // tree; neither contains symlinks, so a plain copy is fine.
  fs.cpSync(path.join(frontendDir, ".next", "static"), path.join(staging, ".next", "static"), {
    recursive: true
  });
  const publicDir = path.join(frontendDir, "public");
  if (fs.existsSync(publicDir)) {
    fs.cpSync(publicDir, path.join(staging, "public"), {
      recursive: true,
      filter: (source) => path.basename(source) !== ".DS_Store"
    });
  }

  const linkCount = Number(
    execSync(`/usr/bin/find ${JSON.stringify(staging)} -type l | /usr/bin/wc -l`, {
      shell: "/bin/bash"
    })
      .toString()
      .trim()
  );
  if (linkCount === 0) {
    fail(
      "the staged frontend contains no symlinks — pnpm's node_modules layout was " +
        "flattened and the server would fail with MODULE_NOT_FOUND on boot."
    );
  }

  const archive = path.join(resourcesDir, "frontend.tar.gz");
  fs.rmSync(archive, { force: true });
  fs.rmSync(path.join(resourcesDir, "frontend"), { recursive: true, force: true });
  run("/usr/bin/tar", ["-czf", archive, "-C", staging, "."]);
  fs.rmSync(staging, { recursive: true, force: true });

  const megabytes = (fs.statSync(archive).size / 1024 / 1024).toFixed(1);
  log(
    `Frontend staged at ${path.relative(repoRoot, archive)} ` +
      `(${megabytes} MB, ${linkCount} symlinks preserved)`
  );
}

/** Copy the backend source the app will `uv sync` and run. */
function stageBackend() {
  log("Staging backend source…");
  const dest = path.join(resourcesDir, "backend");
  resetDir(dest);

  for (const entry of ["app", "migrations"]) {
    copyFiltered(path.join(backendDir, entry), path.join(dest, entry));
  }
  for (const entry of ["pyproject.toml", "uv.lock", "alembic.ini"]) {
    const source = path.join(backendDir, entry);
    if (!fs.existsSync(source)) {
      fail(`missing required backend file: ${entry}`);
    }
    fs.copyFileSync(source, path.join(dest, entry));
  }

  // `uv sync` builds the project itself via hatchling, which needs a README
  // only if pyproject references one; it does not here. Nothing else to add.
  log(`Backend staged at ${path.relative(repoRoot, dest)}`);
}

/**
 * Copy the tracked starter datasets so a fresh install has usable data.
 *
 * Without these the app opens with an empty catalog: the backend's sample
 * locations resolve `SAMPLE_DATA_DIR` and silently skip anything missing, so
 * the Trash classification and NLP datasets simply never appear.
 */
function stageSampleData() {
  const source = path.join(repoRoot, "sample_data");
  if (!fs.existsSync(source)) {
    fail(`missing ${path.relative(repoRoot, source)}; the app would ship with an empty catalog`);
  }

  const dest = path.join(resourcesDir, "sample_data");
  resetDir(dest);
  fs.cpSync(source, dest, {
    recursive: true,
    filter: (entry) => path.basename(entry) !== ".DS_Store"
  });

  const manifests = fs
    .readdirSync(dest, { recursive: true })
    .filter((entry) => String(entry).endsWith("manifest.json"));
  if (manifests.length === 0) {
    fail("sample_data staged without any manifest.json; the datasets would not be discovered");
  }
  log(`Sample data staged (${manifests.length} datasets)`);
}

/** Download one pinned uv build, verify it, and return the extracted binary. */
async function fetchUv(arch) {
  const asset = UV_ASSETS[arch];
  if (!asset) {
    fail(`no pinned uv build for arch "${arch}" (expected arm64 or x64)`);
  }

  fs.mkdirSync(cacheDir, { recursive: true });
  const archive = path.join(cacheDir, `${asset.name}-${UV_VERSION}.tar.gz`);

  if (!fs.existsSync(archive) || sha256(archive) !== asset.sha256) {
    const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset.name}.tar.gz`;
    log(`Downloading uv ${UV_VERSION} for ${arch}…`);
    const response = await fetch(url);
    if (!response.ok) {
      fail(`uv download failed: ${response.status} ${response.statusText}`);
    }
    const partial = `${archive}.part`;
    fs.writeFileSync(partial, Buffer.from(await response.arrayBuffer()));
    // Verify before promoting, so a truncated download is never cached as good.
    const actual = sha256(partial);
    if (actual !== asset.sha256) {
      fs.rmSync(partial, { force: true });
      fail(`uv checksum mismatch\n  expected ${asset.sha256}\n  got      ${actual}`);
    }
    fs.renameSync(partial, archive);
  } else {
    log(`Using cached uv ${UV_VERSION} for ${arch}`);
  }

  const staging = path.join(cacheDir, `uv-extract-${arch}`);
  resetDir(staging);
  run("/usr/bin/tar", ["-xzf", archive, "-C", staging]);

  const binary = path.join(staging, asset.name, "uv");
  if (!fs.existsSync(binary)) {
    fail(`uv archive did not contain ${path.relative(staging, binary)}`);
  }
  return binary;
}

/**
 * Vendor `uv` as a universal binary.
 *
 * The app ships as one universal `.app` so a single download runs on both
 * Apple silicon and Intel. `uv` is a separate executable the app invokes, so
 * it needs the same treatment: an arm64-only `uv` inside a universal bundle
 * leaves Intel users with a first launch that dies as soon as provisioning
 * starts. `lipo` welds the two published builds into one binary that runs
 * either way.
 */
async function stageUv() {
  const arm64 = await fetchUv("arm64");
  const x64 = await fetchUv("x64");

  const binDir = path.join(resourcesDir, "bin");
  resetDir(binDir);
  const target = path.join(binDir, "uv");
  run("/usr/bin/lipo", ["-create", arm64, x64, "-output", target]);
  fs.chmodSync(target, 0o755);

  const archs = execSync(`/usr/bin/lipo -archs ${JSON.stringify(target)}`).toString().trim();
  if (!archs.includes("arm64") || !archs.includes("x86_64")) {
    fail(`uv was not built universal (got "${archs}")`);
  }

  for (const arch of ["arm64", "x64"]) {
    fs.rmSync(path.join(cacheDir, `uv-extract-${arch}`), { recursive: true, force: true });
  }
  log(`uv ${UV_VERSION} staged universal (${archs})`);
}

async function main() {
  if (process.platform !== "darwin") {
    fail("this phase targets macOS only");
  }
  fs.mkdirSync(resourcesDir, { recursive: true });

  stageBackend();
  stageSampleData();
  buildFrontend();
  await stageUv();

  log("Resources ready.");
}

main().catch((error) => fail(error?.stack ?? String(error)));
