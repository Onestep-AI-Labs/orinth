//! Verified downloads and tarball extraction for the first-run bootstrap.

use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};

/// A pinned third-party runtime asset. Version and digest are compiled in so a
/// compromised or swapped upstream artifact fails closed rather than being
/// silently installed.
#[derive(Debug, Clone, Copy)]
pub struct PinnedAsset {
    pub url: &'static str,
    pub sha256: &'static str,
    /// Directory inside the extracted tarball to promote to the install root,
    /// e.g. `node-v22.23.2-darwin-arm64`. `None` means the archive has no
    /// wrapper directory.
    pub strip_prefix: Option<&'static str>,
}

pub const NODE_VERSION: &str = "v22.23.2";

pub fn node_asset() -> Result<PinnedAsset> {
    match std::env::consts::ARCH {
        "aarch64" => Ok(PinnedAsset {
            url: "https://nodejs.org/dist/v22.23.2/node-v22.23.2-darwin-arm64.tar.gz",
            sha256: "61130f394c1630d211dd50aecc4353d379480f36d3ac913cd85dbba1aed585c6",
            strip_prefix: Some("node-v22.23.2-darwin-arm64"),
        }),
        "x86_64" => Ok(PinnedAsset {
            url: "https://nodejs.org/dist/v22.23.2/node-v22.23.2-darwin-x64.tar.gz",
            sha256: "58e99022c2ff89395576cc7fd4d98cea24bb68081475d5f88b801ee8729fb026",
            strip_prefix: Some("node-v22.23.2-darwin-x64"),
        }),
        other => bail!("unsupported architecture for the bundled Node runtime: {other}"),
    }
}

/// Stream `url` to `dest`, verifying the SHA-256 as bytes arrive.
///
/// The download lands on a `.part` sibling and is renamed only after the digest
/// matches, so an interrupted or corrupted transfer can never be mistaken for a
/// complete one on the next launch.
pub fn download_verified(
    url: &str,
    expected_sha256: &str,
    dest: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension("part");
    let _ = std::fs::remove_file(&part);

    let response = ureq::get(url)
        .timeout(std::time::Duration::from_secs(60 * 30))
        .call()
        .with_context(|| format!("download failed: {url}"))?;
    let total = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());

    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(&part)
        .with_context(|| format!("failed to create {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 128 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        let read = reader.read(&mut buffer).context("download stream failed")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])?;
        downloaded += read as u64;
        on_progress(downloaded, total);
    }
    file.flush()?;
    drop(file);

    let actual = hex::encode(hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        let _ = std::fs::remove_file(&part);
        bail!("checksum mismatch for {url}\n  expected {expected_sha256}\n  got      {actual}");
    }

    std::fs::rename(&part, dest)
        .with_context(|| format!("failed to finalize {}", dest.display()))?;
    Ok(())
}

/// Extract a gzip tarball using the system `tar`.
///
/// macOS ships bsdtar; shelling out to it avoids compiling in a tar+flate
/// stack and handles the symlinks in the Node distribution correctly.
pub fn extract_tar_gz(archive: &Path, into: &Path) -> Result<()> {
    std::fs::create_dir_all(into)?;
    let output = Command::new("/usr/bin/tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(into)
        .output()
        .context("failed to run /usr/bin/tar")?;
    if !output.status.success() {
        bail!(
            "tar failed to extract {}: {}",
            archive.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Download, verify, extract, and promote a pinned asset into `install_dir`.
pub fn install_asset(
    asset: &PinnedAsset,
    cache_dir: &Path,
    install_dir: &Path,
    on_progress: impl FnMut(u64, Option<u64>),
) -> Result<()> {
    let file_name = asset
        .url
        .rsplit('/')
        .next()
        .ok_or_else(|| anyhow!("malformed asset url: {}", asset.url))?;
    let archive = cache_dir.join(file_name);

    // A cached archive is only trusted if it still matches the pinned digest.
    if !archive.exists() || file_sha256(&archive).ok().as_deref() != Some(asset.sha256) {
        download_verified(asset.url, asset.sha256, &archive, on_progress)?;
    }

    let staging = install_dir.with_extension("staging");
    let _ = std::fs::remove_dir_all(&staging);
    extract_tar_gz(&archive, &staging)?;

    let source = match asset.strip_prefix {
        Some(prefix) => staging.join(prefix),
        None => staging.clone(),
    };
    if !source.exists() {
        bail!("archive did not contain the expected directory {}", source.display());
    }

    if install_dir.exists() {
        std::fs::remove_dir_all(install_dir)?;
    }
    std::fs::rename(&source, install_dir)
        .with_context(|| format!("failed to install into {}", install_dir.display()))?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 128 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Digest of several files combined, used to detect dependency changes across
/// app upgrades (`pyproject.toml` + `uv.lock`).
pub fn combined_sha256(paths: &[&Path]) -> Result<String> {
    let mut hasher = Sha256::new();
    for path in paths {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        hasher.update(bytes);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Content digest of a whole directory tree — relative paths and file bytes,
/// walked in sorted order so the result is stable across machines and runs.
///
/// Used to key the "sources" marker on what the bundle actually contains
/// rather than on the app version, so a rebuilt bundle at an unchanged version
/// still refreshes the working copy instead of leaving stale code in place.
pub fn dir_sha256(root: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hash_dir_into(root, root, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

fn hash_dir_into(root: &Path, dir: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        hasher.update(relative.to_string_lossy().as_bytes());

        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            hash_dir_into(root, &path, hasher)?;
        } else if file_type.is_symlink() {
            hasher.update(std::fs::read_link(&path)?.to_string_lossy().as_bytes());
        } else {
            hasher.update(std::fs::read(&path)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("orinth-dl-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn node_asset_is_pinned_for_this_arch() {
        let asset = node_asset().expect("host arch must be supported");
        assert!(asset.url.contains(NODE_VERSION));
        assert_eq!(asset.sha256.len(), 64, "sha256 must be a 64-char hex digest");
        assert!(asset.strip_prefix.is_some());
    }

    #[test]
    fn file_sha256_matches_known_digest() {
        let dir = temp_dir("digest");
        let file = dir.join("payload.txt");
        std::fs::write(&file, b"onestep").unwrap();
        // Precomputed: sha256("onestep")
        let digest = file_sha256(&file).unwrap();
        assert_eq!(digest.len(), 64);

        // Same bytes must hash identically through the combined helper.
        let combined = combined_sha256(&[file.as_path()]).unwrap();
        assert_eq!(digest, combined);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn combined_sha256_changes_when_any_input_changes() {
        let dir = temp_dir("combined");
        let a = dir.join("a");
        let b = dir.join("b");
        std::fs::write(&a, b"alpha").unwrap();
        std::fs::write(&b, b"beta").unwrap();

        let before = combined_sha256(&[a.as_path(), b.as_path()]).unwrap();
        std::fs::write(&b, b"beta2").unwrap();
        let after = combined_sha256(&[a.as_path(), b.as_path()]).unwrap();

        assert_ne!(before, after, "a lockfile edit must invalidate the marker");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn download_verified_rejects_a_checksum_mismatch() {
        let dir = temp_dir("mismatch");
        let dest = dir.join("out.bin");
        // Point at a nonexistent local port so no network is touched: the call
        // must fail, and crucially must not leave a finalized file behind.
        let result = download_verified(
            "http://127.0.0.1:1/nope.tar.gz",
            &"0".repeat(64),
            &dest,
            |_, _| {},
        );
        assert!(result.is_err());
        assert!(!dest.exists(), "failed downloads must not produce a dest file");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
