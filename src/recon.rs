//! TomoPy FBP parameters, the test reconstruction of the two selected
//! slices (through `tomopy.recon` in the `all_ct_reconstruction_development`
//! pixi environment), and saving the parameters back into the checkpoint
//! HDF5.

use ct_reconstruction::combine::LoadedStack;
use ct_reconstruction::crop::{read_npy, write_npy};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has tomopy installed.
pub const TOMOPY_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// The name of the config saved into `/metadata`, matching the main
/// application's `<algorithm key>_config` convention.
pub const CONFIG_NAME: &str = "tomopy_fbp_config";

/// The FBP filters offered by tomopy.
pub const FILTERS: [&str; 8] = [
    "none",
    "shepp",
    "cosine",
    "hann",
    "hamming",
    "ramlak",
    "parzen",
    "butterworth",
];

/// The TomoPy FBP parameters used by the pipeline's `tomopy_recon` call,
/// with its defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TomopyParams {
    /// FBP filter (pipeline default: hann); index into `FILTERS`.
    pub filter: usize,
    /// Center of rotation in pixels (column of the sinogram).
    pub center: f64,
}

impl Default for TomopyParams {
    fn default() -> Self {
        Self {
            filter: 3, // hann
            center: 0.0,
        }
    }
}

impl TomopyParams {
    /// Defaults seeded from the stack: the saved `tomopy_fbp_config` when
    /// the checkpoint carries one, otherwise the stack's center of rotation
    /// (falling back to the middle of the detector).
    pub fn from_stack(stack: &LoadedStack) -> Self {
        if let Some((_, json)) = stack
            .metadata
            .iter()
            .find(|(name, _)| name == CONFIG_NAME)
            && let Some(params) = Self::from_json(json)
        {
            return params;
        }
        let mut params = Self::default();
        if let Some(first) = stack.sample.first() {
            params.center = stack
                .center_of_rotation
                .unwrap_or(first.width as f64 / 2.0);
        }
        params
    }

    /// The saved form matches the pipeline's `tomopy_recon` call.
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "algorithm": "fbp",
            "filter_name": FILTERS[self.filter.min(FILTERS.len() - 1)],
            "center": self.center,
        })
        .to_string()
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let doc: serde_json::Value = serde_json::from_str(text).ok()?;
        let mut params = Self::default();
        if let Some(name) = doc.get("filter_name").and_then(|v| v.as_str())
            && let Some(i) = FILTERS.iter().position(|f| *f == name)
        {
            params.filter = i;
        }
        if let Some(v) = doc.get("center").and_then(|v| v.as_f64()) {
            params.center = v;
        }
        Some(params)
    }

    pub fn describe(&self) -> String {
        format!(
            "fbp, {} filter, center {:.2}",
            FILTERS[self.filter.min(FILTERS.len() - 1)],
            self.center
        )
    }
}

const TOMOPY_SCRIPT: &str = r#"
import json
import sys

import numpy as np
from tomopy import recon as tomopy_recon

sino_file, spec_file, out_file = sys.argv[1:4]
with open(spec_file) as f:
    spec = json.load(f)
proj = np.load(sino_file)  # (n_angles, n_selected_slices, width)
angles = np.array(spec["angles_rad"], dtype=np.float32)
p = spec["params"]
result = tomopy_recon(
    tomo=proj,
    theta=angles,
    center=p["center"],
    sinogram_order=False,
    algorithm=p["algorithm"],
    filter_name=p["filter_name"],
)
np.save(out_file, np.array(result, dtype=np.float32))
"#;

/// One test reconstruction of the two slices on a background thread;
/// resolves to the two reconstructed slices
/// `(height, width, values0, values1, seconds)`.
pub struct ReconJob {
    rx: Receiver<Result<(usize, usize, Vec<f32>, Vec<f32>, f64), String>>,
}

impl ReconJob {
    pub fn start(
        stack: Arc<LoadedStack>,
        top_slice: usize,
        bottom_slice: usize,
        params: TomopyParams,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params).map(
                |(h, w, top, bottom)| (h, w, top, bottom, started.elapsed().as_secs_f64()),
            );
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<(usize, usize, Vec<f32>, Vec<f32>, f64), String>> {
        self.rx.try_recv().ok()
    }
}

fn scratch_dir(stack: &LoadedStack) -> Result<PathBuf, String> {
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(".tomopy_optimizer_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_ok() {
        return Ok(dir);
    }
    let dir = std::env::temp_dir().join(format!("tomopy_optimizer_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    Ok(dir)
}

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: TomopyParams,
) -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
    let first = stack
        .sample
        .first()
        .ok_or("no projections in the stack")?;
    let (w, h, n) = (first.width, first.height, stack.sample.len());
    let angles: Vec<f64> = stack
        .sample
        .iter()
        .map(|p| p.angle_deg.map(|a| a.to_radians()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("some projections carry no angle — the reconstruction needs all of them")?;
    let top_slice = top_slice.min(h - 1);
    let bottom_slice = bottom_slice.min(h - 1);

    let dir = scratch_dir(stack)?;
    let sino_npy = dir.join("sino.npy");
    let spec_file = dir.join("spec.json");
    let out_npy = dir.join("recon.npy");
    let script = dir.join("tomopy_run.py");
    let cleanup = || {
        for f in [&sino_npy, &spec_file, &out_npy, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
    };
    let run = || -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
        // FBP reconstructs each sinogram row independently, so only the two
        // selected rows are shipped.
        let mut volume = Vec::with_capacity(n * 2 * w);
        for p in &stack.sample {
            for row in [top_slice, bottom_slice] {
                volume.extend_from_slice(&p.mean[row * w..(row + 1) * w]);
            }
        }
        write_npy(&sino_npy, &[n, 2, w], volume.chunks(2 * w))?;
        let spec = serde_json::json!({
            "angles_rad": angles,
            "params": serde_json::from_str::<serde_json::Value>(&params.to_json()).expect("params json"),
        });
        std::fs::write(&spec_file, spec.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, TOMOPY_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let output = std::process::Command::new(TOMOPY_PYTHON)
            .arg(&script)
            .arg(&sino_npy)
            .arg(&spec_file)
            .arg(&out_npy)
            .output()
            .map_err(|e| format!("cannot launch {TOMOPY_PYTHON}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "tomopy failed ({}): {}",
                output.status,
                tail.join(" | ")
            ));
        }
        let (shape, values) = read_npy(&out_npy)?;
        let [count, rh, rw] = shape.as_slice() else {
            return Err(format!("tomopy returned shape {shape:?}, expected 3-D"));
        };
        if *count != 2 {
            return Err(format!("tomopy returned {count} slices, expected 2"));
        }
        let (top, bottom) = values.split_at(rh * rw);
        Ok((*rh, *rw, top.to_vec(), bottom.to_vec()))
    };
    let result = run();
    cleanup();
    result
}

/// Write (or replace) the `tomopy_fbp_config` JSON in the checkpoint's
/// `/metadata` group, where the main application reads it back.
pub fn save_params(path: &Path, params: &TomopyParams) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open_rw(path)
        .map_err(|e| format!("cannot open {} for writing: {e}", path.display()))?;
    let metadata = match file.group("metadata") {
        Ok(group) => group,
        Err(_) => file
            .create_group("metadata")
            .map_err(|e| format!("create metadata group: {e}"))?,
    };
    if metadata.dataset(CONFIG_NAME).is_ok() {
        metadata
            .unlink(CONFIG_NAME)
            .map_err(|e| format!("replace {CONFIG_NAME}: {e}"))?;
    }
    let value: VarLenUnicode = params.to_json().parse().unwrap_or_default();
    metadata
        .new_dataset::<VarLenUnicode>()
        .create(CONFIG_NAME)
        .and_then(|ds| ds.write_scalar(&value))
        .map_err(|e| format!("write {CONFIG_NAME}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_json_roundtrip() {
        let params = TomopyParams {
            filter: 7,
            center: 251.75,
        };
        let back = TomopyParams::from_json(&params.to_json()).unwrap();
        assert_eq!(back, params);
        let doc: serde_json::Value = serde_json::from_str(&params.to_json()).unwrap();
        assert_eq!(doc["algorithm"], "fbp");
        assert_eq!(doc["filter_name"], "butterworth");
        assert!(TomopyParams::from_json("nope").is_none());
    }
}
