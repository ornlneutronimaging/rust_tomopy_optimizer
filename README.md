# TomoPy Optimizer

Standalone GUI to tune TomoPy FBP reconstruction parameters on a
pre-processed CT checkpoint (the HDF5 written by `rust_ct_reconstruction`:
attenuation data with `/angles_rad` and `/center_of_rotation`). Uses
`tomopy.recon(algorithm='fbp')`, like the pipeline's white-beam CLI.

## Workflow

1. Open a checkpoint (command-line argument or the 📂 button).
2. Pick two slices on the projection view (red and cyan lines).
3. Adjust the parameters — the FBP **filter** (none, shepp, cosine, hann,
   hamming, ramlak, parzen, butterworth) in the open section; the center
   of rotation behind the password-locked **Advanced** section.
4. **▶ Evaluate** reconstructs the two selected slices through the real
   `tomopy` (from the `all_ct_reconstruction_development` pixi
   environment) and shows them side by side — FBP reconstructs each
   sinogram row independently, so this takes seconds. Every run lands in
   the **Run history** with slice thumbnails (hover to enlarge); its `use`
   buttons restore the parameters of a previous run.
5. **💾 Save** writes `tomopy_fbp_config` (JSON) into the checkpoint's
   `/metadata`; `rust_ct_reconstruction` restores it automatically and
   later TomoPy reconstructions use these parameters.

Defaults follow the pipeline: `fbp` algorithm, `hann` filter, center
seeded from the checkpoint.

## Running

```bash
./launch_tomopy_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
