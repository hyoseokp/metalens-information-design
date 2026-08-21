# Reproducibility and provenance

This note records the numerical conventions and provenance controls used to
produce the released designs and evaluation numbers. It is implementation
detail that supports reproduction and is kept out of the paper.

## Exposure and calibration
- One model-electron calibration scalar is fixed on the corrected 540 nm
  hyperbolic anchor (hyp540) and reused unchanged for hyp600 and every evaluated
  design, so all comparisons are at a common exposure.
- Color calibration is a per-design analytic white-balance and color matrix
  built from a predeclared reference-camera DC mapping, with no raster-scene
  input and no per-scene least-squares fit. It is fixed before any test scene is
  seen and applied unchanged across noise realizations.

## Evaluation
- Every design is scored from its stored 720x720 full width map through one
  common evaluator. The evaluator does not refit a reduced profile and does not
  apply annular averaging. The mirror-quadrant parameterization is used only
  during optimization; the stored expanded map is the state that is rescored.
- The objective is the quadrature-weighted mean over the 25 first-quadrant
  fields. Only the calibrated common-exposure transfer is used for optimization;
  row-wise and global fixed-signal transfers are shape-only diagnostics.
- Optimization uses 300 projected-Adam steps with four quadrature-weighted field
  samples per step. Checkpoints are selected by deterministic evaluation over all
  25 fields every ten steps, followed by an identical five-step full-field
  refinement for each arm. Each run records its field-sampling generator seed and
  sampled-index hash in the checkpoint. The terminal optimizer state is retained
  for resumption and is not used in place of the selected checkpoint.

## Image formation vs metrics
- Each scene is rendered once by direct full-scene propagation. The raw
  model-electron frame, demosaicked image, reconstructed image, noise
  realization and calibration coefficients are saved together, and metrics are
  computed later from those saved arrays without rerunning the optical
  simulation.
- PSF and OTF banks enter only the Wiener reconstruction decoder and never form
  the raw image. Reconstruction coefficients, PSNR and color-error values do not
  enter optical gradients or checkpoint selection.

## Records
- Each saved bundle records its exposure role (common-exposure vs
  equal-brightness diagnostic) so the two conventions are not mixed.
- The full-Jones RCWA response table is tabulated at Fourier order (7,7) on a
  256^2 raster with C4 completion; its scope is recorded as
  `direct_full_jones_rcwa_c4_projected`. A separate Fourier-order and raster
  convergence ladder is not certified (see the paper's limitations note).
