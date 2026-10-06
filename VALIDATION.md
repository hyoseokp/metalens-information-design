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
- The scene prior is a frozen scene statistic, so its Hermitian positive
  semidefinite audit runs once per process and the audited prior is reused for
  every field and every step. The audit is a batched eigendecomposition over the
  whole frequency grid and is detached from the objective, so the reused prior
  leaves the objective and its gradient bit-identical to a per-call audit. Any
  change to the scene power spectrum or the scene color covariance is audited
  again.
- Two environment settings lower the peak memory of the scoring path without
  changing the model. `ENGINE2_VEC_SUBBATCH=1` propagates one dipole per pass
  instead of three (peak about 8 GB instead of 14 GB on CPU) and returns
  bit-identical numbers. `ENGINE2_WAVELENGTH_CHUNK=1` also builds the
  pixel-stack transfer and runs the detector stage one wavelength at a time
  (peak about 6 GB). That path evaluates the same elementwise expressions on
  single-wavelength tensors, and the complex64 transcendental kernels round
  differently on batched and unbatched inputs, so the pixel-stack transfer
  differs by at most 2.3e-7 relative and the per-field I_tar by about 2e-12
  relative. The published numbers were produced on the default path.

## Image formation vs metrics
- Each scene is rendered once by direct full-scene propagation. The raw
  model-electron frame, demosaicked image, reconstructed image, noise
  realization and calibration coefficients are saved together, and metrics are
  computed later from those saved arrays without rerunning the optical
  simulation.
- `image.py` is that pipeline. The scene is resized to the 198 x 198 active
  object grid and lifted to the nine design wavelengths by the
  covariance-weighted right inverse of the linear-sRGB target, so the truth
  is the scene as represented under the prior. Object points are placed by
  conserved-k-parallel Snell inversion of the 1 um photosite centres and
  propagated through the full-Jones forward. A width map that is mirror
  symmetric to 1e-7 is rendered on one quadrant and completed by reflection,
  and a D4-symmetric map on one octant. Asymmetric maps are rendered directly.
- Reconstruction is bilinear RGGB demosaic followed by the spatially varying
  Wiener filter over an 8 x 8 bank of exact field PSFs (regularization 0.03,
  sensor-side placement, no flip) and one analytic 3 x 3 colour matrix per
  design, the inverse of the decoder DC matrix times the camera DC matrix on
  a 64 x 64 centre region. PSNR, CIEDE2000 and S-CIELAB are computed on
  clipped linear RGB.
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
