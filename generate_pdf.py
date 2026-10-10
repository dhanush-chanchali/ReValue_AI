#!/usr/bin/env python3
"""
Convert ReValue AI Speaker Cards into a luxury printable PDF document using Chrome Headless.
"""
import os
import subprocess

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ReValue AI — Speaker Cue Cards & Viva Defense Guide</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  @page {
    size: A4 portrait;
    margin: 1.4cm 1.4cm 1.4cm 1.4cm;
  }
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
    background: #ffffff;
    line-height: 1.5;
    font-size: 10.5pt;
  }

  /* HEADER & BANNER */
  .header-card {
    border: 2px solid #0f172a;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    background: #f8fafc;
  }

  .badge-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }

  .badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .badge-gold {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #f59e0b;
  }

  .badge-emerald {
    background: #d1fae5;
    color: #065f46;
    border: 1px solid #10b981;
  }

  .badge-dark {
    background: #0f172a;
    color: #ffffff;
  }

  h1 {
    font-size: 20pt;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
  }

  .meta-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px 16px;
    margin-top: 10px;
    font-size: 9pt;
    color: #475569;
  }

  .meta-grid strong {
    color: #0f172a;
  }

  /* TIMELINE TABLE */
  .section-title {
    font-size: 13pt;
    font-weight: 800;
    color: #0f172a;
    margin: 18px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 5px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 22px;
    font-size: 9pt;
  }

  th, td {
    padding: 7px 10px;
    border: 1px solid #cbd5e1;
    text-align: left;
  }

  th {
    background: #0f172a;
    color: #ffffff;
    font-weight: 700;
  }

  tr:nth-child(even) {
    background: #f8fafc;
  }

  /* SPEAKER CARDS */
  .speaker-card {
    border: 1.5px solid #cbd5e1;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 16px;
    background: #ffffff;
    page-break-inside: avoid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }

  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1.5px solid #f1f5f9;
  }

  .card-title {
    font-size: 11pt;
    font-weight: 800;
    color: #0f172a;
  }

  .card-timing {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5pt;
    font-weight: 700;
    background: #eff6ff;
    color: #1d4ed8;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid #bfdbfe;
  }

  .card-objective {
    font-size: 8.5pt;
    color: #64748b;
    margin-bottom: 8px;
  }

  .script-box {
    background: #f8fafc;
    border-left: 3.5px solid #0f172a;
    padding: 10px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 9.5pt;
    color: #1e293b;
    line-height: 1.55;
    margin-bottom: 8px;
  }

  .script-label {
    font-size: 8pt;
    font-weight: 800;
    text-transform: uppercase;
    color: #0f172a;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    display: block;
  }

  .cues-row {
    display: flex;
    gap: 8px;
    background: #f1f5f9;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 8.5pt;
    color: #334155;
    margin-bottom: 8px;
  }

  .cues-row strong {
    color: #0f172a;
  }

  .defense-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3.5px solid #d97706;
    padding: 8px 11px;
    border-radius: 0 6px 6px 0;
    font-size: 8.5pt;
    color: #92400e;
  }

  .defense-box strong {
    color: #78350f;
  }

  /* CHEAT SHEET */
  .viva-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
    page-break-inside: avoid;
  }

  .viva-q {
    font-weight: 800;
    font-size: 10pt;
    color: #0f172a;
    margin-bottom: 5px;
  }

  .viva-a {
    font-size: 9pt;
    color: #334155;
    line-height: 1.55;
  }

  .formula-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }

  .page-break {
    page-break-after: always;
  }
</style>
</head>
<body>

  <!-- HEADER -->
  <div class="header-card">
    <div class="badge-row">
      <span class="badge badge-dark">Production ML Platform</span>
      <span class="badge badge-emerald">CO-1 to CO-6 Aligned</span>
      <span class="badge badge-gold">Champion R² = 0.9877</span>
    </div>
    <h1>ReValue AI — Speaker Cue Cards & Viva Defense Guide</h1>
    <p style="font-size: 9.5pt; color: #475569; margin-top: 2px;">
      Dual-Engine Regression, Unsupervised Fraud Guardrails & MLOps Drift Telemetry
    </p>
    <div class="meta-grid">
      <div><strong>Presenter:</strong> Dhanush Chanchali</div>
      <div><strong>Target Pacing:</strong> 10 Mins Deck + 5 Mins Viva Defense</div>
      <div><strong>Live App:</strong> revalueai-xhrrkbmbuexrswr3ermwnf.streamlit.app</div>
      <div><strong>Slide Deck:</strong> ReValue_AI_Presentation.pptx (10 Slides)</div>
    </div>
  </div>

  <!-- TIMELINE TABLE -->
  <div class="section-title">⏱️ 10-Minute Presentation Master Timeline</div>
  <table>
    <thead>
      <tr>
        <th style="width: 8%;">Slide</th>
        <th style="width: 32%;">Slide Title</th>
        <th style="width: 25%;">Curriculum Topic</th>
        <th style="width: 17%;">Time Allotment</th>
        <th style="width: 18%;">Elapsed Time</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>#1</strong></td>
        <td>Title & Executive Overview</td>
        <td>System Intro & Scope</td>
        <td>0:45 min</td>
        <td>0:45 min</td>
      </tr>
      <tr>
        <td><strong>#2</strong></td>
        <td>Problem Statement & Motivation</td>
        <td>Marketplace Asymmetry</td>
        <td>1:00 min</td>
        <td>1:45 min</td>
      </tr>
      <tr>
        <td><strong>#3</strong></td>
        <td>System Architecture & Zero Skew</td>
        <td>CO-1: ML Lifecycle</td>
        <td>1:15 min</td>
        <td>3:00 min</td>
      </tr>
      <tr>
        <td><strong>#4</strong></td>
        <td>11 Algorithms Implementation Directory</td>
        <td>Algorithm Directory Mapping</td>
        <td>1:00 min</td>
        <td>4:00 min</td>
      </tr>
      <tr>
        <td><strong>#5</strong></td>
        <td>Supervised: Linear vs Tree Ensembles</td>
        <td>CO-2 & CO-3: Regression</td>
        <td>1:30 min</td>
        <td>5:30 min</td>
      </tr>
      <tr>
        <td><strong>#6</strong></td>
        <td>Empirical Benchmark & Hypothesis Testing</td>
        <td>CO-5: Evaluation & Stats</td>
        <td>1:30 min</td>
        <td>7:00 min</td>
      </tr>
      <tr>
        <td><strong>#7</strong></td>
        <td>Unsupervised Market Tiers & Fraud Guard</td>
        <td>CO-4: K-Means & iForest</td>
        <td>1:15 min</td>
        <td>8:15 min</td>
      </tr>
      <tr>
        <td><strong>#8</strong></td>
        <td>FastAPI Serving & KS Drift Telemetry</td>
        <td>CO-6: Production MLOps</td>
        <td>1:00 min</td>
        <td>9:15 min</td>
      </tr>
      <tr>
        <td><strong>#9</strong></td>
        <td>Live Interactive Studio & Test Suite</td>
        <td>Deployment & Validation</td>
        <td>0:30 min</td>
        <td>9:45 min</td>
      </tr>
      <tr>
        <td><strong>#10</strong></td>
        <td>Conclusion & Academic Defense</td>
        <td>Summary & Wrap-Up</td>
        <td>0:15 min</td>
        <td>10:00 min</td>
      </tr>
    </tbody>
  </table>

  <div class="page-break"></div>

  <!-- CARD 1 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 1 — Slide 1: Title & Executive Overview</div>
      <div class="card-timing">⏱️ 0:00 - 0:45 (45s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Grab attention, project confidence, and establish that this is a production-grade ML engineering system.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "Good morning respected professors and evaluators. Today I am proud to present <strong>ReValue AI</strong> — an end-to-end, production-grade Machine Learning platform designed to eliminate information asymmetry in the secondhand electronics marketplace.<br><br>
      Unlike typical student projects that terminate at an exploratory Jupyter Notebook, ReValue AI is engineered as a <strong>full-lifecycle software system</strong>. It benchmarks regularized linear baselines against advanced gradient boosted ensembles, incorporates unsupervised K-Means clustering and Isolation Forest fraud guardrails, and runs live in production with real-time Kolmogorov-Smirnov drift monitoring, systematically covering <strong>Course Outcomes CO-1 through CO-6</strong>."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Stand tall, make direct eye contact with the lead evaluator, point to the Champion Model badge: <em>R² = 0.9877</em>.
    </div>
    <div class="defense-box">
      <strong>🛡️ Anticipated Examiner Q:</strong> <em>"Why build a valuation platform specifically for electronics?"</em><br>
      <strong>Defense:</strong> Electronics depreciation is non-linear and suffers from severe information asymmetry. Unlike cars which depreciate smoothly by year, devices suffer cliff degradation from battery cycle aging and physical wear that consumers cannot reliably price without algorithmic benchmarks.
    </div>
  </div>

  <!-- CARD 2 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 2 — Slide 2: Problem Statement & Motivation</div>
      <div class="card-timing">⏱️ 0:45 - 1:45 (60s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Prove the problem is mathematically non-trivial and requires machine learning rather than static rules.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "The secondhand smartphone and laptop trade-in industry is expanding rapidly, yet participants face three critical roadblocks:<br><br>
      First, <strong>information asymmetry</strong>: buyers and sellers rely on fragmented classified ads or subjective guesswork, lacking an objective fair market baseline.<br><br>
      Second, <strong>discontinuous degradation boundaries</strong>: consumer electronics depreciation is strictly non-linear. When an iPhone or MacBook battery health drops below 80%, its trade-in price doesn't drop gradually — it falls off a cliff due to battery service replacement costs.<br><br>
      Third, <strong>marketplace fraud</strong>: portals are constantly attacked by bogus or manipulated trade-in quotes — for example, someone attempting to submit an iPhone 16 Pro Max for ₹10,000. ReValue AI addresses each of these with statistical machine learning."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Gesture across the three pillars on the slide: <em>Asymmetry ➔ Non-Linear Degradation ➔ Marketplace Fraud</em>.
    </div>
  </div>

  <!-- CARD 3 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 3 — Slide 3: System Architecture & Zero Skew (CO-1)</div>
      <div class="card-timing">⏱️ 1:45 - 3:00 (75s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Demonstrate architectural maturity by explaining Training-Serving Skew and how your pipeline eliminated it.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "Under <strong>Course Outcome CO-1</strong>, we architected the system to resolve the most common and catastrophic failure mode in production ML: <strong>Training-Serving Skew</strong>.<br><br>
      Training-serving skew happens when preprocessing transformations — like imputation medians, one-hot category mappings, or standard deviation scalers — are implemented separately in training and production serving.<br><br>
      In ReValue AI, we enforce mathematical zero skew. All transformations are encapsulated within a unified Scikit-Learn <code>ColumnTransformer</code> and serialized alongside the estimator as <code>best_model_pipeline.joblib</code>. During live serving, both our asynchronous <strong>FastAPI backend</strong> and our <strong>Streamlit dashboard</strong> feed raw JSON payloads directly into this single atomic pipeline. What was trained is identically what serves."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Trace the flow on screen: <em>Ingestion ➔ Atomic ColumnTransformer ➔ 5-Fold Benchmarking ➔ FastAPI ➔ Streamlit & MLOps</em>.
    </div>
    <div class="defense-box">
      <strong>🛡️ Anticipated Examiner Q:</strong> <em>"What split protocol did you use for validation?"</em><br>
      <strong>Defense:</strong> We implemented a strict 70/15/15 three-way partition: 70% Train ($N=1,750$), 15% Validation ($N=375$), and 15% held-out Test ($N=375$). The held-out test set was isolated and never touched during hyperparameter tuning or threshold selection.
    </div>
  </div>

  <div class="page-break"></div>

  <!-- CARD 4 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 4 — Slide 4: 11 Algorithms Directory</div>
      <div class="card-timing">⏱️ 3:00 - 4:00 (60s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Highlight comprehensive technical depth across supervised, unsupervised, and inferential methods.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "Rather than relying blindly on a single algorithm, ReValue AI systematically implements and benchmarks <strong>11 algorithms</strong> across the machine learning lifecycle:<br><br>
      For <strong>Supervised Regression</strong>, we train Ordinary Least Squares as an unregularized baseline, Ridge L2 to stabilize collinear hardware attributes, Lasso L1 for coordinate-descent feature sparsity, Random Forest for variance reduction via bagging, and <strong>HistGradientBoosting</strong> as our Production Champion.<br><br>
      For <strong>Unsupervised Learning</strong>, we deploy <strong>K-Means++</strong> for latent market tier clustering, <strong>PCA</strong> for orthogonal dimensionality reduction capturing 67.7% of variance, and <strong>Isolation Forest</strong> for live fraud screening.<br><br>
      Finally, for <strong>Inferential Telemetry</strong>, we implement <strong>Paired Student's t-Tests</strong> for statistical superiority verification and two-sample <strong>Kolmogorov-Smirnov tests</strong> for covariate drift detection."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Highlight the Champion row in green on the table: <em>HistGradientBoostingRegressor</em>.
    </div>
  </div>

  <!-- CARD 5 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 5 — Slide 5: Supervised Learning — Linear vs Trees (CO-2, CO-3)</div>
      <div class="card-timing">⏱️ 4:00 - 5:30 (90s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Defend the mathematical formulas for Ridge and Lasso, and explain why trees beat linear models on tabular data.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "Covering <strong>CO-2 and CO-3</strong>, this slide illustrates the fundamental mathematical split in our benchmarking.<br><br>
      In linear models, hardware attributes like RAM, internal storage, and launch price exhibit severe multicollinearity. Under standard OLS, $(X^T X)^{-1}$ becomes near-singular, causing coefficient variance to explode. <strong>Ridge Regression</strong> solves this by adding an L2 penalty $\\alpha \\|\\mathbf{w}\\|_2^2$, making $(X^T X + \\alpha I)$ strictly invertible.<br><br>
      However, linear models assume monotonic hyperplanes. They cannot capture the sharp cliff where a laptop battery drops below 80% and loses 30% of its value overnight. <strong>HistGradientBoosting</strong> solves this naturally. It splits feature space using orthogonal, axis-aligned decision thresholds. By binning continuous features into 256 discrete integer bins, split finding speeds up by 10×, while sequential shrinkage with $\\eta = 0.08$ prevents overfitting."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Contrast the two cards: Left side (<em>L2 Penalty & Invertibility</em>) vs Right side (<em>Histogram Binning & Axis-Aligned Cuts</em>).
    </div>
    <div class="defense-box">
      <strong>🛡️ Anticipated Examiner Q:</strong> <em>"Why is StandardScaler mandatory for Ridge and Lasso but optional for Trees?"</em><br>
      <strong>Defense:</strong> In linear regularization, the penalty $\\alpha \\sum w_j^2$ penalizes weights uniformly. If storage is in GB and damage is binary $\{0,1\}$, unscaled weights would distort the penalty. Standardizing to $\\mu=0, \\sigma=1$ ensures isotropic penalization. Decision trees, however, only care about feature rank order $(\\mathbf{x}_j \\le \\theta)$, making them naturally scale-invariant.
    </div>
  </div>

  <!-- CARD 6 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 6 — Slide 6: Empirical Benchmark & Hypothesis Testing (CO-5)</div>
      <div class="card-timing">⏱️ 5:30 - 7:00 (90s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Present the empirical test leaderboard and cite the Paired Student's t-test statistical proof.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "Moving to <strong>CO-5: Model Evaluation and Verification</strong>. We evaluated all models on the held-out test set of 375 devices.<br><br>
      The empirical results are decisive:
      Linear baselines (OLS, Ridge, Lasso) clustered around a Test MAE of <strong>₹6,798 ($80.93)</strong> with an R² of <strong>0.9048</strong>.<br>
      Random Forest reduced the error to <strong>₹3,460 ($41.19)</strong> with an R² of <strong>0.9738</strong>.<br>
      Our Champion, <strong>HistGradientBoosting</strong>, achieved an outstanding Test MAE of <strong>₹2,305 ($27.44)</strong> and an R² of <strong>0.9877</strong>, beating the linear baseline by <strong>66.1% (a saving of ₹4,493 per device)</strong>.<br><br>
      Crucially, in engineering we do not declare a winner based on mean numbers alone. We conducted a <strong>Paired Student's t-test</strong> on absolute prediction errors. The test yielded a t-statistic of <strong>16.82</strong> and a p-value of <strong>2.8 × 10⁻⁴⁹ ≪ 0.01</strong>. We reject the null hypothesis and confirm that HistGradientBoosting's superiority is mathematically proven beyond random chance."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Point directly to the <em>R² = 0.9877</em> metric and the Paired t-test callout box.
    </div>
  </div>

  <div class="page-break"></div>

  <!-- CARD 7 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 7 — Slide 7: Unsupervised Market Tiers & Fraud Guard (CO-4)</div>
      <div class="card-timing">⏱️ 7:00 - 8:15 (75s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Explain the unsupervised algorithms and demonstrate how Isolation Forest protects the live pricing engine.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "Under <strong>CO-4</strong>, ReValue AI integrates unsupervised learning to provide automated market intelligence and risk mitigation.<br><br>
      First, <strong>K-Means++ clustering</strong> partitions inventory into <strong>4 latent market tiers</strong>: <em>Budget Entry, Daily Workhorse, Depreciated Flagship, and Luxury</em>. We utilized K-Means++ $D(x)^2$ distance seeding to prevent poor local optima and confirmed $K=4$ using Elbow inertia and Silhouette peaks. To visualize this in high dimensions, <strong>PCA</strong> reduces 9 specifications to 2 orthogonal axes capturing <strong>67.7% of total variance</strong>.<br><br>
      Second, <strong>Isolation Forest</strong> acts as our real-time fraud guardrail. Unlike density clustering, Isolation Forest isolates anomalies using random tree cuts. Because fraudulent quotes possess extreme attribute combinations, they require far fewer splits to isolate. Configured with a contamination rate of <strong>0.04</strong>, it intercepts bogus quotes before they reach the pricing model."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Explain the conceptual difference: <em>Profiling normal clusters (K-Means)</em> vs <em>Isolating outliers directly (Isolation Forest)</em>.
    </div>
  </div>

  <!-- CARD 8 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 8 — Slide 8: FastAPI Serving & KS Drift Telemetry (CO-6)</div>
      <div class="card-timing">⏱️ 8:15 - 9:15 (60s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Prove production-grade ML engineering, sub-10ms response SLAs, and automated KS drift retraining.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "In <strong>CO-6</strong>, we operationalize the system into production.<br><br>
      Our backend is an asynchronous <strong>FastAPI REST microservice</strong> executing sub-10ms predictions with strict Pydantic payload validation and generating empirical 90% confidence intervals.<br><br>
      In production, data distribution shifts over time as new device generations launch. To detect this, our monitoring service executes continuous <strong>Two-Sample Kolmogorov-Smirnov (KS) hypothesis tests</strong>, calculating the supremum distance between live request distributions and reference training CDFs.<br><br>
      If $p < 0.05$ on two or more features, the system flags covariate drift and triggers our <strong>Champion-Challenger retraining pipeline</strong>. A newly trained challenger model is automatically evaluated on held-out test data, and will only replace the Champion if it beats its MAE benchmark."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Cite the KS statistic formula: <span class="formula-code">D = sup_x |F_prod(x) - F_ref(x)|</span>.
    </div>
  </div>

  <!-- CARD 9 & 10 -->
  <div class="speaker-card">
    <div class="card-top">
      <div class="card-title">🎴 Card 9 & 10 — Slide 9 & 10: Live Platform & Academic Wrap-Up</div>
      <div class="card-timing">⏱️ 9:15 - 10:00 (45s)</div>
    </div>
    <div class="card-objective"><strong>🎯 Objective:</strong> Showcase the live cloud application, confirm passing unit tests, and transition smoothly into examiner Q&A.</div>
    <div class="script-box">
      <span class="script-label">🎙️ Verbatim Talking Script:</span>
      "The platform is fully verified and live on the cloud at <code>revalueai-xhrrkbmbuexrswr3ermwnf.streamlit.app</code>. It features 5 interactive tabs: step-by-step Smartphone and Laptop appraisal wizards with 60fps battery sliders, ML residual scatter plots, an MLOps command centre, and an academic syllabus inspector with textbook-grade KaTeX vector math.<br><br>
      Every commit is verified against our automated PyTest suite where <strong>all 7 unit and integration tests pass cleanly</strong>.<br><br>
      In conclusion, ReValue AI bridges the gap between academic Machine Learning theory and production software engineering. It achieves high empirical accuracy with R² = 0.9877, defends trade-ins with unsupervised fraud guardrails, and maintains production health with automated KS drift telemetry.<br><br>
      Thank you for your time and attention. I am now open to your questions and viva evaluation."
    </div>
    <div class="cues-row">
      <strong>🎬 Stage Cue:</strong> Switch screen to live Streamlit Cloud browser tab for a 20-second appraisal calculation demo!
    </div>
  </div>

  <div class="page-break"></div>

  <!-- MASTER VIVA DEFENSE CHEAT SHEET -->
  <div class="section-title">🛡️ Master Viva Defense Cheat Sheet (Anticipated Tough Examiner Questions)</div>

  <div class="viva-item">
    <div class="viva-q">1. "Why didn't you use a Deep Neural Network (MLP) for this project?"</div>
    <div class="viva-a">
      <strong>Your Defense:</strong> On tabular data with mixed feature types (continuous, ordinal, and categorical), tree ensembles consistently outperform deep neural networks (as demonstrated in research by Grinsztajn et al., 2022). Tabular features have unaligned coordinate hyperplanes and sharp discrete thresholds (such as battery degradation below 80%), which decision trees split naturally. Furthermore, HistGradientBoosting trains in seconds, requires no GPU infrastructure, and has significantly higher sample efficiency on 2,500 records than a deep neural network.
    </div>
  </div>

  <div class="viva-item">
    <div class="viva-q">2. "How do you detect data drift if you don't receive true resale prices in real-time production?"</div>
    <div class="viva-a">
      <strong>Your Defense:</strong> That is precisely why we chose Covariate Drift Detection via the Two-Sample Kolmogorov-Smirnov test. Label drift requires ground-truth sale prices, which take weeks to settle in a marketplace. Covariate drift, however, monitors the input feature distributions $P(X)$. By testing the live input CDF against the training CDF, we detect if the market has shifted to newer models or older battery cycles immediately, before incorrect predictions harm the business.
    </div>
  </div>

  <div class="viva-item">
    <div class="viva-q">3. "Is R² = 0.9877 too high? Could your Champion model be overfitting?"</div>
    <div class="viva-a">
      <strong>Your Defense:</strong> No, because that R² score is evaluated strictly on the untouched 15% held-out test set ($N=375$). Furthermore, our 5-fold cross-validation MAE was $29.82 and our test set MAE was $27.44 — the metrics are completely consistent with zero generalization gap. In consumer electronics, original launch price, age, battery health, and cosmetic condition account for almost all variance in resale price, which is why a well-tuned gradient boosting model with shrinkage ($\eta=0.08$) achieves near-deterministic accuracy.
    </div>
  </div>

  <div class="viva-item">
    <div class="viva-q">4. "What is the difference between Bagging and Boosting in your project?"</div>
    <div class="viva-a">
      <strong>Your Defense:</strong> Random Forest uses Bagging: it trains 150 deep, low-bias trees independently in parallel using bootstrap samples and $\sqrt{p}$ random feature subsets, reducing variance by averaging decorrelated trees. HistGradientBoosting uses Boosting: it trains shallow trees sequentially in series, where each tree fits the negative gradient (pseudo-residual) of the previous ensemble, systematically reducing bias with shrinkage.
    </div>
  </div>

  <div class="viva-item">
    <div class="viva-q">5. "Why did you use K-Means++ instead of standard Lloyd's random initialization?"</div>
    <div class="viva-a">
      <strong>Your Defense:</strong> Standard Lloyd's algorithm chooses $K$ initial centroids uniformly at random, which frequently traps clustering in poor local minima when two initial seeds fall within the same true cluster. K-Means++ initializes the first centroid randomly, but selects each subsequent centroid with probability proportional to the squared distance $D(x)^2$ from the nearest existing centroid. This ensures maximum spatial separation and is mathematically proven to achieve an $O(\log K)$ approximation bound.
    </div>
  </div>

</body>
</html>
"""

def generate_pdf():
    html_file = "/tmp/revalue_ai_speaker_cards.html"
    pdf_file = "/Users/dhanuu/Desktop/ML PROJECT/ReValue_AI/ReValue_AI_Speaker_Cards.pdf"

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_file}",
        html_file
    ]

    print(f"Running Chrome headless to generate {pdf_file}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 1000:
        print(f"✅ Success! PDF created at: {pdf_file} ({os.path.getsize(pdf_file)} bytes)")
    else:
        print(f"❌ Failed to generate PDF: {res.stderr}")

if __name__ == "__main__":
    generate_pdf()
