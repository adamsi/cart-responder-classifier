"""CAR-T Responder Classifier demo UI (Gradio).

Loads the trained attention-MIL model plus the fitted PCA, lets a user upload
a patient's single-cell count matrix (or pick a bundled example), and shows
the predicted probability of response together with which cell types the
model paid attention to.
"""
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd

from src.infer import load_weights, predict as mil_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
EX = os.path.join(HERE, "examples")
os.chdir(HERE)  # so relative paths and Gradio's file cache work from any launch directory

# ---------------------------------------------------------------- artifacts
pca = pickle.load(open(os.path.join(ART, "pca.pkl"), "rb"))
GENES = open(os.path.join(ART, "genes.txt")).read().split()
WEIGHTS = load_weights(os.path.join(ART, "model.npz"))
DEMO = os.path.exists(os.path.join(ART, "DEMO_WEIGHTS"))
DIAGRAM = ""
_svg = os.path.join(HERE, "diagram", "architecture.svg")
if os.path.exists(_svg):
    DIAGRAM = "<div class='diagram'>" + open(_svg).read() + "</div>"
EXAMPLES = sorted(
    os.path.join(EX, f) for f in os.listdir(EX) if f.endswith(".csv")
) if os.path.isdir(EX) else []


# ---------------------------------------------------------------- inference
def predict(file):
    if file is None:
        raise gr.Error("Upload a CSV or choose an example patient first.")
    path = file if isinstance(file, str) else file.name
    header = pd.read_csv(path, nrows=0, index_col=0).columns
    dtypes = {c: (str if c == "cell_type" else np.float32) for c in header}     # float32 halves memory vs the default
    df = pd.read_csv(path, index_col=0, dtype=dtypes)

    cell_type = df.pop("cell_type") if "cell_type" in df else None
    totals = df.pop("total_counts").values if "total_counts" in df else None

    present = [g for g in GENES if g in df.columns]
    missing = len(GENES) - len(present)
    if len(present) < 0.5 * len(GENES):
        raise gr.Error(f"Only {len(present)} of {len(GENES)} model genes found in the file. "
                       "Columns must be gene names from artifacts/genes.txt.")
    counts = df.reindex(columns=GENES, fill_value=0).to_numpy(dtype=np.float32)
    cell_ids, n_cells = df.index, len(df)
    del df
    if totals is None:
        totals = counts.sum(1)
    totals = np.clip(np.asarray(totals, dtype=np.float32), 1, None)

    x = np.log1p(counts / totals[:, None] * 1e4)
    p, att = mil_predict(WEIGHTS, pca.transform(x).astype(np.float32))

    # ---- probability card
    verdict = "Likely responder ✅" if p >= 0.5 else "Likely non-responder ⚠️"
    color = "#15803d" if p >= 0.5 else "#b91c1c"
    banner = ("<div class='banner'>Placeholder weights are loaded. Replace the artifacts folder "
              "with the Colab export to get real predictions.</div>") if DEMO else ""
    card = f"""
    {banner}
    <div class="card">
      <div class="label">Predicted probability of response</div>
      <div class="big" style="color:{color}">{p:.0%}</div>
      <div class="bar"><div class="fill" style="width:{p*100:.1f}%;background:{color}"></div></div>
      <div class="verdict" style="color:{color}">{verdict}</div>
      <div class="stats">
        <div><span>{n_cells:,}</span>cells analysed</div>
        <div><span>{len(present):,}</span>of {len(GENES):,} genes matched</div>
        <div><span>{missing}</span>genes filled with 0</div>
      </div>
    </div>"""

    # ---- attention by cell type
    if cell_type is not None:
        share = pd.Series(att).groupby(cell_type.values).sum().sort_values(ascending=False)
        freq = cell_type.value_counts(normalize=True).reindex(share.index)
        plot_df = pd.DataFrame({"cell type": share.index,
                                "attention share (%)": (share.values * 100).round(2),
                                "cell frequency (%)": (freq.values * 100).round(2)})
    else:
        plot_df = pd.DataFrame({"cell type": ["(no cell_type column in file)"],
                                "attention share (%)": [100.0], "cell frequency (%)": [100.0]})

    # ---- top attended cells
    top = np.argsort(att)[::-1][:10]
    top_df = pd.DataFrame({
        "cell": cell_ids[top],
        "cell type": cell_type.values[top] if cell_type is not None else "?",
        "attention weight": np.round(att[top], 5),
    })
    return card, plot_df, top_df


# ---------------------------------------------------------------- UI
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Kalam:wght@400;700&display=swap');
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; font-size: 17px; }
#hero { padding: 40px 0 18px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 26px; }
#hero .tag { display:inline-block; font-size: 13px; letter-spacing: .06em; text-transform: uppercase;
             color: var(--primary-600); background: var(--primary-50); border-radius: 6px; padding: 4px 10px; margin-bottom: 14px; }
#hero h1 { font-size: clamp(30px, 4.5vw, 44px); font-weight: 650; letter-spacing: -0.02em; margin: 0 0 10px; line-height: 1.15; }
#hero p  { font-size: clamp(16px, 2vw, 19px); color: var(--body-text-color-subdued); margin: 0; max-width: 720px; line-height: 1.55; }
.section { font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--body-text-color-subdued); margin: 4px 0 2px; }
.card { text-align:center; padding: 30px 20px 22px; border-radius: 14px; background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary); }
.card .label { font-size: 15px; color: var(--body-text-color-subdued); }
.card .big   { font-size: clamp(56px, 8vw, 80px); font-weight: 650; letter-spacing: -0.03em; line-height: 1.05; margin: 6px 0 10px; }
.card .bar   { height: 10px; border-radius: 99px; background: var(--border-color-primary); overflow:hidden; margin: 0 auto 14px; max-width: 420px; }
.card .fill  { height: 100%; border-radius: 99px; transition: width .5s ease; }
.card .verdict { font-size: clamp(19px, 2.6vw, 23px); font-weight: 600; margin-bottom: 18px; }
.card .stats { display:flex; justify-content:center; gap: 28px; flex-wrap: wrap; border-top: 1px solid var(--border-color-primary); padding-top: 16px; }
.card .stats div { font-size: 13px; color: var(--body-text-color-subdued); }
.card .stats span { display:block; font-size: 20px; font-weight: 600; color: var(--body-text-color); }
.card .idle { font-size: 17px; color: var(--body-text-color-subdued); padding: 34px 0; }
.banner { background: var(--color-accent-soft, #f5f3ff); color: var(--primary-700); border: 1px solid var(--primary-200);
          border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; font-size: 14px; }
.diagram { padding: 6px 0 18px; overflow-x: auto; }
.diagram svg { min-width: 640px; }
.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 4px; }
.step { padding: 18px; border-radius: 12px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.step .n { font-size: 13px; font-weight: 600; color: var(--primary-600); letter-spacing: .06em; }
.step b { display:block; margin: 6px 0 6px; font-size: 17px; font-weight: 600; }
.step span { font-size: 15px; line-height: 1.5; color: var(--body-text-color-subdued); }
.note { font-size: 15px; line-height: 1.6; margin-top: 16px; }
.note.muted { font-size: 14px; color: var(--body-text-color-subdued); }
footer { display:none !important; }
#foot { color: var(--body-text-color-subdued); font-size: 13px; padding: 26px 0 10px; border-top: 1px solid var(--border-color-primary); margin-top: 26px; }
"""

HERO = """
<div id="hero">
  <div class="tag">🔬 Research demo</div>
  <h1>🧬 CAR-T Responder Classifier</h1>
  <p>Predicts whether a lymphoma patient will respond to CD19 CAR-T therapy from a pre-treatment
     blood single-cell sample, and shows which immune cells drove the prediction.</p>
</div>"""

IDLE = "<div class='card'><div class='idle'>🤖 Choose an example patient or upload a file to see a prediction.</div></div>"

HOW = """
<div class="steps">
  <div class="step"><div class="n">STEP 1</div><b>🩸 Blood sample</b><span>Thousands of immune cells, each with about 20,000 gene activity values, measured before treatment.</span></div>
  <div class="step"><div class="n">STEP 2</div><b>📉 Compress</b><span>Keep 2,000 informative genes, normalise, and reduce every cell to 50 numbers with PCA.</span></div>
  <div class="step"><div class="n">STEP 3</div><b>👀 Attend</b><span>A small network scores every cell. Softmax turns the scores into weights that sum to one.</span></div>
  <div class="step"><div class="n">STEP 4</div><b>🎯 Predict</b><span>The weighted average of the cells becomes one patient vector, then a probability of response.</span></div>
</div>
<p class="note">
  Trained on 55 lymphoma patients from the public dataset
  <a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267097" target="_blank">GSE267097</a>
  (Weizmann Institute, Amit lab, <i>Cancer Research</i> 2025). There is one label per patient and none per cell,
  so the model learns on its own which cells matter. This is attention-based multiple-instance learning.
  Performance is measured with patient-level cross-validation and reported as AUC with a bootstrap confidence interval.
</p>
<p class="note muted">⚠️ For research use only. Not a medical device and not intended for clinical decisions.</p>"""

FORMAT = """
<p class="note" style="margin-top:0">
  CSV with one row per cell. The first column is the cell id and the remaining columns are gene names with raw counts,
  matching <code>artifacts/genes.txt</code>. Two optional columns: <code>cell_type</code> enables the attention chart,
  and <code>total_counts</code> gives exact per-cell normalisation.
</p>"""

with gr.Blocks(title="CAR-T Responder Classifier") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=300):
            gr.HTML("<div class='section'>🩸 1. Patient</div>")
            file_in = gr.File(label="Upload single-cell CSV", file_types=[".csv"], height=110)
            if EXAMPLES:
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=file_in, label="👇 Example patients",
                            example_labels=[os.path.basename(e).replace(".csv", "").replace("_", " ") for e in EXAMPLES])
            run = gr.Button("🚀 Predict response", variant="primary", size="lg")
            with gr.Accordion("📄 Expected file format", open=False):
                gr.HTML(FORMAT)
        with gr.Column(scale=7, min_width=300):
            gr.HTML("<div class='section'>🎯 2. Result</div>")
            card = gr.HTML(IDLE)
            plot = gr.BarPlot(x="cell type", y="attention share (%)", label="🔍 Attention by cell type", sort="-y", height=300)
            top = gr.Dataframe(label="🏅 Ten most influential cells", interactive=False, wrap=True)
    with gr.Accordion("🧠 How the model works", open=True):
        gr.HTML(DIAGRAM + HOW)
    gr.HTML("<div id='foot'>Attention-MIL on single-cell RNA-seq. Trained with PyTorch, served with numpy and Gradio. Data: GEO GSE267097. Build: numpy-v2.</div>")

    run.click(predict, inputs=file_in, outputs=[card, plot, top])
    file_in.change(predict, inputs=file_in, outputs=[card, plot, top])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")),
                allowed_paths=[EX], server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
