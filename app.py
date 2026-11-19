import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logistic.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# ---------------------------------------------------------------------------
# !! IMPORTANT: CATEGORY ENCODING !!
# Your pickle holds only the LogisticRegression (no encoder/pipeline), so it
# expects numbers. I assumed sklearn LabelEncoder-style codes, i.e. the
# category list sorted A-Z, then 0, 1, 2 ... If you encoded differently in
# your notebook, edit the lists below to match your training order exactly.
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Major_Category": ["Arts", "Business", "Engineering", "Humanities", "Science", "Social Science"],
    "Primary_Use_Case": ["Coding", "Homework", "Research", "Writing"],
    "Prompt_Engineering_Skill": ["Advanced", "Beginner", "Intermediate"],
    "Paid_Subscription": ["No", "Yes"],
    "Institutional_Policy": ["Allowed", "Banned", "Restricted"],
}
ENCODING = {col: {name: i for i, name in enumerate(sorted(opts))} for col, opts in CATEGORIES.items()}

# Numeric inputs: (label, min, max, step, default)
NUMERIC = {
    "Year_of_Study": ("Year of study", 1, 6, 1, 2),
    "Pre_Semester_GPA": ("Pre-semester GPA", 0, 4, 0.01, 3.0),
    "Weekly_GenAI_Hours": ("Weekly GenAI hours", 0, 60, 0.5, 8),
    "Traditional_Study_Hours": ("Traditional study hours / week", 0, 60, 0.5, 10),
    "Perceived_AI_Dependency": ("Perceived AI dependency (1-10)", 1, 10, 1, 5),
    "Skill_Retention_Score": ("Skill retention score (0-100)", 0, 100, 1, 60),
}

LABELS = {
    "Major_Category": "Major",
    "Primary_Use_Case": "Main way you use GenAI",
    "Prompt_Engineering_Skill": "Prompt-writing skill",
    "Paid_Subscription": "Paid subscription",
    "Institutional_Policy": "University AI policy",
}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenAI Impact Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0e1f; --ink:#eef1ff; --muted:#9aa4cf;
    --violet:#7c5cff; --aqua:#22d3ee; --pink:#ff5fa2;
    --high:#34e0a1; --medium:#ffc857; --low:#ff6b81;
    --glass:rgba(255,255,255,.06); --line:rgba(255,255,255,.14);
  }
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);
       font-family:Inter,system-ui,sans-serif;overflow-x:hidden}

  /* Aurora background */
  .aurora{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
  .aurora span{position:absolute;width:55vmax;height:55vmax;border-radius:50%;filter:blur(90px);opacity:.45;
               animation:drift 18s ease-in-out infinite alternate}
  .aurora span:nth-child(1){background:var(--violet);top:-20%;left:-15%}
  .aurora span:nth-child(2){background:var(--aqua);bottom:-25%;right:-15%;animation-delay:-6s;opacity:.32}
  .aurora span:nth-child(3){background:var(--pink);top:30%;left:45%;width:35vmax;height:35vmax;animation-delay:-11s;opacity:.22}
  @keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(8vmax,-6vmax) scale(1.2)}}

  main{position:relative;z-index:1;max-width:1040px;margin:0 auto;padding:48px 20px 64px}
  h1{font-family:Sora,sans-serif;font-weight:800;font-size:clamp(2rem,5vw,3.2rem);margin:0 0 10px;
     background:linear-gradient(90deg,var(--aqua),var(--violet),var(--pink),var(--aqua));
     background-size:300% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;
     animation:shine 8s linear infinite}
  @keyframes shine{to{background-position:300% 0}}
  .sub{color:var(--muted);max-width:56ch;line-height:1.6;margin:0 0 32px}

  .layout{display:grid;grid-template-columns:1.4fr 1fr;gap:24px}
  @media(max-width:860px){.layout{grid-template-columns:1fr}}

  .card{background:var(--glass);border:1px solid var(--line);border-radius:22px;padding:26px;
        backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px 18px}
  @media(max-width:560px){.grid{grid-template-columns:1fr}}
  label{display:block;font-size:.85rem;color:var(--muted);margin-bottom:6px}
  input,select{width:100%;padding:12px 14px;border-radius:12px;border:1px solid var(--line);
               background:rgba(10,14,31,.6);color:var(--ink);font:inherit;outline:none;
               transition:border-color .2s, box-shadow .2s}
  input:focus,select:focus{border-color:var(--aqua);box-shadow:0 0 0 3px rgba(34,211,238,.22)}
  select option{background:#12172f}

  button{margin-top:22px;width:100%;padding:15px;border:0;border-radius:14px;cursor:pointer;
         font:600 1rem Sora,sans-serif;color:#0a0e1f;
         background:linear-gradient(90deg,var(--aqua),var(--violet));background-size:200% 100%;
         transition:transform .15s, background-position .5s, box-shadow .3s;
         box-shadow:0 10px 30px -10px rgba(124,92,255,.7)}
  button:hover{background-position:100% 0;transform:translateY(-2px)}
  button:active{transform:scale(.98)}
  button.loading{pointer-events:none;opacity:.8}

  /* Result panel */
  .result{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;min-height:380px}
  .ring{position:relative;width:190px;height:190px;margin-bottom:18px}
  .ring svg{transform:rotate(-90deg)}
  .ring circle{fill:none;stroke-width:12;stroke-linecap:round}
  .ring .track{stroke:rgba(255,255,255,.09)}
  .ring .bar{stroke:var(--aqua);stroke-dasharray:502;stroke-dashoffset:502;
             transition:stroke-dashoffset 1.3s cubic-bezier(.2,.8,.2,1), stroke .4s}
  .ring .center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .pct{font:800 2.3rem Sora,sans-serif}
  .lvl{font:600 1.1rem Sora,sans-serif;margin-top:2px}
  .idle .ring{animation:pulse 3s ease-in-out infinite}
  @keyframes pulse{50%{transform:scale(1.04);filter:drop-shadow(0 0 18px rgba(124,92,255,.5))}}
  .hint{color:var(--muted);font-size:.9rem;line-height:1.5;max-width:28ch}

  .probs{width:100%;margin-top:14px;display:grid;gap:10px;text-align:left}
  .prow{display:grid;grid-template-columns:64px 1fr 44px;align-items:center;gap:10px;font-size:.85rem}
  .track2{height:8px;border-radius:99px;background:rgba(255,255,255,.09);overflow:hidden}
  .fill{height:100%;width:0;border-radius:99px;transition:width 1.1s cubic-bezier(.2,.8,.2,1)}
  .pop{animation:pop .6s cubic-bezier(.2,1.4,.4,1)}
  @keyframes pop{from{transform:scale(.6);opacity:0}to{transform:scale(1);opacity:1}}
  .err{color:var(--low);font-size:.9rem;margin-top:12px}

  @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="aurora"><span></span><span></span><span></span></div>
<main>
  <h1>GenAI Impact Predictor</h1>
  <p class="sub">Fill in a student profile and the logistic regression model estimates whether the outcome level is High, Medium or Low.</p>

  <div class="layout">
    <form class="card" id="form">
      <div class="grid">
        {% for col, (label, lo, hi, step, default) in numeric.items() %}
        <div>
          <label for="{{col}}">{{label}}</label>
          <input type="number" id="{{col}}" name="{{col}}" min="{{lo}}" max="{{hi}}" step="{{step}}" value="{{default}}" required>
        </div>
        {% endfor %}
        {% for col, opts in categories.items() %}
        <div>
          <label for="{{col}}">{{labels[col]}}</label>
          <select id="{{col}}" name="{{col}}">
            {% for o in opts %}<option value="{{o}}">{{o}}</option>{% endfor %}
          </select>
        </div>
        {% endfor %}
      </div>
      <button type="submit" id="btn">Predict outcome</button>
      <div class="err" id="err"></div>
    </form>

    <section class="card result idle" id="result">
      <div class="ring">
        <svg width="190" height="190" viewBox="0 0 190 190">
          <circle class="track" cx="95" cy="95" r="80"/>
          <circle class="bar" id="bar" cx="95" cy="95" r="80"/>
        </svg>
        <div class="center"><div class="pct" id="pct">--</div><div class="lvl" id="lvl">Waiting</div></div>
      </div>
      <p class="hint" id="hint">Your prediction and class probabilities will appear here.</p>
      <div class="probs" id="probs"></div>
    </section>
  </div>
</main>

<script>
const COLORS = {High:'var(--high)', Medium:'var(--medium)', Low:'var(--low)'};
const form = document.getElementById('form');
const btn = document.getElementById('btn');

function countUp(el, to){
  const t0 = performance.now(), dur = 1200;
  (function step(t){
    const p = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (to * e).toFixed(1) + '%';
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  document.getElementById('err').textContent = '';
  btn.classList.add('loading'); btn.textContent = 'Thinking...';
  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.fromEntries(new FormData(form)))
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');

    const box = document.getElementById('result');
    box.classList.remove('idle');
    const color = COLORS[data.prediction] || 'var(--aqua)';
    const top = Math.max(...Object.values(data.probabilities));

    const bar = document.getElementById('bar');
    bar.style.stroke = color;
    bar.style.strokeDashoffset = 502 - 502 * top / 100;

    const lvl = document.getElementById('lvl');
    lvl.textContent = data.prediction; lvl.style.color = color;
    lvl.classList.remove('pop'); void lvl.offsetWidth; lvl.classList.add('pop');
    countUp(document.getElementById('pct'), top);
    document.getElementById('hint').textContent = 'Model confidence in this class';

    const probs = document.getElementById('probs');
    probs.innerHTML = Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).map(([k,v]) =>
      `<div class="prow"><span>${k}</span><div class="track2"><div class="fill" data-w="${v}" style="background:${COLORS[k]}"></div></div><span>${v.toFixed(1)}%</span></div>`
    ).join('');
    requestAnimationFrame(() => requestAnimationFrame(() =>
      probs.querySelectorAll('.fill').forEach(f => f.style.width = f.dataset.w + '%')));
  } catch (err) {
    document.getElementById('err').textContent = err.message;
  } finally {
    btn.classList.remove('loading'); btn.textContent = 'Predict outcome';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE, numeric=NUMERIC, categories=CATEGORIES, labels=LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        row = {}
        for col in FEATURES:
            if col in ENCODING:
                row[col] = ENCODING[col][payload[col]]
            else:
                row[col] = float(payload[col])
        X = pd.DataFrame([row], columns=FEATURES)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return jsonify(
            prediction=str(pred),
            probabilities={str(c): round(float(p) * 100, 2) for c, p in zip(model.classes_, proba)},
        )
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Invalid or missing input: {e}"), 400


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
