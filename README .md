# 🧠 AI Study Pack Generator

A multi-stage AI workflow that turns a topic + learner profile into a
personalized study pack (explanations, examples, quizzes, flashcards),
built with an open-source **Groq** LLM and deployed on **Streamlit
Community Cloud**.

> **Note on tech stack:** this app's UI is built in **Streamlit**, not
> Gradio. Streamlit Cloud only runs `streamlit run app.py` — a Gradio app
> calling `.launch()` starts its own server on its own port, which
> Streamlit Cloud does not expose, so Gradio apps do not deploy there.
> Streamlit gives you the identical "push to GitHub → one-click deploy →
> add a secret" flow you asked for, and it actually comes up. If you want
> a real Gradio UI, deploy to **Hugging Face Spaces** instead (ask and I
> can give you that variant).

---

## How the workflow works

This isn't a single prompt — it's five agents, each with its own system
prompt, that pass a shared `context` object down the pipeline:

```
inputs
  │
  ▼
[1] PLANNING agent
  → produces: plan { sections: [{ id, title, objectives, key_topics, estimated_minutes }] }
  │
  ▼
[2] CONTENT GENERATION agent (runs once per section)
  → reads: plan.sections[i]
  → produces: content[section_id] { explanation, key_points, examples, summary }
  │
  ▼
[3] ASSESSMENT agent (runs once per section)
  → reads: content[section_id]
  → produces: assessment[section_id] { mcqs, short_answer, flashcards }
  │
  ▼
[4] REVIEW agent (single pass over the whole pack)
  → reads: plan + content summaries + assessment presence
  → produces: review { overall_score, needs_refinement, section_feedback[] }
  │
  ▼
[5] REFINEMENT agent (runs only on sections the reviewer flagged)
  → reads: original content + assessment + reviewer issues/suggestions
  → produces: refined_content[section_id], refined_assessment[section_id]
  │
  ▼
compiled study pack (Markdown + in-app tabs)
```

### Context passing
Everything lives in one `WorkflowContext` dataclass (`inputs`, `plan`,
`content`, `assessment`, `review`, `refined_content`,
`refined_assessment`, `errors`, `stage_status`). Each stage function only
reads the fields it needs from earlier stages and only writes its own
field — no stage re-derives information another stage already produced.

### Error handling
- Every Groq call goes through `GroqClient.chat()`, which retries up to 3
  times with exponential backoff (handles transient network errors, rate
  limits, empty responses).
- All LLM output is JSON; `extract_json()` strips markdown fences and
  falls back to locating the outermost `{...}` block before parsing, so
  minor formatting drift from the model doesn't break the pipeline.
- **Planning** and **Content** are treated as critical: if planning fails,
  or if *every* section fails content generation, the run aborts with a
  clear error message naming the failing stage.
- **Assessment**, **Review**, and **Refinement** degrade gracefully: a
  failure is logged to `context.errors` and `stage_status`, and the
  pipeline continues, ultimately shipping the best pack it could produce
  (e.g. content without quizzes, or pre-refinement content if refinement
  errors out) rather than crashing.
- The UI shows a live status log per stage plus a status-icon strip
  (✅ ok / 🟡 partial / ❌ failed / ⏭️ skipped) so you can see exactly what
  happened on a given run.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | The full Streamlit app: workflow logic + UI |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 1. Get a Groq API key

1. Sign up at [console.groq.com](https://console.groq.com).
2. Create an API key (free tier available).

## 2. Push to GitHub

Create a new repo and add these three files (`app.py`,
`requirements.txt`, `README.md`) at the repo root.

## 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
2. Click **New app**, select your repo/branch, and set the main file
   path to `app.py`.
3. Before (or after) deploying, open **App settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and
   launches the app automatically.

> Local testing: if you don't want to use `st.secrets` locally, the
> sidebar has a manual API-key field for that session only (it is not
> persisted anywhere).

---

## Using the app

1. Pick a model (Groq offers several — `llama-3.3-70b-versatile` is the
   strongest default; `llama-3.1-8b-instant` is faster/cheaper).
2. Set how many sections you want (2–6).
3. Enter a topic, level, target duration, and optional goals.
4. Click **Generate Study Pack** and watch the stage log.
5. Browse the result in per-section tabs (explanation, key points,
   examples, quiz, flashcards), then download the whole pack as Markdown.

---

## Extending this

- **More stages**: add a function following the `stage_*(client, ctx,
  log)` pattern, register it in `run_workflow`, and give it its own
  system prompt constant.
- **Different LLM provider**: swap `GroqClient` for another OpenAI-style
  client — the rest of the pipeline (context, JSON schemas, error
  handling) is provider-agnostic.
- **Persistence**: currently a run's `WorkflowContext` lives only in
  `st.session_state` for the browser session. Swap in a database or file
  write inside `run_workflow` if you want saved history across sessions.
- **PDF export**: `compile_markdown()` already assembles a clean Markdown
  document — pipe it through a Markdown→PDF converter if you want a PDF
  download instead of/alongside the `.md` file.
