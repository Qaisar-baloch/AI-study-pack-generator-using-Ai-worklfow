"""
AI Study Pack Generator
------------------------
A multi-stage agentic workflow (Planning -> Content Generation -> Assessment
-> Review -> Refinement) that turns a topic + learner profile into a
personalized study pack, using Groq's LLM API.

Deploy target: Streamlit Community Cloud.
"""

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st
from groq import Groq


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

APP_TITLE = "🧠 AI Study Pack Generator"
MAX_SECTIONS = 6
MIN_SECTIONS = 2
DEFAULT_MODEL = "llama-3.3-70b-versatile"
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]
STAGE_ORDER = ["planning", "content", "assessment", "review", "refinement"]
CRITICAL_STAGES = {"planning", "content"}  # failure here aborts the run

# --------------------------------------------------------------------------
# Stage system prompts
# --------------------------------------------------------------------------

PLANNING_SYSTEM = """You are an expert curriculum designer and planning agent \
- the first stage in a multi-stage AI study pack pipeline. Your sole job is \
to turn a learner's request into a structured, sequenced study plan as valid \
JSON. You do NOT write teaching content or questions in this stage.

Output ONLY valid JSON, no markdown fences, no commentary. Schema:
{
  "topic": string,
  "level": string,
  "total_duration_minutes": number,
  "sections": [
    {
      "id": number,
      "title": string,
      "objectives": [string, ...],
      "key_topics": [string, ...],
      "estimated_minutes": number
    }
  ]
}
Produce exactly the requested number of sections, ordered from foundational \
to advanced. estimated_minutes across sections should sum close to \
total_duration_minutes."""

CONTENT_SYSTEM = """You are the content-generation agent, stage 2 of a study \
pack pipeline. You receive one section of an approved study plan and must \
produce clear, accurate learning content for it, calibrated to the \
learner's level.

Output ONLY valid JSON, no markdown fences:
{
  "section_id": number,
  "explanation": string,
  "key_points": [string, ...],
  "examples": [string, ...],
  "summary": string
}
Stay strictly within the section's objectives and key topics. Do not invent \
unrelated material. explanation should be 3-6 paragraphs. key_points 4-8 \
items. examples 2-4 items."""

ASSESSMENT_SYSTEM = """You are the assessment-generation agent, stage 3 of a \
study pack pipeline. Given one section's content, produce a matching quiz \
and flashcards that test the stated objectives.

Output ONLY valid JSON, no markdown fences:
{
  "section_id": number,
  "mcqs": [
    {"question": string, "options": [string, string, string, string],
     "correct_index": number, "explanation": string}
  ],
  "short_answer": [
    {"question": string, "model_answer": string}
  ],
  "flashcards": [
    {"front": string, "back": string}
  ]
}
Produce exactly 3 mcqs, 2 short_answer, 3 flashcards. Base every question \
strictly on the provided content. Do not test material not covered."""

REVIEW_SYSTEM = """You are the quality-review agent, stage 4 of a study pack \
pipeline. You audit generated content and assessments against the original \
plan's objectives for accuracy, clarity, level-appropriateness, and \
coverage.

Output ONLY valid JSON, no markdown fences:
{
  "overall_score": number,
  "needs_refinement": boolean,
  "section_feedback": [
    {"section_id": number, "score": number, "issues": [string, ...],
     "suggestions": [string, ...]}
  ]
}
Scores are 1-5. Flag needs_refinement=true if ANY section scores below 4, \
or if you find factual errors, unclear explanations, or objective \
mismatches. Be specific and actionable - the next agent uses your \
suggestions verbatim."""

REFINEMENT_SYSTEM = """You are the refinement agent, the final stage of a \
study pack pipeline. You receive one section's original content, its \
assessment, and reviewer feedback. Revise BOTH content and assessment to \
resolve every issue raised, keeping the same JSON structures used earlier.

Output ONLY valid JSON, no markdown fences:
{
  "section_id": number,
  "content": {"explanation": string, "key_points": [string, ...],
              "examples": [string, ...], "summary": string},
  "assessment": {
    "mcqs": [{"question": string, "options": [string,string,string,string],
              "correct_index": number, "explanation": string}],
    "short_answer": [{"question": string, "model_answer": string}],
    "flashcards": [{"front": string, "back": string}]
  }
}
Address every reviewer issue directly. Preserve what already worked well."""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class WorkflowError(Exception):
    """Raised when a stage cannot produce usable output after retries."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


# --------------------------------------------------------------------------
# JSON extraction helper (LLMs sometimes wrap JSON in prose/fences)
# --------------------------------------------------------------------------

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)  # let this raise if still bad
    raise ValueError("No JSON object found in model output")


# --------------------------------------------------------------------------
# Groq client wrapper with retry/backoff
# --------------------------------------------------------------------------

class GroqClient:
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.4,
              max_tokens: int = 2000, retries: int = 3) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Empty response from model")
                return content
            except Exception as e:  # network, rate limit, empty response, etc.
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise last_err  # caller wraps this into WorkflowError


# --------------------------------------------------------------------------
# Workflow context
# --------------------------------------------------------------------------

@dataclass
class WorkflowContext:
    inputs: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[Dict[str, Any]] = None
    content: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    assessment: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    review: Optional[Dict[str, Any]] = None
    refined_content: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    refined_assessment: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    stage_status: Dict[str, str] = field(default_factory=dict)  # ok/failed/skipped


# --------------------------------------------------------------------------
# Stage functions
# --------------------------------------------------------------------------

def stage_planning(client: GroqClient, ctx: WorkflowContext, log) -> None:
    inp = ctx.inputs
    user_prompt = (
        f"Topic: {inp['topic']}\n"
        f"Learner level: {inp['level']}\n"
        f"Total study duration (minutes): {inp['duration']}\n"
        f"Learning goals: {inp['goals']}\n"
        f"Number of sections required: exactly {inp['num_sections']}"
    )
    try:
        raw = client.chat(PLANNING_SYSTEM, user_prompt, temperature=0.3, max_tokens=1500)
        plan = extract_json(raw)
        if not plan.get("sections"):
            raise ValueError("Plan is missing a non-empty 'sections' list")
        for s in plan["sections"]:
            if "id" not in s or "title" not in s:
                raise ValueError("A section is missing required fields (id/title)")
        ctx.plan = plan
        ctx.stage_status["planning"] = "ok"
        log(f"Plan created with {len(plan['sections'])} sections.")
    except Exception as e:
        ctx.errors.append(f"planning: {e}")
        ctx.stage_status["planning"] = "failed"
        raise WorkflowError("planning", str(e))


def stage_content(client: GroqClient, ctx: WorkflowContext, log) -> None:
    assert ctx.plan is not None
    failures = 0
    for section in ctx.plan["sections"]:
        sid = section["id"]
        user_prompt = (
            f"Section title: {section['title']}\n"
            f"Objectives: {section.get('objectives', [])}\n"
            f"Key topics: {section.get('key_topics', [])}\n"
            f"Learner level: {ctx.inputs['level']}\n"
            f"Section id to echo back: {sid}"
        )
        try:
            raw = client.chat(CONTENT_SYSTEM, user_prompt, temperature=0.5, max_tokens=1800)
            data = extract_json(raw)
            if "explanation" not in data:
                raise ValueError("Content missing 'explanation'")
            ctx.content[sid] = data
            log(f"Content generated for section {sid}: {section['title']}")
        except Exception as e:
            failures += 1
            ctx.errors.append(f"content[section {sid}]: {e}")
            log(f"⚠️ Content generation failed for section {sid}: {e}")

    if not ctx.content:
        ctx.stage_status["content"] = "failed"
        raise WorkflowError("content", "No sections produced usable content")
    ctx.stage_status["content"] = "ok" if failures == 0 else "partial"


def stage_assessment(client: GroqClient, ctx: WorkflowContext, log) -> None:
    failures = 0
    for sid, content in ctx.content.items():
        user_prompt = (
            f"Section id: {sid}\n"
            f"Explanation: {content.get('explanation', '')}\n"
            f"Key points: {content.get('key_points', [])}\n"
            f"Examples: {content.get('examples', [])}"
        )
        try:
            raw = client.chat(ASSESSMENT_SYSTEM, user_prompt, temperature=0.5, max_tokens=1500)
            data = extract_json(raw)
            if "mcqs" not in data:
                raise ValueError("Assessment missing 'mcqs'")
            ctx.assessment[sid] = data
            log(f"Assessment generated for section {sid}.")
        except Exception as e:
            failures += 1
            ctx.errors.append(f"assessment[section {sid}]: {e}")
            log(f"⚠️ Assessment generation failed for section {sid}: {e}")

    if not ctx.assessment:
        ctx.stage_status["assessment"] = "failed"
        raise WorkflowError("assessment", "No sections produced usable assessments")
    ctx.stage_status["assessment"] = "ok" if failures == 0 else "partial"


def stage_review(client: GroqClient, ctx: WorkflowContext, log) -> None:
    summary = []
    for section in ctx.plan["sections"]:
        sid = section["id"]
        summary.append({
            "section_id": sid,
            "title": section["title"],
            "objectives": section.get("objectives", []),
            "content_summary": ctx.content.get(sid, {}).get("summary", ""),
            "has_assessment": sid in ctx.assessment,
        })
    user_prompt = f"Plan + generated material overview:\n{json.dumps(summary, indent=2)}"
    try:
        raw = client.chat(REVIEW_SYSTEM, user_prompt, temperature=0.3, max_tokens=1500)
        review = extract_json(raw)
        if "section_feedback" not in review:
            raise ValueError("Review missing 'section_feedback'")
        ctx.review = review
        ctx.stage_status["review"] = "ok"
        log(f"Review complete. Overall score: {review.get('overall_score')}, "
            f"needs_refinement={review.get('needs_refinement')}")
    except Exception as e:
        ctx.errors.append(f"review: {e}")
        ctx.stage_status["review"] = "failed"
        log(f"⚠️ Review stage failed (non-critical, continuing without it): {e}")


def stage_refinement(client: GroqClient, ctx: WorkflowContext, log) -> None:
    if not ctx.review or not ctx.review.get("needs_refinement"):
        ctx.stage_status["refinement"] = "skipped"
        log("No refinement needed - review found the pack satisfactory (or review was unavailable).")
        return

    flagged = [
        fb for fb in ctx.review.get("section_feedback", [])
        if fb.get("score", 5) < 4 and fb.get("section_id") in ctx.content
    ]
    if not flagged:
        ctx.stage_status["refinement"] = "skipped"
        log("Review flagged no specific sections below threshold.")
        return

    failures = 0
    for fb in flagged:
        sid = fb["section_id"]
        user_prompt = (
            f"Section id: {sid}\n"
            f"Original content: {json.dumps(ctx.content.get(sid, {}))}\n"
            f"Original assessment: {json.dumps(ctx.assessment.get(sid, {}))}\n"
            f"Reviewer issues: {fb.get('issues', [])}\n"
            f"Reviewer suggestions: {fb.get('suggestions', [])}"
        )
        try:
            raw = client.chat(REFINEMENT_SYSTEM, user_prompt, temperature=0.4, max_tokens=2000)
            data = extract_json(raw)
            if "content" not in data or "assessment" not in data:
                raise ValueError("Refinement output missing content/assessment")
            ctx.refined_content[sid] = data["content"]
            ctx.refined_assessment[sid] = data["assessment"]
            log(f"Refined section {sid}.")
        except Exception as e:
            failures += 1
            ctx.errors.append(f"refinement[section {sid}]: {e}")
            log(f"⚠️ Refinement failed for section {sid}, keeping original: {e}")

    ctx.stage_status["refinement"] = "ok" if failures == 0 else "partial"


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def run_workflow(api_key: str, model: str, inputs: Dict[str, Any], log) -> WorkflowContext:
    ctx = WorkflowContext(inputs=inputs)
    client = GroqClient(api_key=api_key, model=model)

    stage_planning(client, ctx, log)      # critical - raises on failure
    stage_content(client, ctx, log)       # critical - raises only if ALL sections fail

    try:
        stage_assessment(client, ctx, log)  # non-critical-ish: degrade if all fail
    except WorkflowError as e:
        log(f"⚠️ Assessment stage failed entirely, continuing without quizzes: {e.message}")

    try:
        stage_review(client, ctx, log)
    except Exception as e:
        ctx.errors.append(f"review: {e}")
        ctx.stage_status["review"] = "failed"

    try:
        stage_refinement(client, ctx, log)
    except Exception as e:
        ctx.errors.append(f"refinement: {e}")
        ctx.stage_status["refinement"] = "failed"
        log(f"⚠️ Refinement stage failed, final pack uses pre-refinement content: {e}")

    return ctx


# --------------------------------------------------------------------------
# Output assembly
# --------------------------------------------------------------------------

def final_content_for(ctx: WorkflowContext, sid: int) -> Dict[str, Any]:
    return ctx.refined_content.get(sid, ctx.content.get(sid, {}))


def final_assessment_for(ctx: WorkflowContext, sid: int) -> Dict[str, Any]:
    return ctx.refined_assessment.get(sid, ctx.assessment.get(sid, {}))


def compile_markdown(ctx: WorkflowContext) -> str:
    plan = ctx.plan or {}
    lines = [f"# Study Pack: {plan.get('topic', ctx.inputs.get('topic', ''))}", ""]
    lines.append(f"**Level:** {plan.get('level', ctx.inputs.get('level', ''))}  ")
    lines.append(f"**Total duration:** {plan.get('total_duration_minutes', ctx.inputs.get('duration', ''))} minutes")
    lines.append("")

    if ctx.review:
        lines.append(f"> Review score: {ctx.review.get('overall_score', 'N/A')}/5 "
                      f"{'(refined)' if ctx.refined_content else ''}")
        lines.append("")

    for section in plan.get("sections", []):
        sid = section["id"]
        content = final_content_for(ctx, sid)
        assessment = final_assessment_for(ctx, sid)

        lines.append(f"## {section['title']}")
        if content.get("explanation"):
            lines.append(content["explanation"])
            lines.append("")
        if content.get("key_points"):
            lines.append("**Key points:**")
            for kp in content["key_points"]:
                lines.append(f"- {kp}")
            lines.append("")
        if content.get("examples"):
            lines.append("**Examples:**")
            for ex in content["examples"]:
                lines.append(f"- {ex}")
            lines.append("")
        if content.get("summary"):
            lines.append(f"*Summary: {content['summary']}*")
            lines.append("")

        if assessment.get("mcqs"):
            lines.append("### Quiz")
            for i, q in enumerate(assessment["mcqs"], 1):
                lines.append(f"{i}. {q['question']}")
                for j, opt in enumerate(q.get("options", [])):
                    marker = "✅" if j == q.get("correct_index") else "  "
                    lines.append(f"   {marker} {chr(97+j)}) {opt}")
                if q.get("explanation"):
                    lines.append(f"   _Explanation: {q['explanation']}_")
            lines.append("")
        if assessment.get("short_answer"):
            lines.append("### Short answer")
            for q in assessment["short_answer"]:
                lines.append(f"- Q: {q['question']}")
                lines.append(f"  A: {q['model_answer']}")
            lines.append("")
        if assessment.get("flashcards"):
            lines.append("### Flashcards")
            for fc in assessment["flashcards"]:
                lines.append(f"- **{fc['front']}** — {fc['back']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    if ctx.errors:
        lines.append("<!-- Workflow notes (not shown to learner):")
        for e in ctx.errors:
            lines.append(f"  - {e}")
        lines.append("-->")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("manual_api_key")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "A 5-stage AI workflow — Planning → Content → Assessment → Review → "
        "Refinement — that builds a personalized study pack with Groq."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = get_api_key()
        if not api_key:
            st.warning("No GROQ_API_KEY found in Streamlit secrets.")
            manual_key = st.text_input("Enter Groq API key (for local testing only)", type="password")
            if manual_key:
                st.session_state["manual_api_key"] = manual_key
                api_key = manual_key
        else:
            st.success("Groq API key loaded from secrets.")

        model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
        num_sections = st.slider("Number of sections", MIN_SECTIONS, MAX_SECTIONS, 4)

        st.markdown("---")
        st.caption(
            "Errors in non-critical stages (assessment, review, refinement) "
            "degrade gracefully instead of failing the whole run."
        )

    st.subheader("📋 Study Pack Request")
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic", placeholder="e.g. Introduction to Neural Networks")
        level = st.selectbox("Learner level", ["Beginner", "Intermediate", "Advanced"])
    with col2:
        duration = st.number_input("Total study duration (minutes)", min_value=15, max_value=600, value=90, step=15)
        goals = st.text_area("Learning goals (optional)", placeholder="e.g. Be able to explain backpropagation and implement a simple network")

    generate = st.button("🚀 Generate Study Pack", type="primary", use_container_width=True)

    if generate:
        if not api_key:
            st.error("Please provide a Groq API key (via Streamlit secrets or the sidebar field).")
            return
        if not topic.strip():
            st.error("Please enter a topic.")
            return

        inputs = {
            "topic": topic.strip(),
            "level": level,
            "duration": duration,
            "goals": goals.strip() or "General mastery of the topic",
            "num_sections": num_sections,
        }

        log_area = st.status("Running workflow...", expanded=True)

        def log(msg: str):
            log_area.write(msg)

        try:
            ctx = run_workflow(api_key, model, inputs, log)
            log_area.update(label="Workflow complete", state="complete")
            st.session_state["last_ctx"] = ctx
        except WorkflowError as e:
            log_area.update(label="Workflow failed", state="error")
            st.error(f"Generation failed at the **{e.stage}** stage: {e.message}")
            return
        except Exception as e:
            log_area.update(label="Workflow failed", state="error")
            st.error(f"Unexpected error: {e}")
            st.code(traceback.format_exc())
            return

    ctx: Optional[WorkflowContext] = st.session_state.get("last_ctx")
    if ctx and ctx.plan:
        st.markdown("---")
        st.subheader("📚 Your Study Pack")

        status_cols = st.columns(len(STAGE_ORDER))
        icons = {"ok": "✅", "partial": "🟡", "failed": "❌", "skipped": "⏭️"}
        for col, stage in zip(status_cols, STAGE_ORDER):
            status = ctx.stage_status.get(stage, "—")
            col.metric(stage.capitalize(), icons.get(status, status))

        tabs = st.tabs(["Overview"] + [s["title"] for s in ctx.plan["sections"]])

        with tabs[0]:
            st.write(f"**Topic:** {ctx.plan.get('topic')}")
            st.write(f"**Level:** {ctx.plan.get('level')}")
            st.write(f"**Total duration:** {ctx.plan.get('total_duration_minutes')} min")
            if ctx.review:
                st.write(f"**Review score:** {ctx.review.get('overall_score')}/5")
            if ctx.errors:
                with st.expander("⚠️ Workflow warnings/errors"):
                    for e in ctx.errors:
                        st.write(f"- {e}")

        for tab, section in zip(tabs[1:], ctx.plan["sections"]):
            sid = section["id"]
            content = final_content_for(ctx, sid)
            assessment = final_assessment_for(ctx, sid)
            with tab:
                if sid in ctx.refined_content:
                    st.caption("✨ This section was refined based on automated review.")
                st.markdown(content.get("explanation", "_No content generated for this section._"))
                if content.get("key_points"):
                    st.markdown("**Key points**")
                    for kp in content["key_points"]:
                        st.markdown(f"- {kp}")
                if content.get("examples"):
                    st.markdown("**Examples**")
                    for ex in content["examples"]:
                        st.markdown(f"- {ex}")

                if assessment.get("mcqs"):
                    st.markdown("### Quiz")
                    for i, q in enumerate(assessment["mcqs"], 1):
                        with st.expander(f"Q{i}. {q['question']}"):
                            for j, opt in enumerate(q.get("options", [])):
                                marker = "✅" if j == q.get("correct_index") else "▫️"
                                st.write(f"{marker} {opt}")
                            st.caption(q.get("explanation", ""))
                if assessment.get("flashcards"):
                    st.markdown("### Flashcards")
                    for fc in assessment["flashcards"]:
                        with st.expander(fc["front"]):
                            st.write(fc["back"])

        st.markdown("---")
        md = compile_markdown(ctx)
        st.download_button(
            "⬇️ Download full study pack (Markdown)",
            data=md,
            file_name=f"study_pack_{ctx.plan.get('topic', 'output').replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
