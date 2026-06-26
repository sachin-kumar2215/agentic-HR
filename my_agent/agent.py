import os
from google.adk.agents import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.models.lite_llm import LiteLlm
import datetime


def get_model():
    return LiteLlm(
        model="ollama_chat/batiai/gemma4-e2b:q4",
        extra_body={"think": False},
    )


# ── Tools ──────────────────────────────────────────────────────────────────────

def read_file(file_path: str) -> dict:
    """
    Reads a text or PDF file and returns its content.

    Args:
        file_path: Path to a .txt, .md, or .pdf file.

    Returns:
        Dict with 'content' string or 'error'.
    """
    try:
        path = os.path.expanduser(file_path.strip())
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                return {"content": text.strip(), "type": "pdf"}
            except ImportError:
                return {"error": "pypdf not installed. Run: pip install pypdf"}

        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read().strip(), "type": ext}

    except Exception as e:
        return {"error": str(e)}


def save_report(filename: str, content: str) -> dict:
    """
    Saves content to a file.

    Args:
        filename: Output filename e.g. 'job_report.md'
        content:  Text content to write.

    Returns:
        Dict with 'path' on success or 'error'.
    """
    try:
        path = os.path.abspath(filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"path": path, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


def get_current_time() -> str:
    """
    Retrieves the current date and time in a readable string format.
    This function can be used as a tool to provide real-time context.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


# ── Step 1: Input Agent ────────────────────────────────────────────────────────
# Reads both files and stores in session state

input_agent = Agent(
    model=get_model(),
    name="InputAgent",
    description="Reads the job description and resume files.",
    instruction="""
        You are a file reader assistant.

        The user will provide two file paths:
        1. Job description file (txt, md, or pdf)
        2. Resume file (txt, md, or pdf)

        Steps:
        1. Call read_file for the job description path.
        2. Call read_file for the resume path.
        3. Combine them into this exact format and output ONLY this:

        JOB_DESCRIPTION:
        <full job description content>

        RESUME:
        <full resume content>

        If any file fails, output: FILE_ERROR: <message>
        Do not add any other commentary.
    """,
    output_key="jd_and_resume",
    tools=[read_file],
)


# ── Step 2a: Skill Match Agent ─────────────────────────────────────────────────

skill_match_agent = Agent(
    model=get_model(),
    name="SkillMatchAgent",
    description="Matches candidate skills against job requirements.",
    instruction="""
        You are a technical recruiter analyzing job fit.
        Here is the job description and resume:

        {jd_and_resume}

        Analyze skill alignment and output ONLY this report:

        ## Skill Match Analysis

        **Match Score:** X/10

        **✅ Strong Matches** (skills you have that they want):
        - skill 1
        - skill 2

        **⚠️ Partial Matches** (you have related but not exact skills):
        - skill 1
        - skill 2

        **❌ Missing Skills** (required but not on resume):
        - skill 1
        - skill 2

        **🌟 Bonus Skills** (you have these but they didn't ask):
        - skill 1

        Keep each section to max 5 bullet points.
    """,
    output_key="skill_match",
)


# ── Step 2b: Gap Analysis Agent ────────────────────────────────────────────────

gap_analysis_agent = Agent(
    model=get_model(),
    name="GapAnalysisAgent",
    description="Identifies experience and qualification gaps.",
    instruction="""
        You are a career coach analyzing qualification gaps.
        Here is the job description and resume:

        {jd_and_resume}

        Analyze the gaps and output ONLY this report:

        ## Gap Analysis

        **Experience Gap:** (Overqualified / Good Fit / Slightly Under / Underqualified)

        **📋 Requirements vs Reality:**
        | Requirement | Candidate Status |
        |-------------|-----------------|
        | X years experience | Has Y years |
        | Degree in X | Has Y |
        | (add more rows) | |

        **🚨 Critical Gaps** (deal-breakers if not addressed):
        - gap 1
        - gap 2

        **📈 Bridgeable Gaps** (can be addressed quickly):
        - gap 1 → how to bridge it
        - gap 2 → how to bridge it

        **💪 Strengths That Compensate:**
        - strength that offsets a gap
    """,
    output_key="gap_analysis",
)


# ── Step 2c: Salary Insights Agent ────────────────────────────────────────────

salary_agent = Agent(
    model=get_model(),
    name="SalaryAgent",
    description="Estimates salary range and negotiation tips.",
    instruction="""
        You are a compensation analyst.
        Here is the job description and resume:

        {jd_and_resume}

        Analyze compensation and output ONLY this report:

        ## Salary Insights

        **Role:** (job title from JD)
        **Seniority Level:** (Junior / Mid / Senior / Lead / Principal)
        **Industry:** (detected from JD)

        **💰 Estimated Salary Range:**
        - Entry point: $X
        - Mid point: $X
        - Top of range: $X
        (Base these on role, seniority, and industry standards)

        **🎯 Recommended Ask:** $X - $Y
        (Based on the candidate's experience level)

        **📊 Negotiation Leverage:**
        - Point 1 (e.g. rare skill they have)
        - Point 2

        **⚠️ Watch Out For:**
        - Red flag 1 in the JD if any
        - Red flag 2

        **🤝 Benefits to Negotiate Beyond Salary:**
        - benefit 1
        - benefit 2
    """,
    output_key="salary_insights",
)


# ── Step 2d: Cover Letter Agent ────────────────────────────────────────────────

cover_letter_agent = Agent(
    model=get_model(),
    name="CoverLetterAgent",
    description="Drafts a tailored cover letter.",
    instruction="""
        You are an expert cover letter writer.
        Here is the job description and resume:

        {jd_and_resume}

        Write a tailored cover letter and output ONLY the letter:

        ---
        [Date]

        Hiring Manager
        [Company Name from JD]

        Dear Hiring Manager,

        [Opening paragraph: Show genuine enthusiasm for the role and company.
         Reference something specific from the JD.]

        [Body paragraph 1: Match your strongest 2-3 skills directly to their
         top requirements. Use specific examples from the resume.]

        [Body paragraph 2: Address one potential concern or gap proactively,
         or highlight a unique differentiator.]

        [Closing paragraph: Clear call to action, thank them.]

        Sincerely,
        [Candidate Name from resume]
        ---

        Keep the total letter under 350 words. Make it specific, not generic.
    """,
    output_key="cover_letter_draft",
)


# ── Step 2: Parallel Agent (fan-out) ──────────────────────────────────────────

parallel_analysis_agent = ParallelAgent(
    name="ParallelJobAnalysisAgent",
    description="Runs all 4 job analysis specialists simultaneously.",
    sub_agents=[
        skill_match_agent,
        gap_analysis_agent,
        salary_agent,
        cover_letter_agent,
    ],
)


# ── Step 3: Action Plan Agent (synthesizer) ────────────────────────────────────

action_plan_agent = Agent(
    model=get_model(),
    name="ActionPlanAgent",
    description="Combines all analyses into a final job application action plan.",
    instruction="""
        You are a senior career advisor. Combine all specialist reports
        into one final job application report.

        SKILL MATCH REPORT:
        {skill_match}

        GAP ANALYSIS REPORT:
        {gap_analysis}

        SALARY INSIGHTS:
        {salary_insights}

        COVER LETTER DRAFT:
        {cover_letter_draft}

        Write a final report in this exact structure:

        # 📋 Job Application Report

        ## 🎯 Should You Apply?
        (Give a clear YES / YES WITH PREPARATION / REACH / NO with 2-3 sentence reasoning)

        ## Skill Match
        {paste skill_match report here}

        ## Gap Analysis
        {paste gap_analysis report here}

        ## Salary Insights
        {paste salary_insights report here}

        ## Cover Letter
        {paste cover_letter_draft here}

        ## ✅ Your 7-Day Action Plan
        Before applying, do these in order:
        Day 1-2: (address most critical gap)
        Day 3-4: (update resume for this role)
        Day 5:   (tailor cover letter further)
        Day 6:   (research company deeply)
        Day 7:   (apply + connect with someone at the company on LinkedIn)

        ## 🎤 Likely Interview Questions
        1. (question based on a gap they might probe)
        2. (question about a key requirement)
        3. (question about a strength to highlight)
        4. (behavioral question based on the role)

        Then call save_report with:
        - filename = "job_application_report.md"
        - content = the full report above

        Confirm the file was saved with its path.
    """,
    tools=[save_report],
)


# ── Root Agent ─────────────────────────────────────────────────────────────────

root_agent = SequentialAgent(
    name="JobApplicationPipeline",
    description=(
        "Parallel job application analyzer. Provide a job description file "
        "and your resume file. Gets skill match, gap analysis, salary insights, "
        "and a cover letter draft all at once, then produces an action plan."
    ),
    sub_agents=[
        input_agent,              # reads JD + resume
        parallel_analysis_agent,  # 4 agents run simultaneously
        action_plan_agent,        # synthesizes final report
    ],
)