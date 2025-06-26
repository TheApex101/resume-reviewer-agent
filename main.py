from crewai import Crew, Agent, Task
from langchain.tools import tool
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env with your OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("Loaded key:", "✔️" if api_key else "❌ Missing")

# Load input files
with open("resume.txt", "r") as f:
    resume_text = f.read()

with open("job_description.txt", "r") as f:
    job_description = f.read()

# Set up LLM
llm = OpenAI(openai_api_key=api_key, temperature=0.4, model="gpt-4")

# Define the Resume Reviewer Agent
resume_reviewer = Agent(
    role="Resume Reviewer",
    goal="Improve the resume to better match the job description",
    backstory=(
        "You are a resume reviewer agent specializing in optimizing resumes for job applications. "
        "You understand tone, clarity, structure, and keyword relevance."
    ),
    llm=llm
)

# Define the Task
review_task = Task(
    description=(
        "Review the resume and compare it to the job description. "
        "Suggest improvements in structure, tone, and keyword relevance."
    ),
    expected_output=(
        "A list of improvement suggestions categorized by structure, tone, and keyword alignment."
    ),
    agent=resume_reviewer
)


# Crew execution
crew = Crew(
    agents=[resume_reviewer],
    tasks=[review_task],
    verbose=True
)

result = crew.kickoff()
print("\n🔍 Suggestions:\n", result)
