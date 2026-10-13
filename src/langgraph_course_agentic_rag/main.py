from dotenv import load_dotenv

load_dotenv()

from langgraph_course_agentic_rag.graph.graph import app

if __name__ == "__main__":
    print(app.invoke({"question": "What is agent memory?"}))