from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import markdown
from main import ChromaQueryHandler


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

query_handler = ChromaQueryHandler(
    chroma_path=r"C:\Users\amank\agrichat-annam\RAG pipeline v2\ChromaDb",
    model_name="gemma3:4b",
    base_url="http://localhost:11434/v1"
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/query", response_class=HTMLResponse)
async def query(request: Request, question: str = Form(...)):
    raw_answer = query_handler.get_answer(question)
    print("\n--- Raw Answer (Markdown) ---\n")
    print(raw_answer)
    if "</think>" in raw_answer:
        answer_only = raw_answer.split("</think>")[-1].strip()
    else:
        answer_only = raw_answer.strip()
    html_answer = markdown.markdown(answer_only, extensions=["extra", "nl2br"])
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": html_answer,
        "question": question
    })

