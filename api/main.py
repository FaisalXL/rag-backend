from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import Optional,List
from fastapi import Request

from dotenv import load_dotenv
from rag_index import build_rag_chain_from_file, _llm,build_rag_chain_from_documents, load_single_file

load_dotenv()

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/project/api
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold current RAG chain
rag_chain = None


class Query(BaseModel):
    question: str


@app.delete("/delete")
async def delete_file(request: Request):
    data = await request.json()
    filename = os.path.basename(data.get("filename", ""))
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        return {"message": f"🗑️ File '{filename}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_doc(files: List[UploadFile] = File(...)):
    """
    Upload one or many files (.txt/.pdf/.docx) and rebuild the RAG index.
    """
    global rag_chain

    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    all_docs = []

    for file in files:
        # ----- save file -----
        filename   = os.path.basename(file.filename)
        save_path  = os.path.join(upload_dir, filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ----- load into LangChain Document -----
        try:
            docs = load_single_file(save_path)   # uses the helper in rag_index.py
            all_docs.extend(docs)
            print(f"✅ Loaded {len(docs)} docs from {filename}")
        except Exception as e:
            raise HTTPException(status_code=400,
                detail=f"Failed to read {filename}: {str(e)}")

    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid documents uploaded.")

    # ----- rebuild RAG chain from all combined docs -----
    rag_chain = build_rag_chain_from_documents(all_docs)
    return {"message": f"📄 {len(files)} file(s) uploaded and indexed!"}

@app.post("/query")
async def query_rag(input: Query):
    global rag_chain

    if not input.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        if rag_chain:
            response = rag_chain.invoke({"query": input.question})
            return {
            "response": {
                "result": response["result"],
                "source_documents": [doc.page_content for doc in response["source_documents"]],
                }
            }
        else:
        # Fall back to vanilla LLM if no docs uploaded
            llm_chain = _llm()
            response = llm_chain.invoke(input.question)
            return {
                "response": {
                    "result": response,
                    "source_documents": []
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
