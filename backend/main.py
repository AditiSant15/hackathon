from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_agent import graph, SYSTEM_PROMPT, parse_response  # Import from your ai_agent.py

# Add this check here to validate the API key
from config import OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

app = FastAPI(title="Safespace AI Agent Backend")

class AskRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_agent(request: AskRequest):
    try:
        user_input = request.message
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", user_input)]}
        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)
        
        # Return JSON response as expected by the frontend
        return {"response": final_response or "I'm here to help. Can you tell me more?", "tool_called": tool_called_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)