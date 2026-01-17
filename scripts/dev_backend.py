from fastapi import FastAPI
from backend.api import admin_routes, debate_routes

app = FastAPI(title="Cygnus Dev Backend")
app.include_router(admin_routes.router)
app.include_router(debate_routes.router)

@app.get('/api/status')
async def status():
    return {"ok": True, "scenarios_loaded": 0}
