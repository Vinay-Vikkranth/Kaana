# main.py — FastAPI application

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_vector_store = None
_user_encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vector_store, _user_encoder
    logger.info("Starting Kaana — seeding vector store...")
    from models import UserEncoder
    from seed_runner import seed
    _vector_store = seed()
    _user_encoder = UserEncoder()
    logger.info("Kaana ready. 70 restaurants loaded across 4 islands.")
    yield
    logger.info("Kaana shutting down.")


app = FastAPI(title="Kaana Restaurant Recommender", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class TasteID(BaseModel):
    # Todd's 80/20 Pareto dimensions
    dietary:      str = Field(default="no_restrictions",
                              description="no_restrictions|vegetarian|vegan|pescatarian|gluten_free")
    occasion:     str = Field(default="friends_group",
                              description="romantic_date|friends_group|family_kids|solo_explorer|special_occasion|adventurous")
    cuisine:      str = Field(default="local_hawaiian",
                              description="local_hawaiian|japanese_pacific|modern_fusion|comfort_casual|healthy_light|world_cuisine")
    atmosphere:   str = Field(default="beach_outdoor",
                              description="beach_outdoor|upscale_elegant|lively_social|cozy_local|scenic_unique")
    budget:       int = Field(default=2, ge=1, le=4)
    noise_pref:   int = Field(default=3, ge=1, le=4)
    island:       str = Field(default="any",
                              description="any|oahu|maui|big_island|kauai")
    current_time: int = Field(default=12, ge=0, le=23)


@app.get("/health")
async def health():
    count = _vector_store.count() if _vector_store else 0
    return {"status": "ok", "restaurants_loaded": count}


@app.get("/restaurants")
async def list_restaurants():
    if not _vector_store:
        raise HTTPException(status_code=503, detail="Not ready")
    return _vector_store.get_all()


@app.post("/recommend")
async def recommend_restaurants(taste_id: TasteID):
    if not _vector_store or not _user_encoder:
        raise HTTPException(status_code=503, detail="Not ready")
    try:
        from recommender import recommend
        results = recommend(taste_id.model_dump(), _vector_store, _user_encoder)
        return JSONResponse(content={"recommendations": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Recommend failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/onboarding.html")
async def serve_onboarding():
    return FileResponse("static/onboarding.html")

@app.get("/onboarding_v2.html")
async def serve_onboarding_v2():
    return FileResponse("static/onboarding_v2.html")

@app.get("/selector.html")
async def serve_selector():
    return FileResponse("static/selector.html")

@app.get("/onboarding_v3.html")
async def serve_onboarding_v3():
    return FileResponse("static/onboarding_v3.html")

@app.get("/onboarding_v4.html")
async def serve_onboarding_v4():
    return FileResponse("static/onboarding_v4.html")

@app.get("/onboarding_v5.html")
async def serve_onboarding_v5():
    return FileResponse("static/onboarding_v5.html")

@app.get("/onboarding_v6.html")
async def serve_onboarding_v6():
    return FileResponse("static/onboarding_v6.html")
