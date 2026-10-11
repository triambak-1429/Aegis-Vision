from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import InspectionResponse, HealthCheckResponse
from app.inference import AegisVisionEngine

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = AegisVisionEngine()
    yield
    engine = None

app = FastAPI(
    title="Aegis-Vision — ViT Surface Defect Inspection Service",
    description="Vision Transformer microservice delivering sub-20ms industrial defect detection.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return {"status": "healthy", "service": "aegis-vision"}

@app.post("/inspect", response_model=InspectionResponse)
async def inspect_surface(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")
        
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    return engine.inspect(contents, filename=file.filename)