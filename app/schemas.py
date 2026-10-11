from pydantic import BaseModel, Field
from typing import Optional

class InspectionResponse(BaseModel):
    filename: str = Field(..., description="Uploaded image file name")
    label: str = Field(..., description="Classification result: PASS or DEFECT")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")
    has_defect: bool = Field(..., description="Boolean flag for defect status")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds")
    attention_map_b64: Optional[str] = Field(None, description="Base64 encoded visual attention heatmap overlay")

class HealthCheckResponse(BaseModel):
    status: str
    service: str