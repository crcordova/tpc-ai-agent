from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class Component(BaseModel):
    commodity: str = Field(..., description="Nombre del commodity (ej: 'copper', 'zinc')")
    percentage: float = Field(..., ge=0.1, le=100.0, description="Porcentaje del componente en el portafolio")
    
    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        return v.capitalize().strip()

class QuoteRequest(BaseModel):
    client: str = Field(..., min_length=1, description="Nombre del cliente")
    components: List[Component] = Field(..., min_items=1, description="Lista de componentes del portafolio")
    
    @field_validator("components")
    @classmethod
    def validate_total_percentage(cls, v: List[Component]) -> List[Component]:
        total = sum(component.percentage for component in v)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Los porcentajes deben sumar 100%. Total actual: {total}%")
        return v

class ComponentAnalysis(BaseModel):
    commodity: str
    percentage: float
    current_price: float
    forecast_mean: float
    forecast_upper: float
    forecast_lower: float
    volatility: float
    trending: str

class RiskAssessment(BaseModel):
    level: str = Field(..., pattern=r'^(LOW|MEDIUM|HIGH)$')
    average_volatility: float
    max_volatility: float
    portfolio_correlation: str

class QuoteResponse(BaseModel):
    client: str
    timestamp: datetime
    weighted_price: float
    confidence: float = Field(..., ge=0.0, le=100.0)
    risk_assessment: RiskAssessment
    components: List[ComponentAnalysis]
    recommendations: List[str]
    agent_analysis: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }