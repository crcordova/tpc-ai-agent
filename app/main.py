from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any
from app.models import QuoteRequest, QuoteResponse
from app.llamaindex_agent import LlamaIndexPricingAgent

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pricing Agent API",
    description="AI-powered commodity pricing agent using Monte Carlo forecasts",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el agente
pricing_agent = LlamaIndexPricingAgent()

@app.on_event("startup")
async def startup_event():
    """Inicializa el agente al arrancar la aplicación"""
    logger.info("Starting Pricing Agent API...")
    await pricing_agent.initialize()
    logger.info("Pricing Agent initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Pricing Agent API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with more details"""
    return {
        "status": "healthy",
        "agent_ready": pricing_agent.is_ready(),
        "available_tools": pricing_agent.get_available_tools()
    }

@app.post("/quote", response_model=QuoteResponse)
async def generate_quote(request: QuoteRequest):
    """
    Genera una cotización inteligente basada en los componentes solicitados
    """
    try:
        logger.info(f"Processing quote request for client: {request.client}")
        logger.info(f"Components: {[f'{c.commodity}({c.percentage}%)' for c in request.components]}")
        
        # Validar que los porcentajes sumen 100
        total_percentage = sum(component.percentage for component in request.components)
        if abs(total_percentage - 100.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Los porcentajes deben sumar 100%. Suma actual: {total_percentage}%"
            )
        
        # Procesar la cotización a través del agente
        quote_result = await pricing_agent.process_quote(request)
        
        logger.info(f"Quote generated successfully for client: {request.client}")
        return quote_result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing quote: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando la cotización: {str(e)}"
        )

@app.get("/commodities/{commodity}/forecast")
async def get_commodity_forecast(commodity: str, days: int = 5):
    """
    Obtiene el forecast de Monte Carlo para un commodity específico
    """
    try:
        forecast = await pricing_agent.get_commodity_forecast(commodity, days)
        return forecast
    except Exception as e:
        logger.error(f"Error getting forecast for {commodity}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo forecast para {commodity}: {str(e)}"
        )

@app.get("/commodities/{commodity}/price")
async def get_commodity_price(commodity: str):
    """
    Obtiene la predicción de precio para un commodity específico
    """
    try:
        price_data = await pricing_agent.get_commodity_price(commodity)
        return price_data
    except Exception as e:
        logger.error(f"Error getting price for {commodity}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo precio para {commodity}: {str(e)}"
        )

@app.get("/commodities/{commodity}/volatility")
async def get_commodity_volatility(commodity: str):
    """
    Obtiene la predicción de volatilidad para un commodity específico
    """
    try:
        volatility_data = await pricing_agent.get_commodity_volatility(commodity)
        return volatility_data
    except Exception as e:
        logger.error(f"Error getting volatility for {commodity}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo volatilidad para {commodity}: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)