import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import statistics
import json

from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.tools import tool

from app.config import GROQ_API_KEY, LLM_MODEL
from app.models import QuoteRequest, QuoteResponse, ComponentAnalysis, RiskAssessment
from app.tools.forecast import GetMonteCarloForecast, getForecastPrice, getForecastVolatility

logger = logging.getLogger(__name__)

class PricingAgent:
    def __init__(self):
        self.llm = None
        self.agent_executor = None
        self.tools = []
        self._ready = False

    async def initialize(self):
        """Inicializa el agente LangChain con Groq"""
        try:
            # Configurar LLM con Groq
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=LLM_MODEL,
                temperature=0.1,
                max_tokens=4000
            )
            
            # Configurar herramientas como Tool objects
            self.tools = [
                Tool(
                    name="get_monte_carlo_forecast",
                    description="Get Monte Carlo simulation forecast for a commodity. Input should be JSON with 'commodity' (string) and 'days' (integer, default 5)",
                    func=self._call_monte_carlo_forecast
                ),
                Tool(
                    name="get_forecast_price", 
                    description="Get price forecast for a commodity. Input should be JSON with 'commodity' (string)",
                    func=self._call_forecast_price
                ),
                Tool(
                    name="get_forecast_volatility",
                    description="Get volatility forecast for a commodity. Input should be JSON with 'commodity' (string)", 
                    func=self._call_forecast_volatility
                )
            ]
            
            # Crear prompt ReAct
            prompt = PromptTemplate.from_template(self._get_react_prompt())
            
            # Crear agente ReAct
            agent = create_react_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=15,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            self._ready = True
            logger.info("PricingAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PricingAgent: {str(e)}")
            raise

    def _get_react_prompt(self) -> str:
        """Prompt template para ReAct agent"""
        return """You are a commodity pricing expert analyzing portfolio components. Use the available tools to gather data and provide comprehensive analysis.

TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    def _call_monte_carlo_forecast(self, input_str: str) -> str:
        """Wrapper para GetMonteCarloForecast"""
        try:
            input_data = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = GetMonteCarloForecast.invoke(input_data)
            return json.dumps(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_forecast_price(self, input_str: str) -> str:
        """Wrapper para getForecastPrice"""
        try:
            input_data = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = getForecastPrice.invoke(input_data)
            return json.dumps(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_forecast_volatility(self, input_str: str) -> str:
        """Wrapper para getForecastVolatility"""
        try:
            input_data = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = getForecastVolatility.invoke(input_data)
            return json.dumps(result)
        except Exception as e:
            return f"Error: {str(e)}"

    async def process_quote(self, request: QuoteRequest) -> QuoteResponse:
        """Procesa una solicitud de cotización completa"""
        if not self._ready:
            raise RuntimeError("Agent not initialized")
        
        try:
            # Construir input para el agente
            input_text = self._build_agent_input(request)
            
            # Ejecutar agente
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": input_text}
            )
            
            # Procesar resultado y construir respuesta estructurada
            quote_response = await self._process_agent_result(request, result)
            
            return quote_response
            
        except Exception as e:
            logger.error(f"Error processing quote: {str(e)}")
            raise

    def _build_agent_input(self, request: QuoteRequest) -> str:
        """Construye el input para el agente LangChain"""
        components_text = []
        for comp in request.components:
            components_text.append(f"- {comp.commodity}: {comp.percentage}%")
        
        return f"""
SOLICITUD DE COTIZACIÓN:
Cliente: {request.client}
Componentes del portafolio:
{chr(10).join(components_text)}

TAREAS:
1. Para cada commodity ({', '.join([c.commodity for c in request.components])}):
   - Obtén forecast Monte Carlo para 5 días
   - Obtén predicción de precio
   - Obtén forecast de volatilidad

2. Calcula precio ponderado final usando los porcentajes especificados

3. Evalúa el riesgo del portafolio considerando:
   - Volatilidades individuales
   - Correlaciones implícitas
   - Trending de cada commodity

4. Genera recomendaciones estratégicas

Proporciona análisis detallado y cuantitativo usando los datos obtenidos.
"""

    async def _process_agent_result(self, request: QuoteRequest, agent_result: Dict) -> QuoteResponse:
        """Procesa el resultado del agente y construye la respuesta estructurada"""
        
        # Obtener datos detallados de cada commodity para construir respuesta estructurada
        component_analyses = []
        weighted_price_sum = 0.0
        volatilities = []
        
        for component in request.components:
            try:
                # Obtener datos de cada commodity
                forecast_data = await asyncio.to_thread(
                    GetMonteCarloForecast.invoke,
                    {"commodity": component.commodity, "days": 5}
                )
                
                price_data = await asyncio.to_thread(
                    getForecastPrice.invoke,
                    {"commodity": component.commodity}
                )
                
                volatility_data = await asyncio.to_thread(
                    getForecastVolatility.invoke,
                    {"commodity": component.commodity}
                )
                
                # Crear análisis del componente
                component_analysis = ComponentAnalysis(
                    commodity=component.commodity,
                    percentage=component.percentage,
                    current_price=price_data["close"],
                    forecast_mean=forecast_data["mean"],
                    forecast_upper=forecast_data["upper"],
                    forecast_lower=forecast_data["lower"],
                    volatility=volatility_data["volatility_predict_5d"],
                    trending=volatility_data["trending"]
                )
                
                component_analyses.append(component_analysis)
                
                # Calcular contribución al precio ponderado
                weighted_contribution = forecast_data["mean"] * (component.percentage / 100.0)
                weighted_price_sum += weighted_contribution
                
                volatilities.append(volatility_data["volatility_predict_5d"])
                
            except Exception as e:
                logger.error(f"Error processing component {component.commodity}: {str(e)}")
                # Crear análisis con datos por defecto en caso de error
                component_analysis = ComponentAnalysis(
                    commodity=component.commodity,
                    percentage=component.percentage,
                    current_price=0.0,
                    forecast_mean=0.0,
                    forecast_upper=0.0,
                    forecast_lower=0.0,
                    volatility=0.0,
                    trending="unknown"
                )
                component_analyses.append(component_analysis)
        
        # Calcular métricas de riesgo
        avg_volatility = statistics.mean(volatilities) if volatilities else 0.0
        max_volatility = max(volatilities) if volatilities else 0.0
        
        # Determinar nivel de riesgo
        risk_level = "LOW"
        if avg_volatility > 0.15:
            risk_level = "HIGH"
        elif avg_volatility > 0.08:
            risk_level = "MEDIUM"
        
        # Evaluar confianza basada en datos disponibles
        confidence = 85.0  # Base
        if len(volatilities) < len(request.components):
            confidence -= 20.0  # Penalizar por datos faltantes
        
        # Crear evaluación de riesgo
        risk_assessment = RiskAssessment(
            level=risk_level,
            average_volatility=avg_volatility,
            max_volatility=max_volatility,
            portfolio_correlation="medium"  # Simplificado por ahora
        )
        
        # Generar recomendaciones básicas
        recommendations = self._generate_recommendations(component_analyses, risk_assessment)
        
        # Construir respuesta final
        quote_response = QuoteResponse(
            client=request.client,
            timestamp=datetime.utcnow(),
            weighted_price=round(weighted_price_sum, 2),
            confidence=confidence,
            risk_assessment=risk_assessment,
            components=component_analyses,
            recommendations=recommendations,
            agent_analysis=agent_result.get("output", "Analysis completed")
        )
        
        return quote_response

    def _generate_recommendations(self, components: List[ComponentAnalysis], risk: RiskAssessment) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones basadas en riesgo
        if risk.level == "HIGH":
            recommendations.append("⚠️ Portafolio de alto riesgo. Considere diversificar más o reducir exposición.")
        elif risk.level == "MEDIUM":
            recommendations.append("📊 Riesgo moderado. Monitoree volatilidades de cerca.")
        else:
            recommendations.append("✅ Portafolio de bajo riesgo con exposición controlada.")
        
        # Recomendaciones por trending
        increasing_commodities = [c for c in components if "increase" in c.trending.lower()]
        decreasing_commodities = [c for c in components if "decrease" in c.trending.lower()]
        
        if increasing_commodities:
            commodities_names = ", ".join([c.commodity for c in increasing_commodities])
            recommendations.append(f"📈 Volatilidad creciente en: {commodities_names}. Considere cobertura adicional.")
        
        if decreasing_commodities:
            commodities_names = ", ".join([c.commodity for c in decreasing_commodities])
            recommendations.append(f"📉 Volatilidad decreciente en: {commodities_names}. Oportunidad de optimización.")
        
        # Recomendación temporal
        recommendations.append("⏰ Revisión recomendada en 3-5 días hábiles dada la naturaleza volátil del mercado.")
        
        return recommendations

    async def get_commodity_forecast(self, commodity: str, days: int = 5) -> Dict[str, Any]:
        """Obtiene forecast para un commodity específico"""
        return await asyncio.to_thread(
            GetMonteCarloForecast.invoke,
            {"commodity": commodity, "days": days}
        )

    async def get_commodity_price(self, commodity: str) -> Dict[str, Any]:
        """Obtiene predicción de precio para un commodity específico"""
        return await asyncio.to_thread(
            getForecastPrice.invoke,
            {"commodity": commodity}
        )

    async def get_commodity_volatility(self, commodity: str) -> Dict[str, Any]:
        """Obtiene predicción de volatilidad para un commodity específico"""
        return await asyncio.to_thread(
            getForecastVolatility.invoke,
            {"commodity": commodity}
        )

    def is_ready(self) -> bool:
        """Verifica si el agente está listo"""
        return self._ready

    def get_available_tools(self) -> List[str]:
        """Obtiene lista de herramientas disponibles"""
        return [tool.name for tool in self.tools] if self.tools else []