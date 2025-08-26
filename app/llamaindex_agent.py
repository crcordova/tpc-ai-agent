import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import statistics
import json
import re

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

from app.config import GROQ_API_KEY, LLM_MODEL
from app.models import QuoteRequest, QuoteResponse, ComponentAnalysis, RiskAssessment
from app.tools.forecast import GetMonteCarloForecast, getForecastPrice, getForecastVolatility

logger = logging.getLogger(__name__)

class LlamaIndexPricingAgent:
    """Agente de cotización usando LlamaIndex con prompt engineering avanzado"""
    
    def __init__(self):
        self.llm = None
        self._ready = False

    async def initialize(self):
        """Inicializa LlamaIndex con Groq"""
        try:
            # Configurar LLM
            self.llm = Groq(
                model=LLM_MODEL,
                api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=3000
            )
            
            # Configurar Settings globales
            Settings.llm = self.llm
            
            self._ready = True
            logger.info("LlamaIndexPricingAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndexPricingAgent: {str(e)}")
            raise

    async def process_quote(self, request: QuoteRequest) -> QuoteResponse:
        """Procesa una solicitud de cotización"""
        if not self._ready:
            raise RuntimeError("Agent not initialized")
        
        try:
            # Paso 1: Recopilar todos los datos
            logger.info("Gathering commodity data...")
            commodity_data = await self._gather_commodity_data(request.components)
            
            # Paso 2: Construir prompt con todos los datos
            logger.info("Building comprehensive prompt...")
            prompt = self._build_comprehensive_prompt(request, commodity_data)
            
            # Paso 3: Generar análisis completo con LLM
            logger.info("Generating comprehensive analysis...")
            analysis_result = await self._generate_comprehensive_analysis(prompt)

            # analysis_result.self_generate_weighted_risk_analysis()

            # Paso 4: Construir respuesta estructurada
            logger.info("Building structured response...")
            quote_response = await self._build_quote_response(request, commodity_data, analysis_result)
            
            logger.info(f"Quote processed successfully for {request.client}")
            return quote_response
            
        except Exception as e:
            logger.error(f"Error processing quote: {str(e)}")
            raise

    async def _gather_commodity_data(self, components) -> Dict[str, Dict]:
        """Recopila todos los datos de commodities de forma directa"""
        commodity_data = {}
        
        for component in components:
            commodity = component.commodity
            logger.info(f"Gathering data for {commodity}...")
            
            try:
                # Obtener datos en paralelo
                forecast_task = asyncio.to_thread(
                    GetMonteCarloForecast.invoke,
                    {"commodity": commodity, "days": 5}
                )
                
                price_task = asyncio.to_thread(
                    getForecastPrice.invoke,
                    {"commodity": commodity}
                )
                
                volatility_task = asyncio.to_thread(
                    getForecastVolatility.invoke,
                    {"commodity": commodity}
                )
                
                # Esperar resultados
                forecast_data, price_data, volatility_data = await asyncio.gather(
                    forecast_task, price_task, volatility_task,
                    return_exceptions=True
                )
                
                # Validar resultados
                if isinstance(forecast_data, Exception):
                    logger.error(f"Forecast error for {commodity}: {forecast_data}")
                    forecast_data = self._get_default_forecast()
                
                if isinstance(price_data, Exception):
                    logger.error(f"Price error for {commodity}: {price_data}")
                    price_data = self._get_default_price()
                
                if isinstance(volatility_data, Exception):
                    logger.error(f"Volatility error for {commodity}: {volatility_data}")
                    volatility_data = self._get_default_volatility()
                
                commodity_data[commodity] = {
                    "percentage": component.percentage,
                    "forecast": forecast_data,
                    "price": price_data,
                    "volatility": volatility_data
                }
                
                logger.info(f"Successfully gathered data for {commodity}")
                
            except Exception as e:
                logger.error(f"Error gathering data for {commodity}: {str(e)}")
                commodity_data[commodity] = {
                    "percentage": component.percentage,
                    "forecast": self._get_default_forecast(),
                    "price": self._get_default_price(),
                    "volatility": self._get_default_volatility()
                }
        
        return commodity_data

    # def _build_comprehensive_prompt(self, request: QuoteRequest, commodity_data: Dict) -> str:
    #     """Construye un prompt comprehensivo y genérico con todos los datos"""

    #     # Cabecera de la solicitud
    #     header = f"""
    #         CLIENT REQUEST: {request.client}
    #         COMPOSITE: {request.components} ({', '.join([f"{c} {d['percentage']}%" for c, d in commodity_data.items()])})
    #             """.strip()

    #     # Secciones dinámicas
    #     current_prices_block = "CURRENT PRICES:\n"
    #     mc_block = "MONTE CARLO PRICE PROJECTIONS (5 days):\n"
    #     ml_block = "ML FORECAST PRICE RANGES (5 days):\n"
    #     vol_block = "VOLATILITY FORECAST (GARCH 5 days):\n"

    #     # Iterar cada commodity y construir string
    #     for commodity, data in commodity_data.items():
    #         name = commodity.upper()
    #         perc = data["percentage"]

    #         # Precios actuales
    #         price_data = data["price"]
    #         current_prices_block += f"- {name}: ${price_data.get('close', 0):.2f}\n"

    #         # Monte Carlo
    #         forecast = data["forecast"]
    #         mc_block += f"- {name}: ${forecast.get('lower', 0):.2f} - ${forecast.get('upper', 0):.2f} (mean ${forecast.get('mean', 0):.2f})\n"

    #         # ML Forecast
    #         ml_block += f"- {name}: ${price_data.get('lower_price', 0):.2f} - ${price_data.get('upper_price', 0):.2f}, Coverage: {price_data.get('coverage_rate', 0.85)*100:.1f}%\n"

    #         # Volatilidad
    #         vol_data = data["volatility"]
    #         vol_block += f"- {name}: Volatility: {vol_data.get('volatility_predict_5d', 0):.3f}, Trend: {vol_data.get('trending', 'neutral')}\n"

    #     # Calcular rangos agregados de la mezcla (como ya tienes)
    #     price_mix_rangemc = {"lower": 0, "upper": 0}
    #     price_mix_rangeml = {"lower": 0, "upper": 0}
    #     model_confidence = 0
    #     for comp, datos in commodity_data.items():
    #         pond = datos["percentage"] / 100
    #         price_mix_rangemc["lower"] += datos["forecast"]["lower"] * pond
    #         price_mix_rangemc["upper"] += datos["forecast"]["upper"] * pond
    #         price_mix_rangeml["lower"] += datos["price"]["lower_price"] * pond
    #         price_mix_rangeml["upper"] += datos["price"]["upper_price"] * pond
    #         coverage_rate = data["price"].get("coverage_rate", 0.55) 
    #         model_confidence += coverage_rate * pond

    #     mix_block = f"""
    #         WEIGHTED MIX PRICE ESTIMATES:
    #         - Monte Carlo range: ${price_mix_rangemc["lower"]:.2f} - ${price_mix_rangemc["upper"]:.2f}
    #         - ML Forecast range: ${price_mix_rangeml["lower"]:.2f} - ${price_mix_rangeml["upper"]:.2f}
    #             """.strip()

    #             # Unir todo
    #     prompt = f"""
    #         {header}

    #         {current_prices_block}

    #         {mc_block}

    #         {ml_block}

    #         {vol_block}

    #         {mix_block}

    #         REQUIRED RESPONSE FORMAT:
    #         [SECTION 1: JSON OUTPUT]
    #         {{
    #         "final_price": X.XX,
    #         "price_method_used": "Monte Carlo" | "ML Forecast",
    #         "risk_level": "LOW" | "MEDIUM" | "HIGH",
    #         "model_confidence": {model_confidence:.1f},
    #         "buy_decision": "BUY NOW" | "WAIT",
    #         "review_timeline_days": N,
    #         "component_analysis": {{
    #             "Cu": "short analysis",
    #             "Zn": "short analysis"
    #         }},
    #         "main_risk_factors": [
    #             "factor 1",
    #             "factor 2",
    #             "factor 3"
    #         ]
    #         }}

    #         [SECTION 2: EXECUTIVE SUMMARY]
    #         Concise professional paragraph suitable to share with client.
    #             """.strip()

    #     return prompt

    def _build_comprehensive_prompt(self, request: QuoteRequest, commodity_data: Dict) -> str:
        """Construye un prompt comprehensivo con todos los datos"""
        
        # Extraer datos para el prompt
        commodities = []
        percentages = []
        mc_ranges = []
        volatilities = []
        trendings = []
        forecast_ranges = []
        coverage_rates = []
        current_prices = []
        comodity_prices = {}
        for commodity, data in commodity_data.items():
            commodities.append(commodity.upper())
            percentages.append(f"{data['percentage']}%")
            # Datos Monte Carlo
            forecast = data['forecast']
            mc_ranges.append(f"${forecast.get('lower', 0):.2f} - ${forecast.get('upper', 0):.2f} (media: ${forecast.get('mean', 0):.2f})")
            
            # Volatilidad
            vol_data = data['volatility']
            volatilities.append(f"{vol_data.get('volatility_predict_5d', 0):.3f}")
            trendings.append(vol_data.get('trending', 'neutral'))
            
            # Pronóstico de precios
            price_data = data['price']
            forecast_ranges.append(f"${price_data.get('lower_price', 0):.2f} - ${price_data.get('upper_price', 0):.2f}")
            coverage_rates.append(f"{price_data.get('coverage_rate', 0.85)*100:.1f}%")
            current_prices.append(f"${price_data.get('close', 0):.2f}")
        
            comodity_prices[commodity.capitalize()] = {
                'poderation':data['percentage'], 
                'price_mc_upper':forecast.get('upper'),
                'price_mc_lower':forecast.get('lower'),
                'price_mc_mean':forecast.get('mean'),
                'price_ml_upper':price_data.get('upper_price'),
                'price_ml_lower':price_data.get('lower_price'),
                'volatility': vol_data.get('volatility_predict_5d'),
                'trending': vol_data.get('trending', 'neutral')
            }
        #TODO hacer formula correcta
        price_mix_rangemc = {"lower": 0, "upper": 0}
        price_mix_rangeml = {"lower": 0, "upper": 0}

        for comp, datos in comodity_prices.items():
            pond = datos["poderation"] / 100  # normalizar ponderación en %
            
            # Monte Carlo
            price_mix_rangemc["lower"] += datos["price_mc_lower"] * pond
            price_mix_rangemc["upper"] += datos["price_mc_upper"] * pond

            # Machine Learning
            price_mix_rangeml["lower"] += datos["price_ml_lower"] * pond
            price_mix_rangeml["upper"] += datos["price_ml_upper"] * pond

        weighted_analysis = self._generate_weighted_risk_analysis(comodity_prices)

        prompt = f"""
            SMART QUOTE REQUEST

            The client "{request.client}" requires a quote for a composite based on the following elements:

            COMPOSITION OF THE MIX:
            - Commodities: {', '.join(commodities)}
            - Respective weights: {', '.join(percentages)}
            - Current prices: {', '.join(current_prices)}

            MONTE CARLO ANALYSIS (5-day Horizon):
            - Projected ranges: {', '.join(mc_ranges)}

            VOLATILITY ANALYSIS:
            - Projected volatilities (5d): {', '.join(volatilities)}
            - Volatility trends: {', '.join(trendings)}

            PRICE FORECASTS:
            - Expected price ranges: {', '.join(forecast_ranges)}
            - Probabilities of success: {', '.join(coverage_rates)}

            ANALYSIS INSTRUCTIONS:

            1. PRICE
            - WEIGHTED PRICE: Calculate the final price of the composite using the specified weights and the Monte Carlo (mean) data.
            - WEIGHTED PRICE considering volatility: Calculate with the Machine Learning range data if market will up or down.

            2. RISK ASSESSMENT:
            - Analyze individual volatilities and their combined impact
            - Consider volatility trends (high increasing, increasing, neutral, decreasing, high decreasing)

            3. MODEL CONFIDENCE:
            - Weight the probabilities of success of each component
            - Adjust for data quality and market trends

            4. STRATEGIC RECOMMENDATIONS:
            - Identify key risks and opportunities
            - Suggest hedging strategies if necessary
            - Provide review timeline

            5. FINAL PRICE OF THE MIX COMPOSITE:
            given for the range {price_mix_rangemc["lower"]:.2f} - {price_mix_rangemc["upper"]:.2f} (Monte Carlo)
            or
            given for the range {price_mix_rangeml["lower"]:.2f} - {price_mix_rangeml["upper"]:.2f} (Machine Learning)

            6. RISK LEVEL:
            - Consider the volatility of the commodities, their trends and ponderation for return a global risk assessment.
            {weighted_analysis}

            REQUIRED RESPONSE FORMAT:
            [SECTION 1: JSON OUTPUT]

            FINAL PRICE OF THE MIX COMPOSITE mean: $X.XX,
            FINAL PRICE OF THE MIX COMPOSITE considering market volatility to be covered: $X.XX,

            RISK LEVEL: [LOW/MEDIUM/HIGH]

            MODEL CONFIDENCE: XX%

            [SECTION 2: Structure your response in the following clearly defined sections:]
            
            **EXECUTIVE SUMMARY:**
            [Concise paragraph with key observations about the mix composite]

            **COMPONENT ANALYSIS:**
            [Brief analysis of each commodity and its contribution to risk/return]

            **MAIN RISK FACTORS:**
            - [Factor 1]
            - [Factor 2]
            - [Factor 3]

            **RECOMMENDATIONS:**
            1. [Immediate tactical recommendation for each commodity, buy immediately or wait to buy]
            2. [Monitoring strategy]
            3. [Review timeline]

            Provides a quantitative, professional and actionable analysis based strictly on the data provided.
            """
        
        return prompt

    def _generate_weighted_risk_analysis(self, commodity_prices: Dict) -> str:
        """Genera análisis de riesgo ponderado detallado"""
        
        # Definir multiplicadores de tendencia
        trend_multipliers = {
            "high increasing": 2.0,
            "increasing": 1.5, 
            "neutral": 1.0,
            "decrease": 0.7,
            "decreasing": 0.7,
            "high decrease": 0.5,
            "high decreasing": 0.5
        }
        
        analysis_lines = ["DETAILED WEIGHTED RISK BREAKDOWN:"]
        total_weighted_risk = 0.0
        dominant_components = []
        
        for commodity, data in commodity_prices.items():
            weight = data['poderation'] / 100.0
            volatility = data['volatility']
            trending = data['trending'].lower()
            
            # Determinar multiplicador de tendencia
            multiplier = trend_multipliers.get(trending, 1.0)
            
            # Calcular contribución al riesgo
            risk_contribution = volatility * weight * multiplier
            total_weighted_risk += risk_contribution
            
            # Identificar componentes dominantes
            if weight >= 0.5:  # >= 50%
                dominant_components.append(f"{commodity} (DOMINANT {weight*100:.0f}%)")
            elif weight >= 0.3:  # >= 30%
                dominant_components.append(f"{commodity} (MAJOR {weight*100:.0f}%)")
            
            analysis_lines.append(
                f"  • {commodity}: {weight*100:.1f}% × {volatility:.3f} volatility × {multiplier:.1f} ({trending}) = {risk_contribution:.4f} risk points"
            )
        
        analysis_lines.append(f"")
        analysis_lines.append(f"TOTAL WEIGHTED PORTFOLIO RISK: {total_weighted_risk:.4f}")
        analysis_lines.append(f"DOMINANT/MAJOR COMPONENTS: {', '.join(dominant_components) if dominant_components else 'None (well diversified)'}")
        
        # Análisis de concentración
        if any(data['poderation'] >= 70 for data in commodity_prices.values()):
            high_weight_components = [
                f"{commodity} ({data['poderation']:.0f}%): {data['trending']}"
                for commodity, data in commodity_prices.items() 
                if data['poderation'] >= 70
            ]
            analysis_lines.append(f"HIGH CONCENTRATION ALERT: {', '.join(high_weight_components)}")
        
        # Predicción de nivel de riesgo basado en cálculos
        if total_weighted_risk > 0.15 or any(
            data['poderation'] >= 50 and 'high increasing' in data['trending'].lower()
            for data in commodity_prices.values()
        ):
            expected_risk = "HIGH"
        elif total_weighted_risk > 0.08 or any(
            data['poderation'] >= 30 and 'increasing' in data['trending'].lower()
            for data in commodity_prices.values()
        ):
            expected_risk = "MEDIUM"
        else:
            expected_risk = "LOW"
        
        analysis_lines.append(f"CALCULATED RISK EXPECTATION: {expected_risk}")
        
        return "\n".join(analysis_lines)

 
    async def _generate_comprehensive_analysis(self, prompt: str) -> str:
        """Genera análisis usando LlamaIndex"""
        try:
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    # content="You are an expert quantitative commodities analyst with 15 years of experience in portfolio risk management and will assist a client with a quote for several commodities. Your analysis must be accurate, quantitative, and data-driven. Follow the requested format exactly."
                    content='''You are an AI Pricing Executive for a chemical trading company that buys commodities (Cu, Zn, etc.) to produce compounds for clients. 
                            Your role is to prepare professional, data-driven price quotes, evaluate market risk, and decide optimal timing of purchases. 
                            You combine quantitative analysis (Monte Carlo, ML Forecasts, GARCH volatility) with strategic recommendations. 
                            Always provide clear, structured outputs that can be used directly in client communication and internal decision-making.
                            '''
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=prompt
                )
            ]
            
            response = await asyncio.to_thread(self.llm.chat, messages)
            return response.message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return f"Technical analysis completed {len(prompt.split('Commodities:')[1].split('Ponderaciones')[0].split(','))} commodities."

    async def _build_quote_response(self, request: QuoteRequest, 
                                   commodity_data: Dict, analysis: str) -> QuoteResponse:
        """Construye la respuesta estructurada final"""
        
        # Crear análisis de componentes
        component_analyses = []
        weighted_price_sum = 0.0
        volatilities = []
        
        for commodity, data in commodity_data.items():
            forecast = data['forecast']
            price = data['price']
            volatility = data['volatility']
            percentage = data['percentage']
            
            component_analysis = ComponentAnalysis(
                commodity=commodity,
                percentage=percentage,
                current_price=price.get('close', 0.0),
                forecast_mean=forecast.get('mean', 0.0),
                forecast_upper=forecast.get('upper', 0.0),
                forecast_lower=forecast.get('lower', 0.0),
                volatility=volatility.get('volatility_predict_5d', 0.0),
                trending=volatility.get('trending', 'neutral')
            )
            
            component_analyses.append(component_analysis)
            
            # Calcular precio ponderado
            weighted_contribution = forecast.get('mean', 0.0) * (percentage / 100.0)
            weighted_price_sum += weighted_contribution
            
            # Recopilar volatilidades
            if volatility.get('volatility_predict_5d', 0) > 0:
                volatilities.append(volatility.get('volatility_predict_5d', 0.0))
        
        # Evaluar riesgo
        risk_assessment = self._assess_risk(volatilities, analysis)
        
        # Extraer confianza del análisis (buscar patrón en el texto)
        confidence = self._extract_confidence_from_analysis(analysis)
        
        # Generar recomendaciones desde el análisis
        recommendations = self._extract_recommendations_from_analysis(analysis)

        recommended_price = self._extract_price_recomendation(analysis, "FINAL PRICE OF THE MIX COMPOSITE mean", round(weighted_price_sum, 2))

        return QuoteResponse(
            client=request.client,
            timestamp=datetime.utcnow(),
            # weighted_price=round(weighted_price_sum, 2),
            weighted_price=recommended_price,
            confidence=confidence,
            risk_assessment=risk_assessment,
            components=component_analyses,
            recommendations=recommendations,
            agent_analysis=analysis
        )

    def _assess_risk(self, volatilities: List[float], analysis: str) -> RiskAssessment:
        """Evalúa riesgo basado en volatilidades y análisis"""
        if not volatilities:
            avg_vol = 0.1
            max_vol = 0.1
        else:
            avg_vol = statistics.mean(volatilities)
            max_vol = max(volatilities)
        
        # Extraer nivel de riesgo del análisis
        analysis_lower = analysis.lower()
        if "high" in analysis_lower and "risk" in analysis_lower:
            risk_level = "HIGH"
        elif "low" in analysis_lower and ("risk" in analysis_lower or "risk" in analysis_lower):
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        return RiskAssessment(
            level=risk_level,
            average_volatility=avg_vol,
            max_volatility=max_vol,
            portfolio_correlation="medium"
        )

    def _extract_confidence_from_analysis(self, analysis: str) -> float:
        """Extrae nivel de confianza del análisis
        Extrae nivel de confianza buscando la línea con MODEL CONFIDENCE"""

        for line in analysis.splitlines():
            if "MODEL CONFIDENCE" in line.upper():

                match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                if match:
                    return float(match.group(1))
        return 75.0  # valor por defecto

    def _extract_price_recomendation(self, analysis: str, field="FINAL PRICE OF THE MIX COMPOSITE mean", default=100.0) -> float:
        """Extrae el valor de un campo buscando la línea con el nombre del campo"""
        for line in analysis.splitlines():
            if field.upper() in line.upper():
                import re
                # Buscar número con posible signo $ y decimales
                match = re.search(r'([-+]?\$?\d+(?:\.\d+)?)', line)
                if match:
                    value = match.group(1)
                    # Si tiene símbolo de dólar, lo limpiamos
                    value = value.replace("$", "")
                    try:
                        return float(value)
                    except ValueError:
                        return value  # devuelve string si no se puede convertir
        return default

    def _extract_recommendations_from_analysis(self, analysis: str) -> List[str]:
        """Extrae recomendaciones del análisis"""
        recommendations = []
        
        # Buscar sección de recomendaciones
        lines = analysis.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            
            if '*RECOMMENDATIONS*' in line.upper():
                in_recommendations = True
                continue
            
            if in_recommendations:
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                    recommendations.append(line[2:].strip() if line[1:2] == '.' else line[1:].strip())
                elif line.startswith('**') and line.endswith('**'):
                    break
                elif line:
                    recommendations.append(line)
        
        # Si no se encontraron recomendaciones, crear algunas por defecto
        if not recommendations:
            recommendations = [
                "Monitorear volatilidades diariamente",
                "Evaluar necesidad de cobertura adicional",
                "Revisión recomendada en 3-5 días hábiles"
            ]
        
        return recommendations[:5]  # Máximo 5 recomendaciones



    def _get_default_forecast(self) -> Dict:
        """Datos por defecto para forecast"""
        return {"mean": 100.0, "upper": 110.0, "lower": 90.0, "days": 5}

    def _get_default_price(self) -> Dict:
        """Datos por defecto para precio"""
        return {
            "close": 100.0,
            "upper_price": 105.0,
            "lower_price": 95.0,
            "coverage_rate": 0.85
        }

    def _get_default_volatility(self) -> Dict:
        """Datos por defecto para volatilidad"""
        return {
            "volatility_predict_5d": 0.10,
            "trending": "neutral"
        }

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
        return ["Direct Data Access", "Monte Carlo Analysis", "Volatility Forecasting"]