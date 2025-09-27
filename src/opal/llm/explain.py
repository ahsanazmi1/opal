"""
Azure OpenAI integration for Opal consumer explanation generation.

This module provides LLM-powered explanations for consumer payment instrument
selection and counter-negotiation decisions.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConsumerExplanationRequest(BaseModel):
    """Request for consumer explanation generation."""
    
    selected_instrument: Dict[str, Any] = Field(..., description="Selected consumer instrument details")
    rejected_instruments: List[Dict[str, Any]] = Field(..., description="Rejected instruments")
    merchant_proposal: Dict[str, Any] = Field(..., description="Merchant proposal details")
    consumer_preferences: Dict[str, Any] = Field(default_factory=dict, description="Consumer preferences")
    transaction_context: Dict[str, Any] = Field(..., description="Transaction context")
    ml_value_scores: Dict[str, float] = Field(..., description="ML value scores for instruments")
    explanation_type: str = Field(default="instrument_selection", description="Type of explanation")


class ConsumerExplanationResponse(BaseModel):
    """Response from consumer explanation generation."""
    
    explanation: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., description="Explanation confidence", ge=0.0, le=1.0)
    key_factors: List[str] = Field(..., description="Key factors in decision")
    structured_analysis: Dict[str, Any] = Field(..., description="Structured analysis data")
    model_provenance: Dict[str, str] = Field(..., description="Model provenance information")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    tokens_used: int = Field(..., description="Tokens used in generation")


class OpalLLMExplainer:
    """Azure OpenAI-based consumer explanation generator."""
    
    def __init__(self):
        self.client = None
        self.is_available = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available")
            return
        
        # Get configuration from environment
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "opal-llm")
        
        if endpoint and api_key:
            try:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                self.is_available = True
                logger.info(f"✅ Azure OpenAI client initialized with deployment: {deployment}")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
                self.is_available = False
        else:
            logger.warning("Azure OpenAI not configured - missing endpoint or API key")
    
    def is_configured(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return self.is_available and self.client is not None
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            "status": "configured" if self.is_configured() else "not_configured",
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "opal-llm"),
            "api_key": "***" if os.getenv("AZURE_OPENAI_API_KEY") else None,
        }
    
    def explain_consumer_choice(self, request: ConsumerExplanationRequest) -> Optional[ConsumerExplanationResponse]:
        """Generate explanation for consumer instrument selection."""
        if not self.is_configured():
            logger.warning("Azure OpenAI not configured, using fallback explanation")
            return self._generate_fallback_explanation(request)
        
        start_time = datetime.now()
        
        try:
            # Build the prompt
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_explanation_prompt(request)
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "opal-llm"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            raw_response = response.choices[0].message.content
            
            # Parse the response
            explanation_data = self._parse_explanation_response(raw_response)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConsumerExplanationResponse(
                explanation=explanation_data.get("explanation", raw_response),
                confidence=explanation_data.get("confidence", 0.8),
                key_factors=explanation_data.get("key_factors", []),
                structured_analysis=explanation_data.get("structured_analysis", {}),
                model_provenance={
                    "model_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "opal-llm"),
                    "provider": "azure_openai",
                    "status": "active",
                    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                },
                processing_time_ms=processing_time,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
        except Exception as e:
            logger.error(f"Error generating consumer explanation: {e}")
            return self._generate_fallback_explanation(request)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for consumer explanation generation."""
        return """You are an expert financial advisor for Opal, the Open Payment Agent.
Your task is to generate clear, helpful explanations for consumer payment instrument selection.

CRITICAL REQUIREMENTS:
1. Explain WHY the selected instrument provides the best value
2. Compare against rejected alternatives with specific reasoning
3. Highlight key factors: rewards, fees, loyalty benefits, convenience
4. Use consumer-friendly language (avoid jargon)
5. Be specific about dollar amounts and percentages

JSON SCHEMA FOR RESPONSE:
{
  "explanation": "string - Clear explanation of why this instrument was chosen (max 200 words)",
  "confidence": "number - Confidence score between 0.0 and 1.0",
  "key_factors": ["string"] - List of 3-5 most important factors
}

Always respond with valid JSON matching this schema."""
    
    def _build_explanation_prompt(self, request: ConsumerExplanationRequest) -> str:
        """Build the user prompt for explanation generation."""
        selected = request.selected_instrument
        rejected = request.rejected_instruments
        merchant = request.merchant_proposal
        context = request.transaction_context
        
        return f"""Generate a clear explanation for this consumer payment choice:

SELECTED INSTRUMENT: {selected.get('instrument_type', 'Unknown')} ({selected.get('provider', 'Unknown')})
- Rewards: {selected.get('total_reward_value', 0):.2f} ({selected.get('rewards', [{}])[0].get('rate', 0)*100:.1f}%)
- Fees: {selected.get('out_of_pocket_cost', 0):.2f}
- Loyalty Tier: {selected.get('loyalty_tier', 'Standard')}
- Value Score: {request.ml_value_scores.get(selected.get('instrument_id', ''), 0):.3f}

REJECTED ALTERNATIVES:
{self._format_rejected_instruments(rejected, request.ml_value_scores)}

MERCHANT PROPOSAL: {merchant.get('rail_type', 'Unknown')} at {merchant.get('merchant_cost', 0)} basis points

TRANSACTION: ${context.get('transaction_amount', 0):.2f} via {context.get('channel', 'unknown')} channel

CONSUMER PREFERENCES: {json.dumps(request.consumer_preferences, indent=2)}

Please explain why the selected instrument provides the best value for this consumer."""
    
    def _format_rejected_instruments(self, rejected: List[Dict], value_scores: Dict[str, float]) -> str:
        """Format rejected instruments for the prompt."""
        formatted = []
        for instrument in rejected[:3]:  # Limit to top 3 rejected
            formatted.append(
                f"- {instrument.get('instrument_type', 'Unknown')}: "
                f"{instrument.get('total_reward_value', 0):.2f} rewards, "
                f"Value Score: {value_scores.get(instrument.get('instrument_id', ''), 0):.3f}"
            )
        return "\n".join(formatted) if formatted else "None"
    
    def _parse_explanation_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse the LLM response."""
        try:
            # Try to extract JSON from the response
            if "```json" in raw_response:
                json_start = raw_response.find("```json") + 7
                json_end = raw_response.find("```", json_start)
                json_str = raw_response[json_start:json_end].strip()
            elif "{" in raw_response and "}" in raw_response:
                json_start = raw_response.find("{")
                json_end = raw_response.rfind("}") + 1
                json_str = raw_response[json_start:json_end]
            else:
                # Fallback to plain text
                return {
                    "explanation": raw_response,
                    "confidence": 0.7,
                    "key_factors": ["LLM-generated explanation"],
                    "structured_analysis": {}
                }
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            # Fallback to plain text
            return {
                "explanation": raw_response,
                "confidence": 0.7,
                "key_factors": ["LLM-generated explanation"],
                "structured_analysis": {}
            }
    
    def _generate_fallback_explanation(self, request: ConsumerExplanationRequest) -> ConsumerExplanationResponse:
        """Generate fallback explanation when LLM is not available."""
        selected = request.selected_instrument
        value_score = request.ml_value_scores.get(selected.get('instrument_id', ''), 0.5)
        
        # Generate contextual explanation
        instrument_type = selected.get('instrument_type', 'payment instrument')
        reward_value = selected.get('total_reward_value', 0)
        fee_cost = selected.get('out_of_pocket_cost', 0)
        
        if reward_value > 0:
            explanation = f"Selected {instrument_type} for maximum rewards of ${reward_value:.2f}"
            if fee_cost == 0:
                explanation += " with no additional fees."
            else:
                explanation += f" after ${fee_cost:.2f} in fees."
        else:
            explanation = f"Selected {instrument_type} as the most cost-effective option"
            if fee_cost == 0:
                explanation += " with no fees."
            else:
                explanation += f" with minimal fees of ${fee_cost:.2f}."
        
        # Add ML score context
        if value_score > 0.7:
            explanation += f" ML value analysis confirms this is an excellent choice (score: {value_score:.3f})."
        elif value_score > 0.5:
            explanation += f" ML value analysis supports this selection (score: {value_score:.3f})."
        
        return ConsumerExplanationResponse(
            explanation=explanation,
            confidence=0.6,
            key_factors=["rewards_optimization", "fee_minimization", "ml_value_score"],
            structured_analysis={
                "value_score": value_score,
                "reward_value": reward_value,
                "fee_cost": fee_cost,
                "explanation_type": "fallback"
            },
            model_provenance={
                "model_name": "fallback",
                "provider": "opal_deterministic",
                "status": "fallback_mode",
                "message": "Azure OpenAI not available, using deterministic explanation"
            },
            processing_time_ms=0,
            tokens_used=0
        )


# Global explainer instance
_explainer: Optional[OpalLLMExplainer] = None


def get_consumer_explainer() -> OpalLLMExplainer:
    """Get the global consumer explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = OpalLLMExplainer()
    return _explainer


def explain_consumer_instrument_choice(
    selected_instrument: Dict[str, Any],
    rejected_instruments: List[Dict[str, Any]],
    merchant_proposal: Dict[str, Any],
    consumer_preferences: Dict[str, Any],
    transaction_context: Dict[str, Any],
    ml_value_scores: Dict[str, float]
) -> Optional[ConsumerExplanationResponse]:
    """
    Generate LLM explanation for consumer instrument selection.
    
    Args:
        selected_instrument: Details of the selected instrument
        rejected_instruments: List of rejected instruments
        merchant_proposal: Merchant's rail proposal
        consumer_preferences: Consumer preferences
        transaction_context: Transaction context
        ml_value_scores: ML value scores for all instruments
        
    Returns:
        ConsumerExplanationResponse with explanation
    """
    request = ConsumerExplanationRequest(
        selected_instrument=selected_instrument,
        rejected_instruments=rejected_instruments,
        merchant_proposal=merchant_proposal,
        consumer_preferences=consumer_preferences,
        transaction_context=transaction_context,
        ml_value_scores=ml_value_scores,
        explanation_type="instrument_selection"
    )
    
    explainer = get_consumer_explainer()
    return explainer.explain_consumer_choice(request)


def is_consumer_llm_configured() -> bool:
    """Check if consumer LLM explanation service is configured."""
    explainer = get_consumer_explainer()
    return explainer.is_configured()