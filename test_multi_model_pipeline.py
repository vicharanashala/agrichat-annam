#!/usr/bin/env python3
"""
Multi-Model Agricultural AI Pipeline Test Script

This is a self-contained test harness that exercises the staged multi-model
pipeline (reasoner -> RAG -> structurer -> evaluator -> fallback).

Run: python3 test_multi_model_pipeline.py
"""

import re
import sys
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass

sys.path.append('/home/ubuntu/agrichat-annam/agrichat-backend')
sys.path.append('/home/ubuntu/agrichat-annam/agrichat-backend/Agentic_RAG')

# Bare imports from the Agentic_RAG folder and backend utilities
from chroma_query_handler import ChromaQueryHandler
from local_llm_interface import OllamaLLMInterface
from langchain.memory import ConversationBufferWindowMemory
from response_formatter import AgriculturalResponseFormatter

@dataclass
class QueryAnalysis:
    """Result of query completeness analysis"""
    is_complete: bool
    confidence: float
    missing_info: List[str]
    query_type: str  # 'direct_fact', 'complex_reasoning', 'personalized_advice'
    agri_relevance: float

@dataclass
class QualityAssessment:
    """Result of response quality evaluation"""
    is_satisfactory: bool
    completeness_score: float
    relevance_score: float
    actionability_score: float
    suggestions: List[str]

class MultiModelPipeline:
    """
    New Multi-Model Architecture Implementation
    Reuses existing ChromaDB, PoPs, and conversation memory
    """
    
    def __init__(self):
        print("Initializing Multi-Model Pipeline...")
        
        self.models = {
            'reasoner': 'qwen3:1.7b',    
            'structurer': 'gemma:latest',   
            'fallback': 'llama3.1:8b'    
        }
        
        self.reasoner_llm = OllamaLLMInterface(model_name=self.models['reasoner'])
        self.structurer_llm = OllamaLLMInterface(model_name=self.models['structurer'])
        self.fallback_llm = OllamaLLMInterface(model_name=self.models['fallback'])
        
        # Initialize response formatter
        self.formatter = AgriculturalResponseFormatter()
        
        print(f"PIPELINE MODEL ASSIGNMENT:")
        print(f"   Reasoner (qwen): {self.reasoner_llm.model_name}")
        print(f"   Structurer (gemma): {self.structurer_llm.model_name}")
        print(f"   Fallback (llama): {self.fallback_llm.model_name}")
        print(f"")
    
        chroma_path = os.getenv('CHROMA_DB_PATH')
        if not chroma_path:
            if os.path.exists('/app') and os.path.exists('/app/chromaDb'):
                chroma_path = '/app/chromaDb'
            else:
                chroma_path = '/home/ubuntu/agrichat-annam/agrichat-backend/chromaDb'

        print(f"[PIPELINE] Using ChromaDB path: {chroma_path}")
        self.chroma_handler = ChromaQueryHandler(chroma_path)
        self.conversation_memory = ConversationBufferWindowMemory(k=8)
        
        print(f"Models configured:")
        print(f"   Reasoner: {self.models['reasoner']}")
        print(f"   Structurer: {self.models['structurer']}")
        print(f"   Fallback: {self.models['fallback']}")
        print(f"Response Formatter: Enabled")
        print(f"Existing components loaded successfully!")

    async def analyze_query_completeness(self, query: str, chat_history: List[str]) -> QueryAnalysis:
        """
        Stage 1: Use qwen2:1.7b to analyze query completeness and classify it
        """
        print(f"Stage 1: Analyzing query completeness with {self.models['reasoner']}")
        
        history_context = "\n".join([f"User: {msg}" if i % 2 == 0 else f"Assistant: {msg[:100]}..." 
                                    for i, msg in enumerate(chat_history[-4:])])  # Last 4 messages
        
        prompt = f"""You are an agricultural expert assistant. Analyze this farmer's question for completeness and classify it.

Chat History: {history_context}

Current Question: "{query}"

CRITICAL GUIDELINES FOR COMPLETENESS:

MARK AS COMPLETE if the question is:
1. **Direct Facts**: "What is the seed rate of [crop]?", "How to control [pest/disease]?", "When to harvest [crop]?"
2. **General Practices**: "How to grow [crop]?", "What fertilizer for [crop]?", "Best varieties of [crop]?"
3. **Technical Questions**: "What causes [disease] in [crop]?", "How to prepare soil for [crop]?"

MARK AS INCOMPLETE only if:
1. **Vague Questions**: "What should I grow?", "Help with my crop problem" (no specific crop mentioned)
2. **Personal Advice Needing Context**: "What's best for my farm?" (no location/conditions given)

EXAMPLES:
- "What is seed rate of bitter gourd?" → COMPLETE (direct fact)
- "How to control aphids in cotton?" → COMPLETE (specific pest + crop)
- "When to plant tomatoes?" → COMPLETE (can give general timing)
- "What should I grow?" → INCOMPLETE (needs location/conditions)
- "My crop is dying, help!" → INCOMPLETE (vague, no crop specified)

Current Question: "{query}"

Respond in this format:
COMPLETENESS: [COMPLETE/INCOMPLETE]  
QUERY_TYPE: [direct_fact/complex_reasoning/personalized_advice]
CONFIDENCE: [0.0-1.0]
MISSING_INFO: [only list if question is truly vague and needs critical context]
AGRICULTURAL_RELEVANCE: [0.0-1.0]
REASONING: [brief explanation]"""

        try:
            response = self.reasoner_llm.generate_content(prompt, temperature=0.1)
            
            print(f"\nQWEN REASONING OUTPUT:")
            print(f"{response[:500]}..." if len(response) > 500 else response)
            print(f"")
            
            is_complete = "COMPLETE" in response and "INCOMPLETE" not in response
            confidence = 0.6
            query_type = "direct_fact"
            missing_info = []
            
            if "CONFIDENCE:" in response:
                try:
                    conf_text = response.split("CONFIDENCE:")[1].split("\n")[0].strip()
                    confidence = float(conf_text)
                except:
                    confidence = 0.6
            
            if "QUERY_TYPE:" in response:
                type_text = response.split("QUERY_TYPE:")[1].split("\n")[0].strip()
                query_type = type_text
            
            if not is_complete and "MISSING_INFO:" in response:
                missing_text = response.split("MISSING_INFO:")[1].split("\n")[0].strip()
                missing_info = [item.strip() for item in missing_text.split(",") if item.strip()]
            
            analysis = QueryAnalysis(
                is_complete=is_complete,
                confidence=confidence,
                missing_info=missing_info if not is_complete else [],
                query_type=query_type,
                agri_relevance=0.8
            )
            
            print(f"Analysis complete: {analysis.query_type} (confidence: {analysis.confidence:.2f})")
            if not analysis.is_complete:
                print(f"Missing information identified: {', '.join(analysis.missing_info)}")
            return analysis
            
        except Exception as e:
            print(f"Error in query analysis: {e}")
            return QueryAnalysis(True, 0.5, [], 'direct_fact', 0.8)

    async def get_rag_response_direct(self, query: str, chat_history: List[str], user_state: Optional[str] = None) -> Dict:
        """
        Direct RAG access with improved query context
        """
        print(f"Stage 2: Querying RAG databases with direct access")
        try:
            enhanced_query = query
            context_keywords = []
            location_info = []
            farming_context = []
            
            if chat_history:
                for msg in chat_history[-4:]: 
                    msg_lower = msg.lower()
                    
                    indian_states = ['punjab', 'haryana', 'uttar pradesh', 'bihar', 'west bengal', 
                                   'maharashtra', 'karnataka', 'tamil nadu', 'andhra pradesh', 
                                   'telangana', 'rajasthan', 'madhya pradesh', 'gujarat', 'kerala']
                    for state in indian_states:
                        if state in msg_lower:
                            location_info.append(state)
                    
                    farming_terms = ['soil', 'season', 'crop', 'rabi', 'kharif', 'summer', 
                                   'wheat', 'rice', 'cotton', 'nitrogen', 'fertilizer', 'alluvial', 
                                   'clay', 'sandy', 'loam', 'hectare', 'acre']
                    for term in farming_terms:
                        if term in msg_lower:
                            farming_context.append(term)
                
                if location_info or farming_context:
                    context_keywords = list(set(location_info + farming_context))
                    enhanced_query = f"{query} {' '.join(context_keywords[:5])}" 
            
            print(f"Enhanced query: {enhanced_query}")
            if context_keywords:
                print(f"Context extracted: {', '.join(context_keywords)}")
            
            queries_to_try = [
                enhanced_query,
                f"crop recommendation {' '.join(location_info)} {' '.join([fc for fc in farming_context if fc in ['rabi', 'kharif', 'summer']])}",
                f"what crops grow {' '.join(location_info)} {' '.join([fc for fc in farming_context if 'season' in fc or fc in ['rabi', 'kharif']])}",
                query  
            ]
            

            queries_to_try = [q.strip() for q in queries_to_try if q.strip() and len(q.strip()) > len(query)]
            queries_to_try.append(query)  # Always include original
            
            best_result = None
            best_score = 0
            
            for search_query in queries_to_try:
                docs = self.chroma_handler.db.similarity_search_with_score(search_query, k=5)

                # prefer documents that match the user's state if provided
                if docs and docs[0][1] < 0.5:  # Lower distance = better match
                    doc, distance = docs[0]
                    current_score = 1.0 - distance

                    # Boost score if the document metadata contains a matching state
                    try:
                            meta = getattr(doc, 'metadata', {}) or {}
                            doc_state = (meta.get('state') or meta.get('State') or '').lower()
                    except Exception:
                        doc_state = ''

                    if user_state and doc_state:
                        us = user_state.strip().lower()
                        if us in doc_state or doc_state in us:
                            # strong preference for same-state documents
                            current_score = min(1.0, current_score + 0.25)

                    # Exact-match shortcut: if the DB document contains a stored question and it
                    # matches the user query exactly (after normalization) AND the state matches,
                    # return verbatim database answer immediately
                    try:
                        stored_question = ''
                        # attempt to detect a 'Question:' or stored question in content/metadata
                        if hasattr(doc, 'page_content') and doc.page_content:
                            m = re.search(r"Question:\s*(.+)", doc.page_content, re.IGNORECASE)
                            if m:
                                stored_question = m.group(1).strip()
                        if not stored_question:
                            stored_question = (doc.metadata.get('question') or '')

                        def _norm(s: str) -> str:
                            return re.sub(r"\s+", " ", (s or '').strip().lower())

                        if stored_question and _norm(stored_question) == _norm(query):
                            # require state match as well for exact verbatim return
                            if not user_state or (doc_state and (user_state.strip().lower() in doc_state or doc_state in user_state.strip().lower())):
                                print(f"Exact stored-question + state match found for query '{query}'. Returning verbatim DB answer.")
                                return {
                                    'response': doc.page_content,
                                    'source': 'rag_direct',
                                    'similarity_score': 1.0,
                                    'metadata': doc.metadata,
                                    'distance': distance
                                }
                    except Exception:
                        pass

                    if current_score > best_score:
                        best_score = current_score
                        best_result = docs[0]
                        print(f"Better match found with query: '{search_query}' (score: {current_score:.3f})")
            
            if best_result and best_score > 0.6:  # Good match found
                best_doc, distance = best_result
                # If user_state provided, ensure the doc metadata state matches the user state
                try:
                        meta = getattr(best_doc, 'metadata', {}) or {}
                        doc_state = (meta.get('state') or meta.get('State') or '').lower()
                except Exception:
                    doc_state = ''

                if user_state:
                    us = user_state.strip().lower()
                    if doc_state and (us in doc_state or doc_state in us):
                        return {
                            'response': best_doc.page_content,
                            'source': 'rag_direct',
                            'similarity_score': best_score,
                            'metadata': best_doc.metadata,
                            'distance': distance
                        }
                    else:
                        print(f"[RAG] Best RAG doc state '{doc_state}' does not match user_state '{user_state}'; skipping RAG result.")
                        # skip RAG result so PoPs / LLM can be attempted
                        pass
                else:
                    return {
                        'response': best_doc.page_content,
                        'source': 'rag_direct',
                        'similarity_score': best_score,
                        'metadata': best_doc.metadata,
                        'distance': distance
                    }
            
            # Try PoPs database with enhanced queries
            if hasattr(self.chroma_handler, 'pops_db') and self.chroma_handler.pops_db:
                try:
                    for search_query in queries_to_try:
                        pops_docs = self.chroma_handler.pops_db.similarity_search_with_score(search_query, k=3)
                        if pops_docs and pops_docs[0][1] < 0.6:
                            best_doc, distance = pops_docs[0]
                            score = 1.0 - distance

                            try:
                                    meta = getattr(best_doc, 'metadata', {}) or {}
                                    pops_state = (meta.get('state') or meta.get('State') or '').lower()
                            except Exception:
                                pops_state = ''

                            # If user_state provided, require state match for PoPs; otherwise accept
                            if user_state:
                                us = user_state.strip().lower()
                                if pops_state and (us in pops_state or pops_state in us) and score > 0.5:
                                    print(f"PoPs state-matched result found for query: '{search_query}' (score: {score:.3f})")
                                    return {
                                        'response': best_doc.page_content,
                                        'source': 'pops_direct',
                                        'similarity_score': score,
                                        'metadata': best_doc.metadata,
                                        'distance': distance
                                    }
                                else:
                                    # skip non-matching PoPs when user_state is provided
                                    continue
                            else:
                                if score > 0.5:
                                    print(f"PoPs match found with query: '{search_query}' (score: {score:.3f})")
                                    return {
                                        'response': best_doc.page_content,
                                        'source': 'pops_direct',
                                        'similarity_score': score,
                                        'metadata': best_doc.metadata,
                                        'distance': distance
                                    }
                except Exception as e:
                    print(f"PoPs query failed: {e}")
            
            # Tier 3: LLM Fallback (using lighter model)
            print(f"Stage 2c: Using LLM fallback - {self.models['fallback']}")
            try:
                # Create context-aware fallback prompt
                context_from_history = ""
                context_details = ""
                
                if chat_history:
                    recent_context = " ".join(chat_history[-4:])
                    context_from_history = f"\nContext from conversation: {recent_context}"
                    
                    # Extract dynamic context from conversation
                    states = ['punjab', 'haryana', 'uttar pradesh', 'up', 'bihar', 'rajasthan', 'madhya pradesh', 'mp', 'maharashtra', 'karnataka', 'andhra pradesh', 'tamil nadu', 'gujarat', 'west bengal', 'odisha', 'chhattisgarh', 'jharkhand', 'assam', 'kerala', 'telangana']
                    seasons = ['rabi', 'kharif', 'summer', 'winter', 'monsoon', 'post-monsoon']
                    soils = ['alluvial', 'black', 'red', 'laterite', 'sandy', 'clay', 'loamy', 'saline', 'alkaline']
                    crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane', 'soybean', 'mustard', 'barley', 'gram', 'pea', 'potato', 'onion']
                    
                    detected_context = []
                    lower_context = recent_context.lower()
                    
                    for state in states:
                        if state in lower_context:
                            detected_context.append(f"location: {state}")
                            break
                    
                    for season in seasons:
                        if season in lower_context:
                            detected_context.append(f"season: {season}")
                            break
                            
                    for soil in soils:
                        if soil in lower_context:
                            detected_context.append(f"soil: {soil}")
                            break
                            
                    for crop in crops:
                        if crop in lower_context:
                            detected_context.append(f"crop: {crop}")
                            break
                    
                    if detected_context:
                        context_details = f"Based on the context, this appears to be about farming with {', '.join(detected_context)}."
                    else:
                        context_details = "Based on the context, this appears to be about general Indian farming practices."
                
                fallback_prompt = f"""You are an agricultural expert specializing in Indian farming. Answer this farming question:

Question: {query}{context_from_history}

{context_details}

Provide specific, actionable advice including:
1. Suitable crops and varieties for the given conditions
2. Expected yields and management practices  
3. Fertilizer and nutrient recommendations
4. Best practices for the specific region/season/soil type

Focus on practical guidance relevant to the farmer's situation."""

                llm_response = self.fallback_llm.generate_content(fallback_prompt, temperature=0.3)
                
                return {
                    'response': llm_response,
                    'source': 'llm_fallback',
                    'similarity_score': 0.6,  # Moderate confidence for general knowledge
                    'metadata': {'model': self.models['fallback']},
                    'distance': 0.4
                }
                
            except Exception as e:
                print(f"LLM fallback also failed: {e}")
                return {
                    'response': f"I apologize, but I'm having trouble accessing information about: {query}. Please try rephrasing your question or contact an agricultural expert.",
                    'source': 'error',
                    'similarity_score': 0.0,
                    'metadata': {},
                    'distance': 1.0
                }
                
        except Exception as e:
            print(f"Error in direct RAG query: {e}")
            return {
                'response': "Database query failed",
                'source': 'error', 
                'similarity_score': 0.0,
                'metadata': {},
                'distance': 1.0
            }

    async def structure_content(self, rag_response: Dict, query: str) -> str:
        """
        Stage 3: Use gemma:latest to structure the content with proper data handling
        """
        print(f"Stage 3: Structuring content with {self.models['structurer']}")
        
        # Check if we have good RAG data to work with
        has_good_data = (
            rag_response.get('source', '') != 'error' and 
            rag_response.get('similarity_score', 0) > 0.7 and 
            len(rag_response.get('response', '')) > 50
        )
        
        if has_good_data:
            prompt = f"""You are an agricultural content structuring specialist. You have found SPECIFIC data that answers the farmer's question. Present this information clearly and accurately.

Original Question: "{query}"

SPECIFIC DATA FOUND:
{rag_response['response']}

IMPORTANT: Use the specific data provided above. Do NOT make up general information. Present the exact facts found in the database in a clear, farmer-friendly format.

Structure the response with:
1. Direct answer using the specific data
2. Additional relevant details from the data  
3. Clear, organized presentation"""
        else:
            prompt = f"""You are an agricultural content structuring specialist. Limited specific data is available. Provide helpful general guidance.

Original Question: "{query}"

Available Information: {rag_response.get('response', 'Limited data available')}

Provide general agricultural guidance while clearly indicating the limitations."""
        
        try:
            structured_response = self.structurer_llm.generate_content(prompt, temperature=0.2)
            print(f"Content structured successfully")
            return structured_response
            
        except Exception as e:
            print(f"Error in content structuring: {e}")
            return rag_response.get('response', 'Unable to structure response.')

    async def evaluate_response_quality(self, query: str, response: str, rag_info: Dict) -> QualityAssessment:
        """
        Stage 4: Use qwen3:1.7b to evaluate response quality
        """
        print(f"Stage 4: Evaluating response quality with {self.models['reasoner']}")
        
        prompt = f"""You are an agricultural quality evaluator. Assess if this response adequately answers the farmer's question.

Original Question: "{query}"

Response to Evaluate: "{response}"

Rate the response quality from 1-10 and respond with just:
QUALITY_SCORE: [score]
SATISFACTORY: [YES/NO]
REASONING: [brief explanation]"""

        try:
            evaluation = self.reasoner_llm.generate_content(prompt, temperature=0.1)
            
            # Parse simple format instead of JSON
            score = 7.0
            is_satisfactory = True
            
            if "QUALITY_SCORE:" in evaluation:
                try:
                    score_text = evaluation.split("QUALITY_SCORE:")[1].split("\n")[0].strip()
                    score = float(score_text) / 10.0  # Convert to 0-1 scale
                except:
                    score = 0.7
            
            if "SATISFACTORY:" in evaluation:
                satisfactory_text = evaluation.split("SATISFACTORY:")[1].split("\n")[0].strip().upper()
                is_satisfactory = satisfactory_text.startswith("YES")
            
            assessment = QualityAssessment(
                is_satisfactory=is_satisfactory,
                completeness_score=score,
                relevance_score=score,
                actionability_score=score,
                suggestions=[]
            )
                
            print(f"Quality assessment: {'Satisfactory' if assessment.is_satisfactory else 'Needs improvement'}")
            return assessment
            
        except Exception as e:
            print(f"Error in quality evaluation: {e}")
            return QualityAssessment(True, 0.7, 0.7, 0.7, [])

    async def handle_incomplete_query(self, analysis: QueryAnalysis, original_query: str) -> str:
        """
        Handle incomplete queries by asking for specific clarification
        """
        # Extract crop/topic from the original query for more relevant messaging
        query_lower = original_query.lower()
        crop_mentioned = None
        crops = ['rice', 'wheat', 'cotton', 'tomato', 'potato', 'maize', 'corn', 'sugarcane', 'onion', 'soybean']
        for crop in crops:
            if crop in query_lower:
                crop_mentioned = crop
                break
        
        # Create dynamic clarification based on what's missing
        missing_info_text = ""
        if analysis.missing_info:
            missing_info_text = f"\n**Missing Information:**\n{chr(10).join(['• ' + info for info in analysis.missing_info])}\n"
        
        # Create context-aware intro
        if crop_mentioned:
            intro = f"""I need some additional information to give you the best advice about {crop_mentioned} cultivation for "{original_query}"."""
        else:
            intro = f"""I need some additional information to give you the best agricultural advice for "{original_query}"."""
        
        # Generate dynamic recommendations based on missing information
        recommendations = []
        info_explanations = []
        
        if analysis.missing_info:
            for missing in analysis.missing_info:
                missing_lower = missing.lower()
                if 'location' in missing_lower or 'state' in missing_lower or 'region' in missing_lower:
                    recommendations.append("• **Location**: Which state/region are you in? (Climate varies significantly across India)")
                    info_explanations.append("location-specific climate conditions")
                    
                elif 'soil' in missing_lower:
                    recommendations.append("• **Soil Type**: What type of soil do you have - sandy, clay, loamy, or mixed?")
                    info_explanations.append("soil-specific growing requirements")
                    
                elif 'season' in missing_lower:
                    recommendations.append("• **Season**: Which season are you planning to grow - Kharif (monsoon), Rabi (winter), or Summer?")
                    info_explanations.append("seasonal growing patterns")
                    
                elif 'space' in missing_lower or 'size' in missing_lower:
                    recommendations.append("• **Space Available**: How much area do you have - in square feet, acres, or container size?")
                    info_explanations.append("space-appropriate varieties and techniques")
                    
                elif 'water' in missing_lower:
                    recommendations.append("• **Water Source**: What water sources do you have available for irrigation?")
                    info_explanations.append("water management strategies")
                    
                elif 'climate' in missing_lower:
                    recommendations.append("• **Local Climate**: What are typical temperatures and rainfall in your area?")
                    info_explanations.append("climate-suitable varieties")
                    
                elif 'experience' in missing_lower:
                    recommendations.append("• **Experience Level**: Are you a beginner or do you have farming experience?")
                    info_explanations.append("appropriate difficulty level and guidance")

        # If no specific missing info, use general recommendations
        if not recommendations:
            recommendations = [
                "• **Location**: Which state/region are you in?",
                "• **Growing Space**: How much area do you have available?",
                "• **Experience**: Are you new to farming or have some experience?"
            ]
            info_explanations = ["location-specific advice", "space-appropriate techniques", "suitable guidance level"]

        recommendations_text = "\n".join(recommendations)
        
        # Generate dynamic explanation
        if crop_mentioned:
            why_text = f"**Why this helps:**\n{crop_mentioned.title()} cultivation success depends on {', '.join(info_explanations[:3])}. With these details, I can recommend the best {crop_mentioned} varieties and growing methods for your specific situation."
        else:
            why_text = f"**Why this helps:**\nAgricultural success depends on {', '.join(info_explanations[:3])}. With these details, I can provide tailored recommendations for your specific situation."

        clarification_prompt = f"""{intro}

{missing_info_text}
**To provide accurate recommendations, please tell me:**
{recommendations_text}

{why_text}"""

        return clarification_prompt

    async def process_query(self, query: str, chat_history: List[str] = None, user_state: Optional[str] = None) -> Dict:
        """
        Main pipeline orchestrator - coordinates all stages
        """
        if chat_history is None:
            chat_history = []
            
        print(f"\n{'='*60}")
        print(f"PROCESSING AGRICULTURAL QUERY")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"History length: {len(chat_history)} messages")
        
        start_time = datetime.now()
        
        # Stage 1: Analyze query completeness
        analysis = await self.analyze_query_completeness(query, chat_history)
        
        # CRITICAL: If query is incomplete, ask for clarification IMMEDIATELY
        if not analysis.is_complete:
            print(f"Query marked as INCOMPLETE - requesting clarification")
            clarification = await self.handle_incomplete_query(analysis, query)
            
            # Return clarification request instead of proceeding
            return {
                'response': clarification,
                'source': 'clarification_request',
                'confidence': analysis.confidence,
                'query_analysis': {
                    'type': analysis.query_type,
                    'complete': analysis.is_complete,
                    'missing_info': analysis.missing_info
                },
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Check agricultural relevance
        if analysis.agri_relevance < 0.3:
            return {
                'response': "I'm designed to help with agricultural questions. Could you please ask something related to farming, crops, livestock, or agricultural practices?",
                'source': 'relevance_filter',
                'confidence': 1.0,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Handle incomplete queries
        if not analysis.is_complete and analysis.confidence > 0.7:
            return {
                'response': await self.handle_incomplete_query(analysis),
                'source': 'clarification_request',
                'confidence': analysis.confidence,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Stage 2: Get RAG data directly (bypass ChromaQueryHandler's LLM calls)
        # Pass user_state down to RAG stage so state-aware filtering and exact-match checks can be performed
        rag_response = await self.get_rag_response_direct(query, chat_history, user_state=user_state)
        
        # Stage 3: Structure content
        structured_response = await self.structure_content(rag_response, query)
        
        # Stage 4: Quality evaluation
        quality = await self.evaluate_response_quality(query, structured_response, rag_response)
        
        # Stage 5: Final response preparation
        final_response = structured_response
        
        # If quality is low, try fallback model
        if not quality.is_satisfactory and quality.completeness_score < 0.6:
            print(f"Quality score low ({quality.completeness_score:.2f}), using fallback model: {self.models['fallback']}")
            
            fallback_prompt = f"""The previous response didn't fully address this agricultural question. Please provide a comprehensive answer.

Question: {query}
Chat History: {chr(10).join(chat_history[-4:])}

Previous attempt: {structured_response}

Provide a complete, accurate, and actionable agricultural response."""
            
            try:
                final_response = self.fallback_llm.generate_content(fallback_prompt, temperature=0.3)
                print("Fallback response generated")
            except Exception as e:
                print(f"Fallback model error: {e}")
        
        # Stage 6: Format the final response for better presentation
        try:
            metadata = {
                'source': rag_response.get('source', ''),
                'similarity_score': rag_response.get('similarity_score', 0),
                'distance': rag_response.get('distance', 1.0)
            }
            final_response = self.formatter.format_agricultural_response(
                final_response, query, metadata
            )
            print("Response formatted for better presentation")
        except Exception as e:
            print(f"Formatting error (using raw response): {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\nPROCESSING SUMMARY:")
        print(f"   Total time: {processing_time:.2f}s")
        print(f"   Query type: {analysis.query_type}")
        print(f"   Quality score: {quality.completeness_score:.2f}")
        print(f"   Data source: {rag_response.get('source', 'unknown')}")
        print(f"   Final model: {'fallback' if not quality.is_satisfactory else 'primary'}")
        
        return {
            'response': final_response,
            'source': rag_response.get('source', 'unknown'),
            'confidence': quality.completeness_score,
            'query_analysis': {
                'type': analysis.query_type,
                'completeness': analysis.is_complete,
                'agri_relevance': analysis.agri_relevance
            },
            'quality_assessment': {
                'satisfactory': quality.is_satisfactory,
                'completeness': quality.completeness_score,
                'relevance': quality.relevance_score,
                'suggestions': quality.suggestions
            },
            'processing_time': processing_time,
            'models_used': {
                'reasoner': self.models['reasoner'],
                'structurer': self.models['structurer'],
                'fallback_used': not quality.is_satisfactory
            }
        }

async def run_interactive_test():
    """
    Interactive test mode - chat with the new pipeline
    """
    pipeline = MultiModelPipeline()
    chat_history = []
    
    print(f"\n{'='*80}")
    print(f"MULTI-MODEL AGRICULTURAL AI PIPELINE - INTERACTIVE TEST")
    print(f"{'='*80}")
    print(f"Type your agricultural questions. Type 'quit' to exit, 'clear' to reset history.")
    print(f"{'='*80}\n")
    
    while True:
        try:
            query = input("Ask your agricultural question: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif query.lower() == 'clear':
                chat_history = []
                print("🧹 Chat history cleared!")
                continue
            elif not query:
                continue
            
            # Process the query (no explicit user_state in interactive mode)
            result = await pipeline.process_query(query, chat_history, user_state=None)
            
            # Display response
            print(f"\n{'='*60}")
            print(f"RESPONSE:")
            print(f"{'='*60}")
            print(result['response'])
            
            # Display metadata
            print(f"\nMETADATA:")
            print(f"   Source: {result['source']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Query type: {result.get('query_analysis', {}).get('type', 'unknown')}")
            
            # Add to history
            chat_history.append(f"User: {query}")
            chat_history.append(f"Assistant: {result['response']}")
            
            print(f"\n{'='*60}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

async def run_batch_test():
    """
    Batch test mode - test with predefined agricultural questions
    """
    pipeline = MultiModelPipeline()
    
    test_queries = [
        "What is the best variety of wheat for Punjab region?",
        "How do I control aphids in my cotton crop?",
        "When should I harvest tomatoes?",
        "What's the weather like today?",  # Non-agricultural
        "I have a problem with my crop",  # Incomplete
        "What fertilizer should I use for rice cultivation in monsoon season?",
        "How to prepare soil for potato planting in March?",
    ]
    
    print(f"\n{'='*80}")
    print(f"🧪 BATCH TEST MODE - {len(test_queries)} TEST QUERIES")
    print(f"{'='*80}")
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}/{len(test_queries)}: {query}")
        print("-" * 80)

        result = await pipeline.process_query(query, [], user_state=None)
        results.append({
            'query': query,
            'response_length': len(result['response']),
            'confidence': result['confidence'],
            'source': result['source'],
            'processing_time': result['processing_time'],
            'query_type': result.get('query_analysis', {}).get('type', 'unknown')
        })

        print(f"Response (first 200 chars): {result['response'][:200]}...")
        
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH TEST SUMMARY")
    print(f"{'='*80}")
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"Average processing time: {avg_time:.2f}s")
    print(f"Sources used: {set(r['source'] for r in results)}")
    print(f"Query types: {set(r['query_type'] for r in results)}")

def main():
    """
    Main function - choose test mode
    """
    print("Multi-Model Agricultural AI Pipeline Tester")
    print("=" * 60)
    print("Choose test mode:")
    print("1. Interactive mode (chat)")
    print("2. Batch test mode (predefined queries)")
    print("3. Single query test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(run_interactive_test())
    elif choice == "2":
        asyncio.run(run_batch_test())
    elif choice == "3":
        query = input("Enter your agricultural question: ").strip()
        pipeline = MultiModelPipeline()
        result = asyncio.run(pipeline.process_query(query, [], user_state=None))

        print(f"\n{'='*60}")
        print(f"RESPONSE: {result['response']}")
        print(f"\nMETADATA:")
        print(f"Source: {result['source']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()