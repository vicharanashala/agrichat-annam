"""
Agricultural Response Formatter
Converts raw LLM responses into well-structured, farmer-friendly formats
"""

import re
from typing import Dict, List, Any

class AgriculturalResponseFormatter:
    """
    Formats agricultural responses to be more readable and structured
    """
    
    def __init__(self):
        self.crop_keywords = [
            'wheat', 'rice', 'maize', 'cotton', 'sugarcane', 'soybean', 
            'mustard', 'barley', 'gram', 'pea', 'potato', 'onion', 'tomato',
            'chilli', 'turmeric', 'coriander', 'cumin', 'groundnut'
        ]
        
        self.season_keywords = ['rabi', 'kharif', 'summer', 'winter', 'monsoon']
        
        self.fertilizer_keywords = [
            'nitrogen', 'phosphorus', 'potassium', 'npk', 'urea', 'dap',
            'muriate', 'sulphate', 'organic', 'compost', 'manure'
        ]

    def format_agricultural_response(self, raw_response: str, query: str = "", 
                                   metadata: Dict = None) -> str:
        """
        Main formatting function that structures agricultural responses
        """
        if not raw_response or not raw_response.strip():
            return "No response available."
            
        # Clean and prepare the response
        formatted = self._clean_response(raw_response)
        
        # Add context-aware headers
        formatted = self._add_contextual_headers(formatted, query)
        
        # Structure key information
        formatted = self._structure_key_info(formatted)
        
        # Format lists and bullet points
        formatted = self._format_lists(formatted)
        
        # Add emphasis to important terms
        formatted = self._emphasize_key_terms(formatted)
        
        # Add practical tips section
        formatted = self._add_practical_tips(formatted)
        
        # Add source information if available
        if metadata:
            formatted = self._add_source_info(formatted, metadata)
            
        return formatted

    def _clean_response(self, response: str) -> str:
        """Clean up raw response text"""
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', response)
        
        # Fix common formatting issues
        cleaned = re.sub(r'(\d+)\.\s*\n', r'\1. ', cleaned)  # Fix numbered lists
        cleaned = re.sub(r'\*\s*\n', '* ', cleaned)  # Fix bullet points
        
        # Ensure proper sentence spacing
        cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)
        
        return cleaned.strip()

    def _add_contextual_headers(self, text: str, query: str) -> str:
        """Add relevant headers based on query context"""
        if not query:
            return text
            
        query_lower = query.lower()
        headers_added = []
        
        # Detect query type and add appropriate headers
        if any(crop in query_lower for crop in self.crop_keywords):
            if 'variety' in query_lower or 'varieties' in query_lower:
                headers_added.append("## 🌾 Recommended Varieties")
            elif 'fertilizer' in query_lower or 'nutrient' in query_lower:
                headers_added.append("## 🧪 Fertilizer Recommendations")
            elif 'disease' in query_lower or 'pest' in query_lower:
                headers_added.append("## 🛡️ Disease & Pest Management")
            elif 'harvest' in query_lower:
                headers_added.append("## ✂️ Harvesting Guidelines")
                
        if any(season in query_lower for season in self.season_keywords):
            headers_added.append("## 📅 Seasonal Considerations")
            
        if headers_added:
            return f"{headers_added[0]}\n\n{text}"
        
        return f"## Agricultural Information\n\n{text}"

    def _structure_key_info(self, text: str) -> str:
        """Structure key agricultural information"""
        # Pattern for seed rates, fertilizer amounts, etc.
        text = re.sub(
            r'(seed rate|seeding rate|sowing rate)[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(kg|g)?\s*(per|/)\s*(hectare|ha|acre)',
            r'**Seed Rate:** \2 \3/\5',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern for fertilizer recommendations - more specific matching
        text = re.sub(
            r'Apply\s+(nitrogen|n)[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(kg|g)\s*(per|/)\s*(hectare|ha|acre)',
            r'Apply **Nitrogen:** \2 \3/\5',
            text,
            flags=re.IGNORECASE
        )
        
        text = re.sub(
            r'Apply\s+(phosphorus|p)[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(kg|g)\s*(per|/)\s*(hectare|ha|acre)',
            r'Apply **Phosphorus:** \2 \3/\5',
            text,
            flags=re.IGNORECASE
        )
        
        text = re.sub(
            r'Apply\s+(potassium|k)[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(kg|g)\s*(per|/)\s*(hectare|ha|acre)',
            r'Apply **Potassium:** \2 \3/\5',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern for yields
        text = re.sub(
            r'(yield|production)[:\s]*([0-9]+(?:\.[0-9]+)?(?:-[0-9]+(?:\.[0-9]+)?)?)\s*(tonnes?|tons?|quintal|qtl)s?\s*(per|/)\s*(hectare|ha|acre)',
            r'**Expected Yield:** \2 \3/\5',
            text,
            flags=re.IGNORECASE
        )
        
        return text

    def _format_lists(self, text: str) -> str:
        """Improve list formatting"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            # Skip empty lines and existing markdown
            if not line or line.startswith('#') or line.startswith('*') or line.startswith('-'):
                formatted_lines.append(original_line)
                continue
            
            # Convert numbered lists to proper markdown
            if re.match(r'^\d+\.?\s+', line):
                line = re.sub(r'^(\d+)\.?\s+', r'\1. ', line)
                formatted_lines.append(line)
                continue
                
            # Convert bullet-like content to proper lists
            # Only if line starts with action words and is not already formatted
            if (len(line) > 15 and 
                any(line.lower().startswith(action) for action in 
                    ['apply', 'use', 'plant', 'sow', 'harvest', 'spray', 'maintain', 'monitor', 'ensure']) and
                not line.startswith(('**', '*', '-', '#'))):
                line = f"* {line}"
                
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def _emphasize_key_terms(self, text: str) -> str:
        """Add emphasis to important agricultural terms"""
        # Emphasize crop names
        for crop in self.crop_keywords:
            text = re.sub(
                rf'\b({crop})\b',
                r'**\1**',
                text,
                flags=re.IGNORECASE
            )
        
        # Emphasize fertilizer types
        for fertilizer in self.fertilizer_keywords:
            text = re.sub(
                rf'\b({fertilizer})\b',
                r'**\1**',
                text,
                flags=re.IGNORECASE
            )
            
        # Emphasize seasons
        for season in self.season_keywords:
            text = re.sub(
                rf'\b({season})\b',
                r'**\1**',
                text,
                flags=re.IGNORECASE
            )
        
        # Emphasize important numbers (but not dates)
        text = re.sub(
            r'\b(\d+(?:\.\d+)?)\s*(kg|g|tonnes?|tons?|quintal|qtl|hectare|ha|acre|days?|weeks?|months?)\b',
            r'**\1** \2',
            text,
            flags=re.IGNORECASE
        )
        
        return text

    def _add_practical_tips(self, text: str) -> str:
        """Add a practical tips section if not present"""
        if "tip" in text.lower() or "advice" in text.lower():
            return text
            
        # Extract actionable sentences for tips
        sentences = re.split(r'[.!?]+', text)
        tips = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(word in sentence.lower() for word in 
                    ['should', 'must', 'need', 'important', 'ensure', 'avoid', 'remember'])):
                tips.append(sentence)
        
        if tips and len(tips) <= 3:  # Don't add if too many tips
            tips_section = "\n\n### Key Tips\n\n"
            for tip in tips:
                tips_section += f"* {tip.strip()}\n"
            return text + tips_section
            
        return text

    def _extract_practical_tips(self, text: str) -> List[str]:
        """Return a short list of action-oriented tips extracted from the text."""
        sentences = re.split(r'[.!?]+', text)
        tips = []
        for sentence in sentences:
            s = sentence.strip()
            if (len(s) > 20 and 
                any(word in s.lower() for word in ['should', 'must', 'need', 'important', 'ensure', 'avoid', 'remember', 'apply', 'sow', 'plant', 'monitor', 'spray'])):
                tips.append(s)
        # Keep at most 4 concise tips
        return tips[:4]

    def _derive_title_from_query_or_text(self, query: str, text: str) -> str:
        """Create a short, human-friendly title for the answer based on query or text."""
        if query and len(query.strip()) > 3:
            q = query.strip()
            # Prefer short, descriptive titles
            if q.endswith('?'):
                q = q[:-1]
            return q[0].upper() + q[1:]

        # Fall back to extracting a first heading if present
        m = re.search(r'^(#{1,3})\s*(.+)$', text, flags=re.MULTILINE)
        if m:
            return m.group(2).strip()

        # Fallback: take the first 6 words
        plain = re.sub(r'<[^>]+>', '', text)
        words = plain.split()
        return ' '.join(words[:6]) + ('...' if len(words) > 6 else '')

    def _add_source_info(self, text: str, metadata: Dict) -> str:
        """Add source information"""
        source = metadata.get('source', 'unknown')
        similarity = metadata.get('similarity_score', 0)
        
        # Normalize common source keys to user-friendly labels
        source_map = {
            'rag_direct': 'RAG Database',
            'pops_direct': 'PoPs Database',
            'llm_fallback': 'Fallback LLM',
            'agricultural_knowledge_base': 'Fallback LLM',
            'golden': 'Golden FAQ',
            'error': 'System'
        }
        
        source_name = source_map.get(source, source.replace('_', ' ').title())
        
        if similarity > 0:
            confidence = "High" if similarity > 0.8 else "Medium" if similarity > 0.6 else "Low"
            source_info = f"\n\n---\n*Source: {source_name} (Confidence: {confidence})*"
        else:
            source_info = f"\n\n---\n*Source: {source_name}*"
            
        return text + source_info

    def format_simple_answer(self, answer_text: str, source: str = 'RAG Database', similarity: float = None, query: str = '') -> str:
        """Format a simple DB/fallback answer into a clear, structured markdown block.

        Produces:
        - Title derived from the query or content (no hard-coded 'Direct Answer')
        - A short overview paragraph (first line/paragraph)
        - Key considerations / bullets (attempt to turn actionable lines into bullets)
        - Key Tips section (extracted by existing helper)
        - Source footer: Source: <label> (Confidence: High/Medium/Low)
        """
        # Derive a short title
        title = self._derive_title_from_query_or_text(query, answer_text) if query else self._derive_title_from_query_or_text('', answer_text)

        # Clean and slightly normalize the answer
        cleaned = self._clean_response(answer_text)

        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n{1,}", cleaned) if p.strip()]

        overview = paragraphs[0] if paragraphs else cleaned

        # Build bullets from remaining paragraphs or long sentences
        bullets = []
        if len(paragraphs) > 1:
            for p in paragraphs[1:]:
                # If paragraph looks like a list or starts with an action word, keep as bullet
                if p.startswith(('*', '-', '\u2022')) or len(p) < 120 or any(p.lower().startswith(w) for w in ['apply', 'use', 'ensure', 'avoid', 'monitor', 'irrigate', 'sow', 'plant']):
                    bullets.append(p)
                else:
                    # break long paragraph into sentences and take the first as a bullet if appropriate
                    s = re.split(r'[.!?]+', p)[0].strip()
                    if len(s) > 10:
                        bullets.append(s)

        # Convert bullets to markdown list with minor formatting
        bullets_md = ''
        if bullets:
            for b in bullets:
                b_clean = b.strip()
                if not b_clean.startswith(('*', '-')):
                    b_clean = f"* {b_clean}"
                bullets_md += b_clean + '\n'

        # Generate key tips using existing extractor
        tips = self._extract_practical_tips(cleaned)
        tips_md = ''
        if tips:
            tips_md = '\n\n### Key Tips\n\n'
            for t in tips:
                tips_md += f"* {t.strip()}\n"

        # Map source key to friendly label
        friendly_map = {
            'RAG Database': 'RAG Database',
            'PoPs Database': 'PoPs Database',
            'Golden FAQ': 'Golden FAQ',
            'Fallback LLM': 'Fallback LLM'
        }
        source_label = friendly_map.get(source, source)

        # map similarity to confidence
        conf_val = similarity if similarity is not None else 0.0
        confidence = 'High' if conf_val >= 0.8 else 'Medium' if conf_val >= 0.6 else 'Low' if conf_val > 0 else 'Unknown'

        # Compose final markdown
        md = f"## {title}\n\n{overview}\n\n"
        if bullets_md:
            md += "### Key Considerations\n\n" + bullets_md + "\n"

        md += tips_md

        md += f"\n---\n*Source: {source_label} (Confidence: {confidence})*"

        # Run a final pass for lists and emphasis
        md = self._format_lists(md)
        md = self._emphasize_key_terms(md)

        return md

    def format_structured_response(self, response_data: Dict) -> str:
        """
        Format a complete response with all metadata
        """
        main_response = response_data.get('response', '')
        query = response_data.get('query', '')
        metadata = {
            'source': response_data.get('source', ''),
            'similarity_score': response_data.get('similarity_score', 0),
            'distance': response_data.get('distance', 1.0)
        }
        
        return self.format_agricultural_response(main_response, query, metadata)

# Convenience function for easy import
def format_response(response: str, query: str = "", metadata: Dict = None) -> str:
    """
    Quick formatting function
    """
    formatter = AgriculturalResponseFormatter()
    return formatter.format_agricultural_response(response, query, metadata)

# Example usage and testing
if __name__ == "__main__":
    # Test the formatter
    formatter = AgriculturalResponseFormatter()
    
    sample_response = """
    Wheat varieties for Punjab include HD 3086, PBW 725, and HD 2967. 
    Seed rate is 100 kg per hectare. Nitrogen requirement is 120 kg per hectare.
    Apply urea in split doses. Sowing time is November to December.
    Expected yield is 45 quintals per hectare.
    """
    
    sample_query = "Best wheat varieties for Punjab"
    
    formatted = formatter.format_agricultural_response(
        sample_response, 
        sample_query,
        {'source': 'rag_direct', 'similarity_score': 0.85}
    )
    
    print("=== FORMATTED RESPONSE ===")
    print(formatted)