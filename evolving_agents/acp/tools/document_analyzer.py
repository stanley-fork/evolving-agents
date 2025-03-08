# evolving_agents/acp/tools/document_analyzer.py

from typing import Dict, Any, List, Optional
import re
import json
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, ToolRunOptions

from evolving_agents.acp.tool import ACPTool

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")
    analyze_mode: str = Field(
        description="Analysis mode: 'basic', 'detailed', or 'extract'",
        default="basic"
    )

class DocumentAnalyzer(ACPTool):
    """
    ACP-compatible tool for analyzing documents.
    """
    input_schema = DocumentAnalyzerInput
    
    def __init__(self, options: Dict[str, Any] | None = None):
        """Initialize the document analyzer tool."""
        super().__init__(
            name="DocumentAnalyzer",
            description="Analyzes documents to identify type, extract key information, and provide insights",
            options=options
        )
    
    async def _run(
        self,
        input: DocumentAnalyzerInput,
        options: ToolRunOptions | None,
        context: RunContext
    ) -> StringToolOutput:
        """
        Analyze a document.
        
        Args:
            input: Document text and analysis mode
            options: Optional tool options
            context: Execution context
            
        Returns:
            Analysis results
        """
        text = input.text
        mode = input.analyze_mode
        
        # Basic document analysis
        result = {
            "document_type": self._determine_document_type(text),
            "language": self._detect_language(text),
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
        # Add more detailed analysis based on mode
        if mode in ["detailed", "extract"]:
            result.update({
                "keywords": self._extract_keywords(text),
                "dates": self._extract_dates(text),
                "named_entities": self._extract_entities(text)
            })
        
        # Extract structured data for specific document types
        if mode == "extract":
            result["extracted_data"] = self._extract_structured_data(text, result["document_type"])
        
        return StringToolOutput(json.dumps(result, indent=2))
    
    def _determine_document_type(self, text: str) -> str:
        """Determine the document type based on content."""
        text_lower = text.lower()
        
        if "invoice" in text_lower and ("total" in text_lower or "amount" in text_lower):
            return "invoice"
        elif "patient" in text_lower and ("medical" in text_lower or "diagnosis" in text_lower):
            return "medical_record"
        elif "contract" in text_lower and "agreement" in text_lower:
            return "contract"
        elif "report" in text_lower:
            return "report"
        elif "resume" in text_lower or "cv" in text_lower:
            return "resume"
        elif "memo" in text_lower or "memorandum" in text_lower:
            return "memo"
        elif "article" in text_lower or "publication" in text_lower:
            return "article"
        else:
            return "unknown"
    
    def _detect_language(self, text: str) -> str:
        """Detect the document language (simplified)."""
        # Very basic language detection - in a real implementation, 
        # you'd use a proper language detection library
        english_words = ["the", "and", "to", "of", "in", "is", "for", "with"]
        spanish_words = ["el", "la", "de", "en", "es", "por", "con", "para"]
        french_words = ["le", "la", "de", "en", "est", "pour", "avec", "et"]
        
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        english_count = sum(1 for word in english_words if word in words)
        spanish_count = sum(1 for word in spanish_words if word in words)
        french_count = sum(1 for word in french_words if word in words)
        
        if english_count > spanish_count and english_count > french_count:
            return "english"
        elif spanish_count > english_count and spanish_count > french_count:
            return "spanish"
        elif french_count > english_count and french_count > spanish_count:
            return "french"
        else:
            return "unknown"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the document."""
        # Simple keyword extraction
        common_words = {"the", "and", "to", "of", "in", "is", "for", "with", "this", "that", "it", "on", "as", "by"}
        words = re.findall(r'\b\w{3,}\b', text.lower())
        word_counts = {}
        
        for word in words:
            if word not in common_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get the top 10 most frequent words
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, _ in keywords]
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from the document."""
        # Extract dates in various formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        
        return dates
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from the document."""
        # Very simple entity extraction
        # In a real implementation, you'd use a proper NER model
        entities = {
            "names": [],
            "organizations": [],
            "locations": []
        }
        
        # Simple pattern matching for names (Mr./Ms./Dr. followed by capitalized words)
        name_matches = re.findall(r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.) [A-Z][a-z]+(?: [A-Z][a-z]+)*', text)
        entities["names"].extend(name_matches)
        
        # Simple pattern matching for organizations (capitalized words followed by Inc./Corp./LLC)
        org_matches = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)* (?:Inc\.|Corp\.|LLC|Ltd\.)', text)
        entities["organizations"].extend(org_matches)
        
        return entities
    
    def _extract_structured_data(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract structured data based on document type."""
        structured_data = {}
        
        if document_type == "invoice":
            # Extract invoice details
            invoice_num_match = re.search(r'(?:INVOICE|Invoice)[ #:]+([A-Z0-9-]+)', text)
            if invoice_num_match:
                structured_data["invoice_number"] = invoice_num_match.group(1)
            
            # Extract amount
            amount_match = re.search(r'(?:Total|Amount|Due)[^:]*:?\s*\$?([0-9,]+\.\d{2})', text)
            if amount_match:
                structured_data["amount"] = amount_match.group(1)
            
            # Extract vendor
            vendor_match = re.search(r'(?:Vendor|From|Supplier)[^:]*:?\s*([^\n]+)', text)
            if vendor_match:
                structured_data["vendor"] = vendor_match.group(1).strip()
            
        elif document_type == "medical_record":
            # Extract patient information
            patient_match = re.search(r'(?:Patient|Name)[^:]*:?\s*([^\n]+)', text)
            if patient_match:
                structured_data["patient_name"] = patient_match.group(1).strip()
            
            # Extract diagnosis
            diagnosis_match = re.search(r'(?:Diagnosis|Assessment)[^:]*:?\s*([^\n]+)', text)
            if diagnosis_match:
                structured_data["diagnosis"] = diagnosis_match.group(1).strip()
            
        return structured_data