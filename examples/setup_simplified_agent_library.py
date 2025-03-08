# examples/setup_simplified_agent_library.py

import asyncio
import logging
import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define real BeeAI-compatible tool and agent code snippets

# Real BeeAI-compatible DocumentAnalyzer tool
DOCUMENT_ANALYZER_TOOL = '''
from typing import Dict, Any
import re
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")

class DocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
    """
    Analyzes a document to identify its type and key characteristics.
    """
    name = "DocumentAnalyzer"
    description = "Identifies document type and extracts key information"
    input_schema = DocumentAnalyzerInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "document", "analyzer"],
            creator=self,
        )
    
    async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """
        Analyzes a document to identify its type and key characteristics.
        
        Args:
            input: Document text to analyze
            
        Returns:
            Document analysis including type, confidence, and keywords
        """
        text = input.text.lower()
        result = {
            "document_type": "unknown",
            "confidence": 0.5,
            "keywords": []
        }
        
        # Extract keywords (words that appear frequently or seem important)
        words = text.split()
        word_counts = {}
        
        for word in words:
            # Clean the word
            clean_word = word.strip(".,;:()[]{}\"'")
            if len(clean_word) > 3:  # Only count words with at least 4 characters
                word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # Get the top 5 most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        result["keywords"] = [word for word, count in sorted_words[:5]]
        
        # Determine document type based on content
        if "invoice" in text:
            if "total" in text and ("payment" in text or "due" in text):
                result["document_type"] = "invoice"
                result["confidence"] = 0.9
                
                # Check for invoice amount
                money_pattern = r'\\$(\d+,?\d*\.\d{2})'
                amounts = re.findall(money_pattern, input.text)
                if amounts:
                    result["has_monetary_values"] = True
                    try:
                        result["highest_amount"] = max([float(amt.replace(",", "")) for amt in amounts])
                    except:
                        pass
        
        elif "patient" in text:
            if "medical" in text or "assessment" in text or "diagnosis" in text:
                result["document_type"] = "medical"
                result["confidence"] = 0.92
                
                # Check for medical keywords
                medical_keywords = ["prescribed", "symptoms", "treatment", "follow-up", "medication"]
                for keyword in medical_keywords:
                    if keyword in text:
                        result["keywords"].append(keyword)
        
        elif "contract" in text or "agreement" in text:
            result["document_type"] = "contract"
            result["confidence"] = 0.85
        
        elif "report" in text:
            result["document_type"] = "report"
            result["confidence"] = 0.7
        
        # Clean up keywords to remove duplicates and sort
        result["keywords"] = list(set(result["keywords"]))
        
        import json
        return StringToolOutput(json.dumps(result, indent=2))
'''

# Real BeeAI-compatible AgentCommunicator tool
AGENT_COMMUNICATOR_TOOL = '''
from typing import Dict, Any, Optional
import json
import re
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class AgentCommunicatorInput(BaseModel):
    agent_name: str = Field(description="Name of the agent to communicate with")
    message: str = Field(description="The message/request to send")
    data: Dict[str, Any] = Field(description="Any supporting data to include", default_factory=dict)

class AgentCommunicator(Tool[AgentCommunicatorInput, ToolRunOptions, StringToolOutput]):
    """
    Facilitates communication between agents by formatting requests and routing them to specialized agents.
    """
    name = "AgentCommunicator"
    description = "Enables communication between different specialized agents"
    input_schema = AgentCommunicatorInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "agent", "communicator"],
            creator=self,
        )
    
    async def _run(self, input: AgentCommunicatorInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """
        Process a communication request between agents.
        
        Args:
            input: Communication request details including agent name, message, and data
            
        Returns:
            Response from the requested agent
        """
        try:
            # Log the communication attempt
            agent_name = input.agent_name
            message = input.message
            data = input.data
            
            # In a real implementation, we would use an agent registry or a more sophisticated
            # way to communicate between agents. For this example, we'll simulate responses.
            
            if agent_name == "SpecialistAgent":
                # Simulate specialist analysis
                result = self._specialist_analysis(message, data)
            else:
                return StringToolOutput(json.dumps({"error": f"Unknown agent: {agent_name}"}))
            
            return StringToolOutput(json.dumps(result, indent=2))
            
        except Exception as e:
            return StringToolOutput(json.dumps({"error": f"Communication error: {str(e)}"}))
    
    def _specialist_analysis(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs specialized analysis based on document content.
        In a real implementation, this would call the SpecialistAgent.
        """
        document_type = options.get("document_type", "unknown")
        lower_text = text.lower()
        
        results = {
            "analysis": {},
            "extracted_data": {}
        }
        
        # Extract basic data
        # Dates (YYYY-MM-DD format)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, text)
        if dates:
            results["extracted_data"]["dates"] = dates
        
        # Monetary values
        money_pattern = r'\\$(\d+,?\d*\.\d{2})'
        monetary_values = re.findall(money_pattern, text)
        if monetary_values:
            results["extracted_data"]["monetary_values"] = [value.replace(",", "") for value in monetary_values]
        
        # Document-specific analysis
        if document_type == "invoice" or "invoice" in lower_text:
            # Invoice analysis
            results["analysis"]["document_type"] = "invoice"
            results["analysis"]["priority"] = "medium"
            
            # Extract vendor
            vendor_match = re.search(r'Vendor: ([^\\n]+)', text)
            if vendor_match:
                results["extracted_data"]["vendor"] = vendor_match.group(1).strip()
            
            # Extract total
            total_match = re.search(r'Total[^:]*: ?\\$(\d+,?\d*\.\d{2})', text)
            if total_match:
                total = total_match.group(1).replace(",", "")
                results["extracted_data"]["total_amount"] = total
                
                # Set priority based on amount
                try:
                    amount = float(total)
                    if amount > 1000:
                        results["analysis"]["priority"] = "high"
                        results["analysis"]["approval_required"] = True
                        results["analysis"]["notes"] = "Large invoice requires manager approval"
                    else:
                        results["analysis"]["approval_required"] = False
                except:
                    pass
                
        elif document_type == "medical" or "patient" in lower_text:
            # Medical record analysis
            results["analysis"]["document_type"] = "medical_record"
            results["analysis"]["priority"] = "medium"
            
            # Extract patient name
            name_match = re.search(r'Name: ([^\\n]+)', text)
            if name_match:
                results["extracted_data"]["patient_name"] = name_match.group(1).strip()
            
            # Extract diagnosis
            diagnosis_match = re.search(r'Assessment: ([^\\n]+)', text)
            if diagnosis_match:
                diagnosis = diagnosis_match.group(1).strip()
                results["extracted_data"]["diagnosis"] = diagnosis
                
                # Set priority based on diagnosis
                if "acute" in diagnosis.lower() or "emergency" in diagnosis.lower():
                    results["analysis"]["priority"] = "high"
                    results["analysis"]["follow_up_required"] = True
                    results["analysis"]["notes"] = "Urgent condition requires immediate follow-up"
                else:
                    results["analysis"]["follow_up_required"] = False
        
        return results
'''

# Real BeeAI-compatible SpecialistAgent
SPECIALIST_AGENT = '''
from typing import List, Dict, Any, Optional
import json
import re

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class SpecialistAgentInitializer:
    """
    Specialist agent that performs detailed document analysis.
    
    This agent provides deep expertise for specific document types,
    extracting important information and providing domain-specific insights.
    """
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        """Create and configure the specialist agent with tools."""
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="SpecialistAgent",
            description=(
                "Specialist agent that performs detailed document analysis. "
                "This agent provides deep expertise for specific document types, "
                "extracting important information and providing domain-specific insights."
            ),
            tools=tools
        )
        
        # Create the agent
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=TokenMemory(llm),
            meta=meta
        )
        
        return agent
        
    @staticmethod
    async def analyze_document(document_text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes a document and returns structured information.
        
        Args:
            document_text: The text of the document to analyze
            document_type: Optional type hint for the document
            
        Returns:
            Structured analysis of the document
        """
        # Determine document type if not provided
        if not document_type:
            lower_text = document_text.lower()
            if "invoice" in lower_text:
                document_type = "invoice"
            elif "patient" in lower_text or "medical" in lower_text:
                document_type = "medical"
            else:
                document_type = "unknown"
        
        # Initialize results
        results = {
            "analysis": {
                "document_type": document_type,
                "priority": "medium"
            },
            "extracted_data": {}
        }
        
        # Extract common data
        # Dates (YYYY-MM-DD format)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, document_text)
        if dates:
            results["extracted_data"]["dates"] = dates
        
        # Perform document-specific analysis
        if document_type == "invoice":
            # Analyze invoice
            SpecialistAgentInitializer._analyze_invoice(document_text, results)
        elif document_type == "medical":
            # Analyze medical record
            SpecialistAgentInitializer._analyze_medical_record(document_text, results)
        else:
            # Generic analysis for unknown types
            results["analysis"]["notes"] = "Document type not recognized for specialized analysis"
        
        return results
    
    @staticmethod
    def _analyze_invoice(text: str, results: Dict[str, Any]) -> None:
        """Specialized invoice analysis"""
        # Extract vendor
        vendor_match = re.search(r'Vendor: ([^\\n]+)', text)
        if vendor_match:
            results["extracted_data"]["vendor"] = vendor_match.group(1).strip()
        
        # Extract invoice number
        invoice_num_match = re.search(r'(?:INVOICE|Invoice)[ #:]+([A-Z0-9]+)', text)
        if invoice_num_match:
            results["extracted_data"]["invoice_number"] = invoice_num_match.group(1).strip()
        
        # Extract total amount
        total_match = re.search(r'Total[^:]*: ?\\$(\d+,?\d*\.\d{2})', text)
        if total_match:
            total = total_match.group(1).replace(",", "")
            results["extracted_data"]["total_amount"] = total
            
            # Set priority based on amount
            try:
                amount = float(total)
                if amount > 1000:
                    results["analysis"]["priority"] = "high"
                    results["analysis"]["approval_required"] = True
                    results["analysis"]["notes"] = "Large invoice requires manager approval"
                else:
                    results["analysis"]["approval_required"] = False
            except:
                pass
        
        # Extract due date
        due_date_match = re.search(r'Due Date: (\d{4}-\d{2}-\d{2})', text)
        if due_date_match:
            results["extracted_data"]["due_date"] = due_date_match.group(1)
    
    @staticmethod
    def _analyze_medical_record(text: str, results: Dict[str, Any]) -> None:
        """Specialized medical record analysis"""
        # Extract patient name
        name_match = re.search(r'Name: ([^\\n]+)', text)
        if name_match:
            results["extracted_data"]["patient_name"] = name_match.group(1).strip()
        
        # Extract patient ID
        patient_id_match = re.search(r'Patient ID: ([^\\n]+)', text)
        if patient_id_match:
            results["extracted_data"]["patient_id"] = patient_id_match.group(1).strip()
        
        # Extract diagnosis
        diagnosis_match = re.search(r'Assessment: ([^\\n]+)', text)
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
            results["extracted_data"]["diagnosis"] = diagnosis
            
            # Set priority based on diagnosis severity
            if "acute" in diagnosis.lower() or "emergency" in diagnosis.lower():
                results["analysis"]["priority"] = "high"
                results["analysis"]["follow_up_required"] = True
                results["analysis"]["notes"] = "Urgent condition requires immediate follow-up"
            else:
                results["analysis"]["follow_up_required"] = False
        
        # Extract vital signs
        vitals = {}
        temp_match = re.search(r'Temperature: ([^\\n]+)', text)
        if temp_match:
            temp = temp_match.group(1).strip()
            vitals["temperature"] = temp
            
            # Flag fever
            if "°F" in temp and float(temp.replace("°F", "").strip()) > 100:
                results["analysis"]["has_fever"] = True
        
        bp_match = re.search(r'Blood Pressure: ([^\\n]+)', text)
        if bp_match:
            vitals["blood_pressure"] = bp_match.group(1).strip()
        
        if vitals:
            results["extracted_data"]["vitals"] = vitals
'''

# Real BeeAI-compatible CoordinatorAgent
COORDINATOR_AGENT = '''
from typing import List, Dict, Any, Optional
import json

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class CoordinatorAgentInitializer:
    """
    Primary agent that orchestrates document processing.
    
    This agent:
    1. Analyzes the document to identify its type
    2. Sends the document to a specialist agent for detailed analysis
    3. Synthesizes the results and provides recommendations
    """
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        """Create and configure the coordinator agent with tools."""
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="CoordinatorAgent",
            description=(
                "Primary agent that orchestrates document processing. "
                "This agent analyzes documents, delegates to specialists, "
                "and synthesizes results with recommendations."
            ),
            tools=tools
        )
        
        # Create the agent with the necessary tools
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=TokenMemory(llm),
            meta=meta
        )
        
        return agent
        
    @staticmethod
    def generate_recommendations(document_type: str, analysis: Dict[str, Any], extracted_data: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on document analysis.
        
        Args:
            document_type: Type of document (invoice, medical, etc.)
            analysis: Analysis data from specialist agent
            extracted_data: Data extracted from the document
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if document_type == "invoice":
            # Invoice recommendations
            if "total_amount" in extracted_data:
                try:
                    amount = float(extracted_data["total_amount"])
                    if amount > 1000:
                        recommendations.append("HIGH PRIORITY: Review large invoice amount")
                    
                    if "approval_required" in analysis and analysis["approval_required"]:
                        recommendations.append("Route to finance manager for approval")
                    else:
                        recommendations.append("Process for payment within standard timeframe")
                except:
                    recommendations.append("Verify invoice amount")
            
            if "vendor" in extracted_data:
                recommendations.append(f"Confirm vendor details for {extracted_data['vendor']}")
                
            if "dates" in extracted_data and len(extracted_data["dates"]) > 0:
                recommendations.append(f"Document date: {extracted_data['dates'][0]}")
        
        elif document_type == "medical":
            # Medical record recommendations
            if "diagnosis" in extracted_data:
                recommendations.append(f"Noted diagnosis: {extracted_data['diagnosis']}")
                
            if "follow_up_required" in analysis and analysis["follow_up_required"]:
                recommendations.append("PRIORITY: Schedule follow-up appointment")
            
            if "patient_name" in extracted_data:
                recommendations.append(f"Update patient record for {extracted_data['patient_name']}")
        
        else:
            # Default recommendations
            recommendations.append("Document requires manual review")
            recommendations.append("Route to appropriate department based on content")
        
        # Add standard recommendation
        recommendations.append("Archive document according to record retention policy")
        
        return recommendations
'''

# Fix for SystemAgent.execute_item method - to be applied at runtime
async def fixed_execute_item(self, name: str, input_text: str):
    """
    Execute an active item (agent or tool).
    
    Args:
        name: Name of the item to execute
        input_text: Input text for the item
        
    Returns:
        Execution result
    """
    if name not in self.active_items:
        return {
            "status": "error",
            "message": f"Item '{name}' not found in active items"
        }
    
    item = self.active_items[name]
    record = item["record"]
    instance = item["instance"]
    
    logger.info(f"Executing {record['record_type']} '{name}' with input: {input_text[:50]}...")
    
    try:
        # Execute based on record type
        if record["record_type"] == "AGENT":
            result = await self.agent_factory.execute_agent(
                name,  # Use name instead of instance
                input_text
            )
        else:  # TOOL
            result = await self.tool_factory.execute_tool(
                instance, 
                input_text
            )
        
        # Update usage metrics
        await self.library.update_usage_metrics(record["id"], True)
        
        return {
            "status": "success",
            "item_name": name,
            "item_type": record["record_type"],
            "result": result,
            "message": f"Executed {record['record_type']} '{name}'"
        }
    except Exception as e:
        logger.error(f"Error executing {record['record_type']} '{name}': {str(e)}")
        
        # Update usage metrics as failure
        await self.library.update_usage_metrics(record["id"], False)
        
        return {
            "status": "error",
            "message": f"Error executing {record['record_type']} '{name}': {str(e)}"
        }

async def main():
    try:
        # Check if library exists, and if so, delete it to recreate
        library_path = "simplified_agent_library.json"
        if os.path.exists(library_path):
            os.remove(library_path)
            print(f"Deleted existing library at {library_path} to create a fresh version")
            
        # Initialize library
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        print("Setting up simplified agent library with real BeeAI agents and tools...")
        
        # Add the AgentCommunicator tool
        await library.create_record(
            name="AgentCommunicator",
            record_type="TOOL",
            domain="document_processing",
            description="Tool for facilitating communication between agents",
            code_snippet=AGENT_COMMUNICATOR_TOOL,
            tags=["communication", "agent", "tool"]
        )
        print("✓ Created AgentCommunicator tool (BeeAI-compatible)")
        
        # Add the DocumentAnalyzer tool
        await library.create_record(
            name="DocumentAnalyzer",
            record_type="TOOL",
            domain="document_processing",
            description="Tool to analyze documents and identify their type",
            code_snippet=DOCUMENT_ANALYZER_TOOL,
            tags=["analysis", "tool"]
        )
        print("✓ Created DocumentAnalyzer tool (BeeAI-compatible)")
        
        # Add the SpecialistAgent 
        await library.create_record(
            name="SpecialistAgent",
            record_type="AGENT",
            domain="document_processing",
            description="Specialist agent that performs detailed document analysis",
            code_snippet=SPECIALIST_AGENT,
            metadata={"framework": "beeai"},
            tags=["specialist", "agent", "analysis"]
        )
        print("✓ Created SpecialistAgent (BeeAI-compatible)")
        
        # Add the CoordinatorAgent 
        await library.create_record(
            name="CoordinatorAgent",
            record_type="AGENT",
            domain="document_processing",
            description="Primary agent that orchestrates document processing",
            code_snippet=COORDINATOR_AGENT,
            metadata={
                "framework": "beeai",
                "required_tools": ["DocumentAnalyzer", "AgentCommunicator"]
            },
            tags=["coordinator", "agent", "orchestration"]
        )
        print("✓ Created CoordinatorAgent (BeeAI-compatible)")
        
        # Included in the setup file: Monkey patch the SystemAgent.execute_item method
        # This will be applied by importing SystemAgent and replacing its method
        try:
            from evolving_agents.core.system_agent import SystemAgent
            # Apply the fix - this ensures that when system_agent is imported later, 
            # it already has the fixed method
            SystemAgent.execute_item = fixed_execute_item
            print("✓ Applied fix to SystemAgent.execute_item method")
        except ImportError:
            print("! Unable to patch SystemAgent - skipping this step")
        
        print("\nLibrary setup complete with real BeeAI agents and tools!")
        print(f"Library saved to: {library.storage_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())