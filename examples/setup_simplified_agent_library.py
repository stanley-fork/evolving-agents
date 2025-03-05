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

# Define tool and agent code snippets
AGENT_COMMUNICATOR_TOOL = '''
# Tool to facilitate communication between agents
import json
import re

def communicate_with_agent(input):
    """
    Facilitates communication between agents by formatting requests and routing them to specialized agents.
    
    Args:
        input: JSON string containing:
            - agent_name: Name of the agent to communicate with
            - message: The message/request to send
            - data: Any supporting data to include
    
    Returns:
        Response from the requested agent
    """
    try:
        # Parse the input
        request = json.loads(input)
        agent_name = request.get("agent_name")
        message = request.get("message")
        data = request.get("data", {})
        
        if not agent_name or not message:
            return {"error": "Missing required fields: agent_name and message"}
        
        # Log the communication attempt
        print(f"Communication request to agent: {agent_name}")
        
        # Route to the appropriate agent
        if agent_name == "SpecialistAgent":
            # Call the specialist function
            result = specialist_function(message, data)
        else:
            return {"error": f"Unknown agent: {agent_name}"}
        
        return result
    except Exception as e:
        return {"error": f"Communication error: {str(e)}"}

def specialist_function(text, options=None):
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
    money_pattern = r'\$(\d+,?\d*\.\d{2})'
    monetary_values = re.findall(money_pattern, text)
    if monetary_values:
        results["extracted_data"]["monetary_values"] = [value.replace(",", "") for value in monetary_values]
    
    # Document-specific analysis
    if document_type == "invoice" or "invoice" in lower_text:
        # Invoice analysis
        results["analysis"]["document_type"] = "invoice"
        results["analysis"]["priority"] = "medium"
        
        # Extract vendor
        vendor_match = re.search(r'Vendor: ([^\n]+)', text)
        if vendor_match:
            results["extracted_data"]["vendor"] = vendor_match.group(1).strip()
        
        # Extract total
        total_match = re.search(r'Total[^:]*: ?\$(\d+,?\d*\.\d{2})', text)
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
        name_match = re.search(r'Name: ([^\n]+)', text)
        if name_match:
            results["extracted_data"]["patient_name"] = name_match.group(1).strip()
        
        # Extract diagnosis
        diagnosis_match = re.search(r'Assessment: ([^\n]+)', text)
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

# Process the input and return the result
result = communicate_with_agent(input)
'''

# FIXED: Document analyzer tool with properly fixed string literal issue
DOCUMENT_ANALYZER_TOOL = '''
# Tool to analyze documents and identify their type
import json
import re

def analyze_document(input):
    """
    Analyzes a document to identify its type and key characteristics.
    
    Args:
        input: Document text to analyze
        
    Returns:
        Document analysis including type, confidence, and keywords
    """
    text = input.lower()
    result = {
        "document_type": "unknown",
        "confidence": 0.5,
        "keywords": []
    }
    
    # Extract keywords (words that appear frequently or seem important)
    words = text.split()
    word_counts = {}
    
    for word in words:
        # Clean the word - FIXED string literals
        clean_word = word.strip(".,;:()[]{}\"'!?")
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
            money_pattern = r'\$(\d+,?\d*\.\d{2})'
            amounts = re.findall(money_pattern, input)
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
    
    return result

# Process the input text
result = analyze_document(input)
'''

# UPDATED: CoordinatorAgent with improved BeeAI compatibility
COORDINATOR_AGENT = '''
# Primary agent that orchestrates document processing
import json

def process_document(input_text):
    """
    Process a document by coordinating with a specialist agent.
    
    This agent:
    1. Analyzes the document to identify its type
    2. Sends the document to a specialist agent for detailed analysis
    3. Synthesizes the results and provides recommendations
    
    Args:
        input_text: The document text to process
        
    Returns:
        Comprehensive document analysis with recommendations
    """
    # Extract the document text from the input
    if "Process this document:" in input_text:
        document_text = input_text.split("Process this document:", 1)[1].strip()
    else:
        document_text = input_text
    
    try:
        # Step 1: Use the DocumentAnalyzer tool to identify the document type
        document_analysis = analyze_document(document_text)
        
        # Step 2: Send to specialist agent for detailed analysis
        specialist_request = {
            "agent_name": "SpecialistAgent",
            "message": document_text,
            "data": {"document_type": document_analysis.get("document_type")}
        }
        
        # Use the AgentCommunicator tool to communicate with the specialist
        specialist_result_json = communicate_with_agent(json.dumps(specialist_request))
        
        # Parse the specialist result
        if isinstance(specialist_result_json, str):
            try:
                specialist_result = json.loads(specialist_result_json)
            except:
                specialist_result = specialist_result_json
        else:
            specialist_result = specialist_result_json
        
        # Step 3: Generate recommendations based on all analyses
        recommendations = generate_recommendations(
            document_type=document_analysis.get("document_type"),
            analysis=specialist_result.get("analysis", {}),
            extracted_data=specialist_result.get("extracted_data", {})
        )
        
        # Step 4: Compile the final response
        final_result = {
            "document_type": document_analysis.get("document_type"),
            "confidence": document_analysis.get("confidence"),
            "keywords": document_analysis.get("keywords"),
            "analysis": specialist_result.get("analysis", {}),
            "extracted_data": specialist_result.get("extracted_data", {}),
            "recommendations": recommendations
        }
        
        # Format the final result for BeeAI compatibility
        return format_response_for_beeai(final_result)
    
    except Exception as e:
        # Handle any errors
        return f"RESULT: Error processing document: {str(e)}"

def generate_recommendations(document_type, analysis, extracted_data):
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

def format_response_for_beeai(result):
    """Format the response to ensure BeeAI can process it correctly."""
    # For BeeAI to parse properly, we need to start with a prefix like "THINKING:" or "RESULT:"
    
    # Convert the JSON result to a formatted text response
    if isinstance(result, dict):
        # Determine document type
        doc_type = result.get("document_type", "unknown")
        
        # Create a formatted response
        if doc_type == "medical":
            # Medical record format
            medical_response = "RESULT: Medical Record Analysis\\n\\n"
            
            # Add patient info
            extracted_data = result.get("extracted_data", {})
            if "patient_name" in extracted_data:
                medical_response += f"Patient: {extracted_data['patient_name']}\\n"
            if "patient_id" in extracted_data:
                medical_response += f"ID: {extracted_data['patient_id']}\\n"
                
            # Add diagnosis
            if "diagnosis" in extracted_data:
                medical_response += f"Diagnosis: {extracted_data['diagnosis']}\\n"
            
            # Add vitals
            vitals = extracted_data.get("vitals", {})
            if vitals:
                medical_response += "\\nVitals:\\n"
                for key, value in vitals.items():
                    medical_response += f"- {key}: {value}\\n"
                    
            # Add recommendations
            recommendations = result.get("recommendations", [])
            if recommendations:
                medical_response += "\\nRecommendations:\\n"
                for rec in recommendations:
                    medical_response += f"- {rec}\\n"
                    
            return medical_response
        
        else:
            # Default formatting for other document types
            text_response = ["RESULT: Document Analysis\\n"]
            
            # Add document type and confidence
            confidence = result.get("confidence", 0)
            text_response.append(f"Document Type: {doc_type} (Confidence: {confidence:.2f})\\n")
            
            # Add keywords
            keywords = result.get("keywords", [])
            if keywords:
                text_response.append(f"Keywords: {', '.join(keywords)}\\n")
            
            # Add extracted data
            extracted_data = result.get("extracted_data", {})
            if extracted_data:
                text_response.append("Extracted Data:")
                for key, value in extracted_data.items():
                    text_response.append(f"- {key}: {value}")
            
            # Add recommendations
            recommendations = result.get("recommendations", [])
            if recommendations:
                text_response.append("\\nRecommendations:")
                for rec in recommendations:
                    text_response.append(f"- {rec}")
            
            return "\\n".join(text_response)
    else:
        # If already a string, prefix with RESULT:
        if not str(result).startswith("RESULT:"):
            return f"RESULT: {str(result)}"
        return str(result)

# These functions simulate the tool calls
def analyze_document(text):
    """Simulate the DocumentAnalyzer tool"""
    if "invoice" in text.lower():
        return {
            "document_type": "invoice",
            "confidence": 0.9,
            "keywords": ["invoice", "payment", "total", "vendor", "due"]
        }
    elif "patient" in text.lower():
        return {
            "document_type": "medical",
            "confidence": 0.92,
            "keywords": ["patient", "assessment", "prescribed", "treatment", "follow-up"]
        }
    else:
        return {
            "document_type": "unknown",
            "confidence": 0.5,
            "keywords": []
        }

def communicate_with_agent(input_json):
    """Simulate the AgentCommunicator tool"""
    request = json.loads(input_json)
    message = request.get("message")
    document_type = request.get("data", {}).get("document_type")
    
    extracted_data = {}
    analysis = {"document_type": document_type}
    
    if document_type == "invoice":
        # Extract invoice data
        if "TechSupplies" in message:
            extracted_data["vendor"] = "TechSupplies Inc."
        if "Total Due: $1,822.80" in message:
            extracted_data["total_amount"] = "1822.80"
        
        analysis["priority"] = "high"
        analysis["approval_required"] = True
    
    elif document_type == "medical":
        # Extract medical data
        if "John Smith" in message:
            extracted_data["patient_name"] = "John Smith"
        if "Acute bronchitis" in message:
            extracted_data["diagnosis"] = "Acute bronchitis"
        
        analysis["priority"] = "medium"
        analysis["follow_up_required"] = True
    
    return {
        "analysis": analysis,
        "extracted_data": extracted_data
    }

# Process the input
result = process_document(input)
'''

SPECIALIST_AGENT = '''
# Specialist agent that performs detailed document analysis
import json
import re

def analyze_specialized_document(input_text):
    """
    Performs specialized analysis on a document based on its type.
    
    This agent provides deep expertise for specific document types,
    extracting important information and providing domain-specific insights.
    
    Args:
        input_text: Document text to analyze (may contain document_type)
        
    Returns:
        Detailed analysis and extracted information
    """
    # Parse input to check for JSON format with document_type
    document_type = None
    document_text = input_text
    
    try:
        # Check if input is JSON
        input_data = json.loads(input_text)
        if isinstance(input_data, dict):
            document_type = input_data.get("document_type")
            document_text = input_data.get("text", input_text)
    except:
        # Not JSON, treat as plain text
        pass
    
    # If document_type not found in JSON, determine from content
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
        analyze_invoice(document_text, results)
    elif document_type == "medical":
        # Analyze medical record
        analyze_medical_record(document_text, results)
    else:
        # Generic analysis for unknown types
        results["analysis"]["notes"] = "Document type not recognized for specialized analysis"
    
    return results

def analyze_invoice(text, results):
    """Specialized invoice analysis"""
    # Extract vendor
    vendor_match = re.search(r'Vendor: ([^\n]+)', text)
    if vendor_match:
        results["extracted_data"]["vendor"] = vendor_match.group(1).strip()
    
    # Extract invoice number
    invoice_num_match = re.search(r'(?:INVOICE|Invoice)[ #:]+([A-Z0-9]+)', text)
    if invoice_num_match:
        results["extracted_data"]["invoice_number"] = invoice_num_match.group(1).strip()
    
    # Extract total amount
    total_match = re.search(r'Total[^:]*: ?\$(\d+,?\d*\.\d{2})', text)
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
        
        # Check if payment is urgent
        from datetime import datetime
        try:
            due_date = datetime.strptime(results["extracted_data"]["due_date"], "%Y-%m-%d")
            if (due_date - datetime.now()).days < 7:
                results["analysis"]["urgent_payment"] = True
        except:
            pass

def analyze_medical_record(text, results):
    """Specialized medical record analysis"""
    # Extract patient name
    name_match = re.search(r'Name: ([^\n]+)', text)
    if name_match:
        results["extracted_data"]["patient_name"] = name_match.group(1).strip()
    
    # Extract patient ID
    patient_id_match = re.search(r'Patient ID: ([^\n]+)', text)
    if patient_id_match:
        results["extracted_data"]["patient_id"] = patient_id_match.group(1).strip()
    
    # Extract diagnosis
    diagnosis_match = re.search(r'Assessment: ([^\n]+)', text)
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
    temp_match = re.search(r'Temperature: ([^\n]+)', text)
    if temp_match:
        temp = temp_match.group(1).strip()
        vitals["temperature"] = temp
        
        # Flag fever
        if "°F" in temp and float(temp.replace("°F", "").strip()) > 100:
            results["analysis"]["has_fever"] = True
    
    bp_match = re.search(r'Blood Pressure: ([^\n]+)', text)
    if bp_match:
        vitals["blood_pressure"] = bp_match.group(1).strip()
    
    if vitals:
        results["extracted_data"]["vitals"] = vitals

# Process the input
result = analyze_specialized_document(input)
'''

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
        
        print("Setting up simplified agent library...")
        
        # Add the AgentCommunicator tool
        await library.create_record(
            name="AgentCommunicator",
            record_type="TOOL",
            domain="document_processing",
            description="Tool for facilitating communication between agents",
            code_snippet=AGENT_COMMUNICATOR_TOOL,
            tags=["communication", "agent", "tool"]
        )
        print("✓ Created AgentCommunicator tool")
        
        # Add the DocumentAnalyzer tool with fixed code
        await library.create_record(
            name="DocumentAnalyzer",
            record_type="TOOL",
            domain="document_processing",
            description="Tool to analyze documents and identify their type",
            code_snippet=DOCUMENT_ANALYZER_TOOL,
            tags=["analysis", "tool"]
        )
        print("✓ Created DocumentAnalyzer tool (with fixed string literals)")
        
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
        print("✓ Created CoordinatorAgent")
        
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
        print("✓ Created SpecialistAgent")
        
        print("\nSimplified agent library setup complete!")
        print(f"Library saved to: {library.storage_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())