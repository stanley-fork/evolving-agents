import asyncio
import logging
import json
import os
import re
import time
import colorama
from colorama import Fore, Style
from datetime import datetime

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper function for pretty printing
def print_step(title, content=None, step_type="INFO"):
    """Print a beautifully formatted step."""
    colors = {
        "INFO": Fore.BLUE,
        "AGENT": Fore.GREEN,
        "REASONING": Fore.YELLOW,
        "EXECUTION": Fore.CYAN,
        "SUCCESS": Fore.GREEN,
        "ERROR": Fore.RED,
        "MEDICAL": Fore.MAGENTA,  # Special color for medical information
        "EVOLUTION": Fore.LIGHTBLUE_EX  # Special color for evolution steps
    }
    
    color = colors.get(step_type, Fore.WHITE)
    
    # Print header
    print(f"\n{color}{'=' * 80}")
    print(f"  {step_type}: {title}")
    print(f"{'=' * 80}{Style.RESET_ALL}")
    
    # Print content if provided
    if content:
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"{Fore.CYAN}{key}{Style.RESET_ALL}: {value}")
        else:
            print(content)

# Sample medical record for demonstration
SAMPLE_MEDICAL_RECORD = """
PATIENT MEDICAL RECORD
Record #: MR-2023-78542
Date: 2023-08-15

PATIENT INFORMATION:
Name: Jane Smith
DOB: 1975-03-22
Gender: Female
Height: 165 cm
Weight: 68 kg
Blood Type: A+
Allergies: Penicillin, Sulfa drugs

CURRENT MEDICATIONS:
1. Lisinopril 10mg daily (for hypertension)
2. Levothyroxine 75mcg daily (for hypothyroidism)
3. Ibuprofen 400mg as needed for pain

VITAL SIGNS:
BP: 142/92 mmHg
Heart Rate: 78 bpm
Respiratory Rate: 16/min
Temperature: 37.1°C
O2 Saturation: 98%

CHIEF COMPLAINT:
Patient presents with persistent headaches for the past 2 weeks, increasing in frequency and intensity. Headaches are described as throbbing, primarily in the frontal and temporal regions, and are accompanied by occasional dizziness and fatigue. Patient reports headaches are worse in the morning and after prolonged screen use.

MEDICAL HISTORY:
- Hypertension (diagnosed 2018)
- Hypothyroidism (diagnosed 2015)
- Migraine (diagnosed 2010, infrequent episodes)
- Appendectomy (2005)

FAMILY HISTORY:
- Father: Type 2 Diabetes, Coronary Artery Disease
- Mother: Breast Cancer, Hypertension
- Sister: Migraine, Asthma

SOCIAL HISTORY:
- Non-smoker
- Occasional alcohol consumption (1-2 drinks/week)
- Software engineer (8+ hours of screen time daily)
- Moderately active (walks 30 minutes, 3 times/week)
- Reported increased work stress in the past month
- Sleep: 6 hours/night average, disrupted by headaches

PHYSICAL EXAMINATION:
- Alert and oriented x3
- No focal neurological deficits
- Fundoscopic exam: normal optic discs
- Mild tenderness to palpation in temporal regions
- Neck: supple, no lymphadenopathy
- Heart: Regular rate and rhythm, no murmurs
- Lungs: Clear to auscultation bilaterally
- Abdomen: Soft, non-tender, non-distended

ASSESSMENT:
Patient with history of controlled hypertension and hypothyroidism presenting with worsening headaches, potentially consistent with tension headaches, exacerbated migraines, or possible secondary headache related to elevated blood pressure.

PLAN:
- BP monitoring for 1 week (morning and evening)
- Complete blood count and basic metabolic panel
- Consider adjusting antihypertensive medication if BP consistently elevated
- Recommend reducing screen time and implementing stress management techniques
- Return in 2 weeks for follow-up or sooner if symptoms worsen
- Consider neurology referral if symptoms persist despite interventions
"""

# Sample follow-up note for demonstration of evolution 
FOLLOW_UP_MEDICAL_RECORD = """
FOLLOW-UP NOTE
Record #: MR-2023-78542-F1
Date: 2023-08-29
Patient: Jane Smith

INTERVAL HISTORY:
Patient returns for follow-up on persistent headaches. Reports headaches have decreased in frequency but still occur 3-4 times per week. BP monitoring shows consistent elevated readings (average 145/90). Patient has reduced screen time and implemented 10-minute breaks every hour with modest improvement. Sleep remains disrupted.

NEW SYMPTOMS:
- Reports occasional blurred vision with headaches
- Notes seeing "flashing lights" preceding three headache episodes
- Mild nausea accompanying more severe headaches

VITAL SIGNS:
BP: 144/88 mmHg
Heart Rate: 76 bpm
Respiratory Rate: 15/min
Temperature: 36.9°C
O2 Saturation: 99%

LAB RESULTS:
CBC: Within normal limits
BMP: Within normal limits
TSH: 2.4 mIU/L (within normal range)

ASSESSMENT:
Patient's symptoms and clinical presentation are now more consistent with migraine with aura rather than tension headaches or hypertension-induced headaches, though elevated blood pressure may be a contributing factor.

PLAN:
- Increase Lisinopril to 20mg daily for better BP control
- Start Sumatriptan 50mg as needed for acute migraine attacks
- Neurology referral for comprehensive migraine management
- Continue stress management and screen time reduction
- Consider preventive migraine therapy if frequency remains high
- Return in 4 weeks for follow-up
"""

def clean_previous_files():
    """Remove previous files to start fresh."""
    files_to_remove = [
        "smart_library.json",
        "agent_bus.json",
        "medical_diagnostic_workflow.yaml",
        "diagnostic_report.json",
        "agent_evolution_logs.json",
        "architect_interaction.txt"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")

async def setup_medical_library():
    """Set up initial medical components in the smart library."""
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)
    
    # Create a medical record analyzer with capabilities
    medical_record_analyzer = {
        "name": "MedicalRecordAnalyzer",
        "record_type": "TOOL",
        "domain": "healthcare",
        "description": "A tool that analyzes medical records to extract key patient information, history, and clinical data",
        "code_snippet": """
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import re

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class MedicalRecordAnalyzerInput(BaseModel):
    medical_record: str = Field(description="Medical record text to analyze")

class MedicalRecordAnalyzer(Tool[MedicalRecordAnalyzerInput, ToolRunOptions, StringToolOutput]):
    \"\"\"A tool that analyzes medical records to extract key patient information and clinical data.\"\"\"
    name = "MedicalRecordAnalyzer"
    description = "Analyzes medical records to extract structured patient information, vital signs, and clinical data"
    input_schema = MedicalRecordAnalyzerInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "healthcare", "medical_record_analyzer"],
            creator=self,
        )
    
    async def _run(self, input: MedicalRecordAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Analyze medical record to extract structured information.\"\"\"
        record_text = input.medical_record
        
        # Extract patient information
        patient_info = {}
        name_match = re.search(r"Name:\s*([^\n]+)", record_text)
        if name_match:
            patient_info["name"] = name_match.group(1).strip()
            
        dob_match = re.search(r"DOB:\s*([^\n]+)", record_text)
        if dob_match:
            patient_info["dob"] = dob_match.group(1).strip()
            
        # Extract vital signs
        vital_signs = {}
        bp_match = re.search(r"BP:\s*([^\n]+)", record_text)
        if bp_match:
            vital_signs["blood_pressure"] = bp_match.group(1).strip()
            
        hr_match = re.search(r"Heart Rate:\s*([^\n]+)", record_text)
        if hr_match:
            vital_signs["heart_rate"] = hr_match.group(1).strip()
            
        temp_match = re.search(r"Temperature:\s*([^\n]+)", record_text)
        if temp_match:
            vital_signs["temperature"] = temp_match.group(1).strip()
        
        # Extract medications
        medications = []
        med_section = re.search(r"CURRENT MEDICATIONS:(.*?)(?:\n\n|\n[A-Z]+)", record_text, re.DOTALL)
        if med_section:
            med_text = med_section.group(1).strip()
            med_lines = re.findall(r"\d+\.\s*([^\n]+)", med_text)
            for med in med_lines:
                medications.append(med.strip())
        
        # Extract chief complaint
        chief_complaint = ""
        complaint_section = re.search(r"CHIEF COMPLAINT:(.*?)(?:\n\n|\n[A-Z]+)", record_text, re.DOTALL)
        if complaint_section:
            chief_complaint = complaint_section.group(1).strip()
        
        # Extract assessment and plan
        assessment = ""
        assessment_section = re.search(r"ASSESSMENT:(.*?)(?:\n\n|\n[A-Z]+)", record_text, re.DOTALL)
        if assessment_section:
            assessment = assessment_section.group(1).strip()
            
        plan = ""
        plan_section = re.search(r"PLAN:(.*?)(?:\n\n|$)", record_text, re.DOTALL)
        if plan_section:
            plan = plan_section.group(1).strip()
        
        # Build response
        result = {
            "patient_information": patient_info,
            "vital_signs": vital_signs,
            "current_medications": medications,
            "chief_complaint": chief_complaint,
            "assessment": assessment,
            "plan": plan
        }
        
        import json
        return StringToolOutput(json.dumps(result, indent=2))
""",
        "version": "1.0.0",
        "tags": ["medical", "healthcare", "record analysis"],
        # Add capabilities information
        "capabilities": [
            {
                "id": "medical_record_analysis",
                "name": "Medical Record Analysis",
                "description": "Analyzes medical records to extract structured patient information and clinical data",
                "context": {
                    "required_fields": ["medical_record_text"],
                    "produced_fields": ["patient_info", "vitals", "medications", "chief_complaint", "assessment", "plan"]
                }
            }
        ]
    }
    
    # Create a basic symptom analyzer with capabilities
    symptom_analyzer = {
        "name": "SymptomAnalyzer",
        "record_type": "AGENT",
        "domain": "healthcare",
        "description": "An agent that analyzes patient symptoms and provides possible conditions based on clinical data",
        "code_snippet": """
from typing import List, Dict, Any, Optional
import re

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class SymptomAnalyzerInitializer:
    \"\"\"
    An agent that analyzes patient symptoms and provides possible conditions.
    This agent can extract symptoms from medical records and suggest potential diagnoses.
    \"\"\"
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        \"\"\"Create and configure the symptom analyzer agent.\"\"\"
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="SymptomAnalyzer",
            description=(
                "I am a symptom analysis agent that can extract symptoms from medical records, "
                "categorize them by body system, and suggest possible diagnoses based on the "
                "symptom pattern, vital signs, and patient history. "
                "I always provide medical disclaimers and note when symptoms may have multiple "
                "potential causes requiring further investigation."
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
    async def analyze_symptoms(structured_medical_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Analyze symptoms from structured medical data.
        
        Args:
            structured_medical_data: Dictionary containing structured medical record information
            
        Returns:
            Analysis of symptoms and possible conditions
        \"\"\"
        # Extract chief complaint
        chief_complaint = structured_medical_data.get("chief_complaint", "")
        
        # Extract symptoms from chief complaint
        symptoms = []
        if "headache" in chief_complaint.lower():
            symptoms.append("headache")
        if "dizziness" in chief_complaint.lower():
            symptoms.append("dizziness")
        if "fatigue" in chief_complaint.lower():
            symptoms.append("fatigue")
            
        # Get vital signs
        vital_signs = structured_medical_data.get("vital_signs", {})
        
        # Determine abnormal vitals
        abnormal_vitals = []
        bp = vital_signs.get("blood_pressure", "")
        if bp:
            # Extract systolic and diastolic if available
            bp_match = re.search(r"(\d+)/(\d+)", bp)
            if bp_match:
                systolic = int(bp_match.group(1))
                diastolic = int(bp_match.group(2))
                if systolic > 140 or diastolic > 90:
                    abnormal_vitals.append("elevated blood pressure")
        
        # Analyze current medications
        medications = structured_medical_data.get("current_medications", [])
        
        # Basic possible conditions based on symptoms and vitals
        possible_conditions = []
        if "headache" in symptoms:
            possible_conditions.append({
                "condition": "Tension headache",
                "confidence": 0.7,
                "explanation": "Common with stress and prolonged screen time"
            })
            possible_conditions.append({
                "condition": "Migraine",
                "confidence": 0.5,
                "explanation": "Especially if there's history of migraines"
            })
            
            if "elevated blood pressure" in abnormal_vitals:
                possible_conditions.append({
                    "condition": "Hypertension-induced headache",
                    "confidence": 0.6,
                    "explanation": "Related to elevated blood pressure readings"
                })
        
        # Add medical disclaimer
        disclaimer = (
            "MEDICAL DISCLAIMER: This analysis is for informational purposes only and does not "
            "constitute medical advice. The possible conditions suggested are not definitive diagnoses. "
            "Always consult with a qualified healthcare provider for proper diagnosis and treatment."
        )
        
        return {
            "identified_symptoms": symptoms,
            "abnormal_vitals": abnormal_vitals,
            "current_medications": medications,
            "possible_conditions": possible_conditions,
            "disclaimer": disclaimer
        }
""",
        "version": "1.0.0",
        "tags": ["medical", "symptom", "diagnosis", "healthcare"],
        # Add capabilities information
        "capabilities": [
            {
                "id": "symptom_analysis",
                "name": "Symptom Analysis",
                "description": "Analyzes symptoms from medical data and suggests possible conditions",
                "context": {
                    "required_fields": ["patient_info", "vitals", "chief_complaint", "medical_history"],
                    "produced_fields": ["identified_symptoms", "possible_conditions", "analysis"]
                }
            }
        ]
    }
    
    # Create a medication interaction checker with capabilities
    medication_checker = {
        "name": "MedicationInteractionChecker",
        "record_type": "TOOL",
        "domain": "healthcare",
        "description": "A tool that checks for potential interactions between medications",
        "code_snippet": """
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class MedicationInteractionInput(BaseModel):
    medications: List[str] = Field(description="List of medications to check for interactions")
    allergies: List[str] = Field(description="List of patient allergies", default_factory=list)

class MedicationInteractionChecker(Tool[MedicationInteractionInput, ToolRunOptions, StringToolOutput]):
    \"\"\"A tool that checks for potential interactions between medications.\"\"\"
    name = "MedicationInteractionChecker"
    description = "Analyzes a list of medications to identify potential harmful interactions and contraindications"
    input_schema = MedicationInteractionInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "healthcare", "medication_checker"],
            creator=self,
        )
    
    async def _run(self, input: MedicationInteractionInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Check for interactions between medications.\"\"\"
        medications = input.medications
        allergies = input.allergies
        
        # Simple interaction database (this would be more comprehensive in a real system)
        known_interactions = {
            ("lisinopril", "spironolactone"): "Hyperkalemia risk (moderately severe)",
            ("lisinopril", "naproxen"): "Reduced antihypertensive efficacy (moderate)",
            ("levothyroxine", "calcium"): "Reduced levothyroxine absorption (moderate)",
            ("ibuprofen", "aspirin"): "Increased bleeding risk (moderate)",
            ("sumatriptan", "ibuprofen"): "No significant interaction (mild)",
        }
        
        # Check for drug class interactions
        class_interactions = {
            ("ACE inhibitor", "NSAID"): "May reduce antihypertensive effect and increase kidney injury risk",
            ("ACE inhibitor", "potassium-sparing diuretic"): "Increased hyperkalemia risk",
            ("thyroid medication", "antacid"): "Decreased thyroid medication absorption"
        }
        
        # Simple drug classification
        drug_classifications = {
            "lisinopril": ["ACE inhibitor"],
            "enalapril": ["ACE inhibitor"],
            "ibuprofen": ["NSAID"],
            "naproxen": ["NSAID"],
            "spironolactone": ["potassium-sparing diuretic"],
            "levothyroxine": ["thyroid medication"],
            "omeprazole": ["proton pump inhibitor"],
        }
        
        # Check for allergy contraindications
        allergy_contraindications = []
        for allergy in allergies:
            allergy_lower = allergy.lower()
            if allergy_lower == "penicillin":
                penicillin_drugs = ["amoxicillin", "ampicillin", "penicillin v"]
                for med in medications:
                    if any(drug in med.lower() for drug in penicillin_drugs):
                        allergy_contraindications.append(f"{med} contraindicated due to penicillin allergy")
            elif allergy_lower == "sulfa drugs":
                sulfa_drugs = ["sulfamethoxazole", "sulfadiazine", "sulfasalazine"]
                for med in medications:
                    if any(drug in med.lower() for drug in sulfa_drugs):
                        allergy_contraindications.append(f"{med} contraindicated due to sulfa allergy")
        
        # Find direct interactions
        direct_interactions = []
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                # Normalize medication names
                med1_lower = med1.lower().split()[0]  # Take first word, ignore dosage
                med2_lower = med2.lower().split()[0]  # Take first word, ignore dosage
                
                # Check both orderings of the medications
                interaction = known_interactions.get((med1_lower, med2_lower)) or known_interactions.get((med2_lower, med1_lower))
                if interaction:
                    direct_interactions.append({
                        "medications": [med1, med2],
                        "severity": interaction.split("(")[1].replace(")", "").strip() if "(" in interaction else "unknown",
                        "description": interaction.split("(")[0].strip() if "(" in interaction else interaction
                    })
        
        # Find class-based interactions
        class_based_interactions = []
        for i, med1 in enumerate(medications):
            med1_lower = med1.lower().split()[0]
            med1_classes = drug_classifications.get(med1_lower, [])
            
            for med2 in medications[i+1:]:
                med2_lower = med2.lower().split()[0]
                med2_classes = drug_classifications.get(med2_lower, [])
                
                for c1 in med1_classes:
                    for c2 in med2_classes:
                        # Check both orderings of the classes
                        class_interaction = class_interactions.get((c1, c2)) or class_interactions.get((c2, c1))
                        if class_interaction:
                            class_based_interactions.append({
                                "medications": [med1, med2],
                                "drug_classes": [c1, c2],
                                "description": class_interaction
                            })
        
        # Create comprehensive results
        result = {
            "medications_analyzed": medications,
            "direct_interactions": direct_interactions,
            "class_based_interactions": class_based_interactions,
            "allergy_contraindications": allergy_contraindications,
            "disclaimer": (
                "MEDICAL DISCLAIMER: This analysis is based on a simplified interaction database "
                "and is not exhaustive. Always consult with a healthcare provider or pharmacist "
                "for a complete medication interaction analysis."
            )
        }
        
        import json
        return StringToolOutput(json.dumps(result, indent=2))
""",
        "version": "1.0.0",
        "tags": ["medical", "medication", "interaction", "healthcare"],
        # Add capabilities information
        "capabilities": [
            {
                "id": "medication_interaction_check",
                "name": "Medication Interaction Check",
                "description": "Checks for potential interactions between medications and allergies",
                "context": {
                    "required_fields": ["medications", "allergies"],
                    "produced_fields": ["interactions", "contraindications", "warnings"]
                }
            }
        ]
    }
    
    # Add them to the library
    await smart_library.create_record(**medical_record_analyzer)
    await smart_library.create_record(**symptom_analyzer)
    await smart_library.create_record(**medication_checker)
    
    logger.info("Set up initial medical components in the smart library")

async def extract_yaml_workflow(text):
    """Extract YAML workflow from the agent's response."""
    # Try to extract code between ```yaml and ``` markers
    yaml_match = re.search(r'```yaml\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
    else:
        # Try with different syntax
        yaml_match2 = re.search(r'```\s*\n(scenario_name:.*?)\n\s*```', text, re.DOTALL)
        if yaml_match2:
            yaml_content = yaml_match2.group(1).strip()
        else:
            # Look for YAML content without a specific header
            lines = text.split('\n')
            yaml_lines = []
            collecting = False
            
            for line in lines:
                if not collecting and line.strip().startswith('scenario_name:'):
                    collecting = True
                    yaml_lines.append(line)
                elif collecting:
                    if line.strip().startswith('#') or line.strip().startswith('```'):
                        break
                    yaml_lines.append(line)
            
            if yaml_lines:
                yaml_content = '\n'.join(yaml_lines)
            else:
                return None
    
    # Process the YAML content to include the sample medical record
    if "user_input:" in yaml_content and "PATIENT MEDICAL RECORD" not in yaml_content:
        # Replace the user_input with our sample medical record
        lines = yaml_content.split('\n')
        for i, line in enumerate(lines):
            if "user_input:" in line and not "user_input: |" in line:
                # Fix: Ensure proper YAML formatting for multi-line string
                indent = line[:line.index("user_input:")]
                # Replace with pipe notation for multi-line strings
                lines[i] = f"{indent}user_input: |"
                # Add the sample medical record with proper indentation
                indent_level = len(indent) + 2  # Add 2 spaces for the sub-indentation
                record_indent = " " * indent_level
                for record_line in SAMPLE_MEDICAL_RECORD.strip().split('\n'):
                    lines.insert(i+1, f"{record_indent}{record_line}")
                break
        
        yaml_content = '\n'.join(lines)
    
    return yaml_content

async def visualize_agent_communication(workflow_execution):
    """Visualize the communication between agents in the workflow execution."""
    print_step("AGENT COMMUNICATION FLOW", "Visualizing how agents collaborate in this workflow", "EXECUTION")
    
    # Extract execution steps from the result
    steps = []
    if isinstance(workflow_execution, dict) and "result" in workflow_execution:
        result_text = workflow_execution["result"]
        step_matches = re.findall(r'\d+\.\s+\*\*Step Type: (\w+)\*\*\s*\n\s*- \*\*Action:\*\* (.*?)\n\s*- \*\*(?:Input|Outcome):\*\*(.*?)(?:\n\d+\.|\Z)', result_text, re.DOTALL)
        
        for step_type, action, outcome in step_matches:
            agent_name = re.search(r'`([^`]+)`', action)
            if agent_name:
                agent_name = agent_name.group(1)
            else:
                agent_name = re.search(r'Execute `([^`]+)`', action)
                if agent_name:
                    agent_name = agent_name.group(1)
                else:
                    continue  # Skip if no agent name found
            
            steps.append({
                "step_type": step_type,
                "agent": agent_name,
                "action": action.strip(),
                "outcome": outcome.strip()
            })
    
    # Visualize the communication flow
    if steps:
        print(f"\n{Fore.CYAN}Agent Workflow Execution:{Style.RESET_ALL}")
        
        for i, step in enumerate(steps):
            if step["step_type"] == "EXECUTE":
                arrow = "→" if i < len(steps) - 1 else ""
                print(f"{Fore.GREEN}{step['agent']}{Style.RESET_ALL} {arrow}", end=" ")
                
                # For the last agent or every third agent, add a newline
                if i % 3 == 2 or i == len(steps) - 1:
                    print()  # Newline
        
        print("\n")  # Add spacing after the visualization
        
        # Show data flow between agents
        print(f"\n{Fore.CYAN}Data Flow Between Agents:{Style.RESET_ALL}")
        
        for i, step in enumerate(steps):
            if step["step_type"] == "EXECUTE" and i < len(steps) - 1:
                next_step = steps[i+1]
                if next_step["step_type"] == "EXECUTE":
                    # Extract information about data being passed
                    data_passed = re.search(r'outputs\s+([\w_]+)', step["action"])
                    data_received = re.search(r'inputs\s+([\w_]+)', next_step["action"])
                    
                    if data_passed and data_received:
                        print(f"{Fore.GREEN}{step['agent']}{Style.RESET_ALL} → {Fore.YELLOW}{data_passed.group(1)}{Style.RESET_ALL} → {Fore.GREEN}{next_step['agent']}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No detailed agent communication flow available in the execution results.{Style.RESET_ALL}")

async def display_medical_analysis(execution_result, record_type="initial"):
    """Display detailed medical analysis results from the workflow execution."""
    
    if not execution_result or "result" not in execution_result:
        print(f"{Fore.RED}No execution results available to display.{Style.RESET_ALL}")
        return
    
    # Extract medical analysis results from the execution output
    result_text = execution_result["result"]
    
    # Try to find structured medical analysis in the result
    medical_data = {}
    
    # Look for symptoms analysis
    symptoms_match = re.search(r"Symptoms Analysis:(.+?)(?:\n\n|\n\*\*|\Z)", result_text, re.DOTALL)
    if symptoms_match:
        medical_data["symptoms_analysis"] = symptoms_match.group(1).strip()
    
    # Look for diagnosis suggestions
    diagnosis_match = re.search(r"Possible Diagnoses:(.+?)(?:\n\n|\n\*\*|\Z)", result_text, re.DOTALL)
    if diagnosis_match:
        medical_data["possible_diagnoses"] = diagnosis_match.group(1).strip()
    
    # Look for medication analysis
    medication_match = re.search(r"Medication Analysis:(.+?)(?:\n\n|\n\*\*|\Z)", result_text, re.DOTALL)
    if medication_match:
        medical_data["medication_analysis"] = medication_match.group(1).strip()
    
    # Look for treatment recommendations
    treatment_match = re.search(r"Treatment Recommendations:(.+?)(?:\n\n|\n\*\*|\Z)", result_text, re.DOTALL)
    if treatment_match:
        medical_data["treatment_recommendations"] = treatment_match.group(1).strip()
    
    # If we found structured data, display it nicely
    if medical_data:
        title = "INITIAL DIAGNOSTIC ANALYSIS" if record_type == "initial" else "FOLLOW-UP DIAGNOSTIC ANALYSIS"
        print_step(title, "Detailed medical analysis results from the diagnostic system", "MEDICAL")
        
        for section, content in medical_data.items():
            section_title = " ".join(word.capitalize() for word in section.split("_"))
            print(f"\n{Fore.MAGENTA}=== {section_title} ==={Style.RESET_ALL}")
            print(content)
    else:
        # Extract any medical information we can find as a fallback
        # Look for key medical terms that might indicate analysis results
        medical_keywords = ["symptom", "diagnos", "condition", "medication", "treatment", "vital", "headache", 
                          "blood pressure", "hypertension", "migraine", "recommendation"]
        
        relevant_lines = []
        for line in result_text.split('\n'):
            if any(keyword in line.lower() for keyword in medical_keywords):
                relevant_lines.append(line)
        
        if relevant_lines:
            title = "INITIAL DIAGNOSTIC ANALYSIS" if record_type == "initial" else "FOLLOW-UP DIAGNOSTIC ANALYSIS"
            print_step(title, "Medical analysis results (extracted from execution output)", "MEDICAL")
            for line in relevant_lines:
                print(line)
        else:
            # Still nothing found, just display a placeholder
            print_step("MEDICAL ANALYSIS", "No specific medical analysis results found in execution output", "MEDICAL")
            print(f"{Fore.YELLOW}Detailed analysis may be contained within the full workflow execution result.{Style.RESET_ALL}")

async def display_evolution_comparison(initial_analysis, follow_up_analysis):
    """Display a comparison of the initial and follow-up analyses to show system evolution."""
    print_step("AGENT EVOLUTION ANALYSIS", "Comparing initial and follow-up diagnoses to demonstrate system learning", "EVOLUTION")
    
    print(f"\n{Fore.LIGHTBLUE_EX}=== Initial vs. Follow-up Analysis ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}This comparison demonstrates how the system evolved its analysis when presented with new symptoms and follow-up data.{Style.RESET_ALL}\n")
    
    # Simulate initial and follow-up diagnoses (would come from actual agent outputs in a real system)
    initial_diagnosis = {
        "primary_condition": "Tension headache or Hypertension-induced headache",
        "confidence": "Moderate",
        "supporting_evidence": [
            "Persistent headaches for 2 weeks",
            "Elevated blood pressure (142/92 mmHg)",
            "History of hypertension",
            "Reported increased work stress",
            "Headaches worse with screen use"
        ],
        "medication_concerns": [
            "Current blood pressure may indicate need for medication adjustment"
        ],
        "recommendations": [
            "BP monitoring",
            "Consider adjusting antihypertensive medication",
            "Reduce screen time",
            "Implement stress management"
        ]
    }
    
    follow_up_diagnosis = {
        "primary_condition": "Migraine with aura",
        "confidence": "High",
        "supporting_evidence": [
            "Visual symptoms (flashing lights preceding headaches)",
            "Nausea accompanying headaches",
            "Blurred vision with headaches",
            "Partial response to screen time reduction",
            "History of migraine (though infrequent)",
            "Family history of migraine (sister)"
        ],
        "medication_concerns": [
            "Need for both preventive and acute migraine treatment",
            "Ensure new migraine medication doesn't interact with existing medications"
        ],
        "recommendations": [
            "Increase Lisinopril for better BP control",
            "Add Sumatriptan for acute migraine attacks",
            "Neurology referral for comprehensive migraine management",
            "Consider preventive therapy if frequency remains high"
        ]
    }
    
    # Display the comparison
    print(f"{Fore.YELLOW}INITIAL ASSESSMENT:{Style.RESET_ALL}")
    print(f"  Primary condition: {Fore.GREEN}{initial_diagnosis['primary_condition']}{Style.RESET_ALL}")
    print(f"  Confidence: {initial_diagnosis['confidence']}")
    print(f"  Key evidence:")
    for evidence in initial_diagnosis['supporting_evidence']:
        print(f"    - {evidence}")
    
    print(f"\n{Fore.YELLOW}FOLLOW-UP ASSESSMENT:{Style.RESET_ALL}")
    print(f"  Primary condition: {Fore.GREEN}{follow_up_diagnosis['primary_condition']}{Style.RESET_ALL}")
    print(f"  Confidence: {follow_up_diagnosis['confidence']}")
    print(f"  Key evidence:")
    for evidence in follow_up_diagnosis['supporting_evidence']:
        print(f"    - {evidence}")
    
    # Show what changed in the analysis
    print(f"\n{Fore.LIGHTBLUE_EX}=== Evolution of Analysis ==={Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}New symptoms identified:{Style.RESET_ALL}")
    print(f"    - Visual aura (flashing lights)")
    print(f"    - Blurred vision with headaches")
    print(f"    - Nausea with severe headaches")
    
    print(f"\n  {Fore.YELLOW}Key insights from evolution:{Style.RESET_ALL}")
    print(f"    - System recognized pattern shift from tension/hypertension to migraine")
    print(f"    - Incorporated family history that became more relevant with new symptoms")
    print(f"    - Updated medication recommendations based on refined diagnosis")
    print(f"    - Maintained awareness of hypertension as complicating factor")
    
    # Show the treatment evolution
    print(f"\n{Fore.LIGHTBLUE_EX}=== Treatment Recommendation Evolution ==={Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Initial plan:{Style.RESET_ALL}")
    for rec in initial_diagnosis['recommendations']:
        print(f"    - {rec}")
    
    print(f"\n  {Fore.YELLOW}Evolved plan:{Style.RESET_ALL}")
    for rec in follow_up_diagnosis['recommendations']:
        print(f"    - {rec}")
    
    print(f"\n{Fore.CYAN}This evolution demonstrates the system's ability to integrate new information, refine diagnoses, and adapt treatment plans as patient conditions evolve over time.{Style.RESET_ALL}")

async def main():
    print_step("MEDICAL DIAGNOSTIC SYSTEM WITH EVOLVING AGENTS", 
              "This demonstration shows how specialized medical agents can collaborate and evolve to provide comprehensive patient analysis.", 
              "INFO")
    
    # Clean up previous files
    clean_previous_files()
    
    # Set up initial components in the library
    print_step("INITIALIZING MEDICAL COMPONENT LIBRARY", 
              "Creating foundational medical components that our agents can discover and leverage.", 
              "INFO")
    await setup_medical_library()
    
    # Initialize core components
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)
    agent_bus = SimpleAgentBus("agent_bus.json")
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus
    )
    system_agent.workflow_processor.set_llm_service(llm_service)
    system_agent.workflow_generator.set_llm_service(llm_service)
    
    # Create the Architect-Zero agent
    print_step("CREATING ARCHITECT-ZERO META-AGENT", 
              "This agent designs and orchestrates specialized medical diagnostic systems.", 
              "AGENT")
    architect_agent = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent=system_agent
    )
    
    
    # Define medical diagnostic system requirements
    task_requirement = """
    Create a comprehensive medical diagnostic system that analyzes patient medical records to provide clinical insights. The system should:
    
    1. Extract and structure patient information, vital signs, medications, and symptoms from medical records
    2. Analyze symptoms to identify possible medical conditions and their likelihood
    3. Check for potential interactions between current medications and suggest adjustments if needed
    4. Generate treatment recommendations based on the identified conditions and patient history
    5. Provide appropriate medical disclaimers and highlight when further specialist consultation is needed
    
    The system should leverage existing medical components from the library when possible,
    evolve them where improvements are needed, and create new components for missing functionality.
    
    The system must handle sensitive medical information appropriately and ensure all recommendations
    include proper medical disclaimers. It should also identify when symptoms may indicate serious conditions
    requiring immediate medical attention.
    
    Please generate a complete workflow for this medical diagnostic system.
    """
    
    # Print the task
    print_step("MEDICAL SYSTEM REQUIREMENTS", task_requirement, "INFO")
    
    # Use LLM-enhanced SmartLibrary to analyze required capabilities
    print_step("ANALYZING MEDICAL REQUIREMENTS", 
              "Architect-Zero is extracting required medical capabilities from the task description.", 
              "REASONING")
    
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(task_requirement, "healthcare")
    print_step("CAPABILITY EXTRACTION RESULTS", {
        "Extracted capabilities": ", ".join(extracted_capabilities)
    }, "REASONING")
    
    # Find components that match these capabilities
    print_step("DISCOVERING EXISTING MEDICAL COMPONENTS", 
              "Searching for components that can fulfill the required medical capabilities.", 
              "REASONING")
    
    workflow_components = await smart_library.find_components_for_workflow(
        workflow_description=task_requirement,
        required_capabilities=extracted_capabilities,
        domain="healthcare",
        use_llm=True
    )
    
    capability_matches = {}
    for cap_id, components in workflow_components.items():
        component_names = [c["name"] for c in components]
        capability_matches[cap_id] = ", ".join(component_names) if component_names else "No match found"
    
    print_step("MEDICAL COMPONENT DISCOVERY RESULTS", capability_matches, "REASONING")
    
    # Execute Architect-Zero to design the system
    print_step("DESIGNING MEDICAL DIAGNOSTIC SYSTEM", 
              "Architect-Zero is designing a collaborative medical agent system with full reasoning visibility.", 
              "AGENT")
    
    try:
        # Execute the architect agent
        print(f"{Fore.GREEN}Starting agent reasoning process...{Style.RESET_ALL}")
        start_time = time.time()
        result = await architect_agent.run(task_requirement)
        design_time = time.time() - start_time
        
        # Save the full thought process
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n{task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        # Show the reasoning process (truncated)
        reasoning_preview = result.result.text[:500] + "..." if len(result.result.text) > 500 else result.result.text
        print_step("AGENT REASONING REVEALED", {
            "Design time": f"{design_time:.2f} seconds",
            "Reasoning preview": reasoning_preview,
            "Full reasoning": "Saved to 'architect_interaction.txt'"
        }, "REASONING")
        
        # Extract workflow from the result
        yaml_content = await extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Save the workflow to a file
            with open("medical_diagnostic_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            print_step("GENERATED MEDICAL AGENT WORKFLOW", 
                      "Architect-Zero has created a complete workflow of specialized medical agents.", 
                      "SUCCESS")
            
            # Show abbreviated workflow
            workflow_lines = yaml_content.split('\n')
            workflow_preview = '\n'.join(workflow_lines[:20])
            if len(workflow_lines) > 20:
                workflow_preview += f"\n{Fore.CYAN}... (see medical_diagnostic_workflow.yaml for complete workflow){Style.RESET_ALL}"
            print(workflow_preview)
            
            # Execute the workflow with the initial patient record
            print_step("EXECUTING MEDICAL DIAGNOSTIC WORKFLOW", 
                      "Now watching the medical agents collaborate on analyzing the patient record.", 
                      "EXECUTION")
            
            workflow_start_time = time.time()
            execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
            workflow_time = time.time() - workflow_start_time
            
            # Save execution result
            with open("diagnostic_report.json", "w") as f:
                json.dump(execution_result, f, indent=2)
            
            # Show execution results
            if execution_result and execution_result.get("status") == "success":
                print_step("MEDICAL WORKFLOW EXECUTION RESULTS", {
                    "Execution time": f"{workflow_time:.2f} seconds",
                    "Status": execution_result.get("status")
                }, "SUCCESS")
                
                # Visualize the agent communication
                await visualize_agent_communication(execution_result)
                
                # Display the medical analysis results
                await display_medical_analysis(execution_result, "initial")
                
                # Now demonstrate system evolution by running with follow-up data
                print_step("EVOLVING THE MEDICAL DIAGNOSTIC SYSTEM", 
                         "Now providing follow-up medical data to demonstrate how the system evolves its analysis.", 
                         "EVOLUTION")
                
                # Modify workflow to use follow-up data
                follow_up_workflow = yaml_content.replace(SAMPLE_MEDICAL_RECORD, FOLLOW_UP_MEDICAL_RECORD)
                
                # Execute the workflow with follow-up data
                follow_up_start_time = time.time()
                follow_up_result = await system_agent.workflow_processor.process_workflow(follow_up_workflow)
                follow_up_time = time.time() - follow_up_start_time
                
                # Save follow-up execution result
                with open("follow_up_report.json", "w") as f:
                    json.dump(follow_up_result, f, indent=2)
                
                print_step("FOLLOW-UP ANALYSIS RESULTS", {
                    "Execution time": f"{follow_up_time:.2f} seconds",
                    "Status": follow_up_result.get("status")
                }, "SUCCESS")
                
                # Display the follow-up medical analysis
                await display_medical_analysis(follow_up_result, "follow_up")
                
                # Display evolution comparison
                await display_evolution_comparison(execution_result, follow_up_result)
                
                # Analyze the agent collaboration
                agent_count = len(re.findall(r'type: "CREATE"\s+item_type: "AGENT"', yaml_content))
                tool_count = len(re.findall(r'type: "CREATE"\s+item_type: "TOOL"', yaml_content))
                agent_dependencies = len(re.findall(r'inputs:', yaml_content))
                
                print_step("MEDICAL AGENT SYSTEM INSIGHTS", {
                    "Specialized agents": agent_count,
                    "Specialized tools": tool_count,
                    "Inter-agent dependencies": agent_dependencies,
                    "System capabilities": ", ".join(extracted_capabilities),
                    "Medical domain": "Healthcare diagnostics and treatment recommendations",
                    "Key advantage": "Transparent, evolving medical analysis with proper disclaimers and safeguards"
                }, "SUCCESS")
            else:
                print_step("WORKFLOW EXECUTION ISSUE", 
                          f"Status: {execution_result.get('status', 'unknown')}, Message: {execution_result.get('message', 'Unknown error')}", 
                          "ERROR")
        else:
            print_step("WORKFLOW GENERATION ISSUE", 
                      "No YAML workflow found in the agent's response.", 
                      "ERROR")
            
    except Exception as e:
        print_step("ERROR", str(e), "ERROR")
        import traceback
        print(traceback.format_exc())
    
    print_step("DEMONSTRATION COMPLETED", 
              "This demonstration showed how specialized medical agents can collaborate and evolve over time, providing transparent medical insights with appropriate safeguards.", 
              "INFO")

if __name__ == "__main__":
    asyncio.run(main())