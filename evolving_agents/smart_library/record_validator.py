# evolving_agents/smart_library/record_validator.py

import re
from typing import Dict, List, Any, Optional

def validate_record(
    code: str,
    record_type: str,
    domain: str,
    firmware_content: str,
    required_disclaimers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate that a code snippet follows firmware rules and includes required disclaimers.
    
    Args:
        code: Code snippet to validate
        record_type: Type of record (AGENT or TOOL)
        domain: Domain of the record
        firmware_content: Firmware content for validation
        required_disclaimers: List of disclaimers that must be included
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check if code is empty
    if not code or len(code.strip()) < 10:
        issues.append("Code snippet is empty or too short")
        return {"valid": False, "issues": issues}
    
    # Check for docstrings
    if not re.search(r'""".*?"""', code, re.DOTALL) and not re.search(r"'''.*?'''", code, re.DOTALL):
        issues.append("Missing docstring")
    
    # Check for disallowed imports if mentioned in firmware
    disallowed_imports = []
    
    if "disallowedImports" in firmware_content or "disallowed_imports" in firmware_content:
        # Extract disallowed imports from firmware content
        if "os" in firmware_content.lower() and "subprocess" in firmware_content.lower():
            disallowed_imports = ["os", "subprocess"]
    
    for imp in disallowed_imports:
        if re.search(rf'import\s+{imp}|from\s+{imp}\s+import', code):
            issues.append(f"Disallowed import: {imp}")
    
    # Check for required disclaimers
    if required_disclaimers:
        for disclaimer in required_disclaimers:
            if disclaimer.strip() and disclaimer.strip() not in code:
                issues.append(f"Missing required disclaimer: {disclaimer}")
    
    # Domain-specific checks
    if domain == "medical" and "MEDICAL_DISCLAIMER" not in code:
        issues.append("Missing MEDICAL_DISCLAIMER in medical domain code")
    
    if domain == "finance" and "FINANCIAL_DISCLAIMER" not in code:
        issues.append("Missing FINANCIAL_DISCLAIMER in finance domain code")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }