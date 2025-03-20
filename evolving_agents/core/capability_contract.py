# evolving_agents/core/capability_contract.py

from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime

class ContextRequirement(BaseModel):
    """Definition of context requirements for a capability."""
    
    required_fields: Set[str] = Field(
        default_factory=set,
        description="Fields that must be present in the context"
    )
    optional_fields: Set[str] = Field(
        default_factory=set,
        description="Fields that enhance performance but aren't required"
    )
    produced_fields: Set[str] = Field(
        default_factory=set,
        description="Fields that this capability adds or modifies in the context"
    )
    field_schemas: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schemas for the context fields"
    )

class PerformanceMetrics(BaseModel):
    """Performance metrics for a capability."""
    
    total_executions: int = Field(
        default=0,
        description="Total number of execution attempts"
    )
    successful_executions: int = Field(
        default=0,
        description="Number of successful executions"
    )
    average_execution_time: float = Field(
        default=0.0,
        description="Average execution time in seconds"
    )
    last_execution_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last execution"
    )
    domain_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Success rates by domain"
    )

class CapabilityContract(BaseModel):
    """Full contract for a capability, used by both Library and Bus."""
    
    id: str = Field(description="Unique identifier for this capability")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Detailed description")
    version: str = Field(default="1.0.0", description="Capability version")
    
    context: ContextRequirement = Field(
        default_factory=ContextRequirement,
        description="Context requirements for this capability"
    )
    
    performance: PerformanceMetrics = Field(
        default_factory=PerformanceMetrics,
        description="Performance metrics for this capability"
    )
    
    domain: Optional[str] = Field(default=None, description="Domain this capability belongs to")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    
    example_inputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example inputs for this capability"
    )
    example_outputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example outputs for this capability"
    )
    
    provider_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Requirements for providers implementing this capability"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate the overall success rate for this capability."""
        if self.performance.total_executions == 0:
            return 0.0
        return self.performance.successful_executions / self.performance.total_executions

def dict_to_capability_contract(capability_dict: Dict[str, Any]) -> CapabilityContract:
    """Convert a dictionary representation to a CapabilityContract model."""
    # Handle context requirements
    context_dict = capability_dict.get("context", {})
    context = ContextRequirement(
        required_fields=set(context_dict.get("required_fields", [])),
        optional_fields=set(context_dict.get("optional_fields", [])),
        produced_fields=set(context_dict.get("produced_fields", [])),
        field_schemas=context_dict.get("field_schemas", {})
    )
    
    # Handle performance metrics
    perf_dict = capability_dict.get("performance", {})
    last_time = perf_dict.get("last_execution_time")
    if isinstance(last_time, str):
        try:
            last_time = datetime.fromisoformat(last_time)
        except ValueError:
            last_time = None
    
    performance = PerformanceMetrics(
        total_executions=perf_dict.get("total_executions", 0),
        successful_executions=perf_dict.get("successful_executions", 0),
        average_execution_time=perf_dict.get("average_execution_time", 0.0),
        last_execution_time=last_time,
        domain_performance=perf_dict.get("domain_performance", {})
    )
    
    # Create the capability contract
    return CapabilityContract(
        id=capability_dict.get("id", ""),
        name=capability_dict.get("name", ""),
        description=capability_dict.get("description", ""),
        version=capability_dict.get("version", "1.0.0"),
        context=context,
        performance=performance,
        domain=capability_dict.get("domain"),
        tags=capability_dict.get("tags", []),
        example_inputs=capability_dict.get("example_inputs", []),
        example_outputs=capability_dict.get("example_outputs", []),
        provider_requirements=capability_dict.get("provider_requirements", {})
    )

def capability_contract_to_dict(contract: CapabilityContract) -> Dict[str, Any]:
    """Convert a CapabilityContract model to a dictionary representation."""
    # Convert context to dict
    context_dict = {
        "required_fields": list(contract.context.required_fields),
        "optional_fields": list(contract.context.optional_fields),
        "produced_fields": list(contract.context.produced_fields),
        "field_schemas": contract.context.field_schemas
    }
    
    # Convert performance to dict
    last_time = contract.performance.last_execution_time
    performance_dict = {
        "total_executions": contract.performance.total_executions,
        "successful_executions": contract.performance.successful_executions,
        "average_execution_time": contract.performance.average_execution_time,
        "last_execution_time": last_time.isoformat() if last_time else None,
        "domain_performance": contract.performance.domain_performance
    }
    
    # Create the full dict
    return {
        "id": contract.id,
        "name": contract.name,
        "description": contract.description,
        "version": contract.version,
        "context": context_dict,
        "performance": performance_dict,
        "domain": contract.domain,
        "tags": contract.tags,
        "example_inputs": contract.example_inputs,
        "example_outputs": contract.example_outputs,
        "provider_requirements": contract.provider_requirements
    }