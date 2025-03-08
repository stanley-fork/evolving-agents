# evolving_agents/tools/acp_tool.py

from typing import Dict, Any
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions
from beeai_framework.acp.tool import ACPToolInterface

class ACPToolWrapper(Tool):
    """
    A wrapper that makes standard BeeAI tools compatible with ACP.
    """
    
    def __init__(self, wrapped_tool: Tool, options: Dict[str, Any] | None = None):
        """Initialize with a standard tool to wrap."""
        super().__init__(options=options or {})
        self.wrapped_tool = wrapped_tool
        self._name = wrapped_tool.name
        self._description = wrapped_tool.description
    
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def input_schema(self):
        return self.wrapped_tool.input_schema
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "acp", self._name],
            creator=self,
        )
    
    async def _run(self, input: Any, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """Run the wrapped tool with ACP compatibility."""
        # Convert to ACP message format if needed
        if hasattr(context, "acp_message") and context.acp_message:
            # Process following ACP standards
            pass
        
        # Fall back to standard execution
        result = await self.wrapped_tool._run(input, options, context)
        
        # Format response according to ACP standards
        if isinstance(result, StringToolOutput):
            return result
        
        # Ensure ACP-compatible output
        return StringToolOutput(str(result))