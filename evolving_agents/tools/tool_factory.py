# evolving_agents/tools/tool_factory.py

import logging
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field

from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.utils.strings import to_safe_word

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import LibraryRecord

logger = logging.getLogger(__name__)

class SimpleTool(Tool):
    """A simple tool implementation compatible with beeai-framework."""
    
    class SimpleToolInput(BaseModel):
        text: str = Field(description="Input text to process")
    
    def __init__(self, name: str, description: str, code_snippet: str, options: Optional[Dict[str, Any]] = None):
        # First initialize the parent with only the options parameter
        super().__init__(options=options)
        
        # Then store our tool-specific attributes
        self._name = name
        self._description = description
        self.code_snippet = code_snippet
        
        # Initialize emitter (similar to OpenMeteoTool example)
        self.emitter = Emitter.root().child(
            namespace=["tool", "simple", to_safe_word(self._name)],
            creator=self,
        )
    
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def input_schema(self) -> Type[BaseModel]:
        return self.SimpleToolInput
    
    def _run(self, input: Any, options: Optional[Dict[str, Any]] = None) -> Any:
        """Run the tool with the provided input."""
        try:
            # Create a local scope with the input
            local_scope = {"input": input, "result": None}
            
            # Execute the code with the local scope
            exec(self.code_snippet, {}, local_scope)
            
            # Return the result
            if "result" in local_scope:
                return local_scope.get("result")
            else:
                return {"error": "Tool did not produce a result"}
        except Exception as e:
            logger.error(f"Error executing tool {self._name}: {str(e)}")
            return {"error": f"Execution error: {str(e)}"}

class ToolFactory:
    """
    Factory for creating and executing tools from library records.
    """
    def __init__(self, smart_library: SmartLibrary, llm_service: ChatModel):
        self.library = smart_library
        self.llm = llm_service
        self.active_tools = {}
    
    async def create_tool(
        self, 
        record: LibraryRecord,
        firmware_content: Optional[str] = None
    ) -> Tool:
        """
        Create a tool instance from a library record.
        
        Args:
            record: Tool record from the Smart Library
            firmware_content: Optional firmware content to inject
            
        Returns:
            Instantiated tool
        """
        logger.info(f"Creating tool {record.name} from record {record.id}")
        
        tool = SimpleTool(
            name=record.name,
            description=record.description,
            code_snippet=record.code_snippet
        )
        
        # Store in active tools
        self.active_tools[record.name] = {
            "record": record,
            "instance": tool,
            "type": "TOOL"
        }
        
        return tool
    
    async def execute_tool(self, tool_instance: Tool, input_text: str) -> str:
        """
        Execute a tool with input text.
        
        Args:
            tool_instance: The tool instance
            input_text: Input text (will be converted to dict)
            
        Returns:
            Tool output as string
        """
        logger.info(f"Executing tool with input: {input_text[:50]}...")
        
        # Convert input text to dict - this is simplistic
        try:
            input_dict = {"text": input_text}
            result = tool_instance.run(input_dict)
            
            # Convert result to string
            if isinstance(result, dict):
                import json
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            return f"Error: {str(e)}"