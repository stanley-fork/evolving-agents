# evolving_agents/tools/tool_factory.py

import logging
import importlib.util
import sys
import os
from typing import Dict, Any, Optional

from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import LibraryRecord

logger = logging.getLogger(__name__)

class CustomTool(Tool):
    """A simple tool implementation that wraps dynamic code execution."""
    
    def __init__(self, name: str, description: str, code_snippet: str):
        super().__init__(name=name, description=description)
        self.code_snippet = code_snippet
        self._compiled_code = None
        
        try:
            # Compile but don't execute the code yet
            self._compiled_code = compile(code_snippet, f"<tool_{name}>", "exec")
        except Exception as e:
            logger.error(f"Error compiling tool {name}: {str(e)}")
    
    def run(self, input: Dict[str, Any]) -> Any:
        """Execute the tool with provided input."""
        if not self._compiled_code:
            return {"error": "Tool code could not be compiled"}
        
        # Create a local scope with the input
        local_scope = {"input": input, "result": None}
        
        try:
            # Execute the code with the local scope
            exec(self._compiled_code, {}, local_scope)
            return local_scope.get("result", {"error": "Tool did not produce a result"})
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
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
        
        # For simplicity, we'll create a CustomTool that wraps the code
        # In a real implementation, you would parse and execute the code properly
        
        tool = CustomTool(
            name=record.name,
            description=record.description,
            code_snippet=record.code_snippet
        )
        
        # Store in active tools
        self.active_tools[record.id] = tool
        
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
                return str(result)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            return f"Error: {str(e)}"