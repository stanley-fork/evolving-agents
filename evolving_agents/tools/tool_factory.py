# evolving_agents/tools/tool_factory.py

import logging
import traceback
import re
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field

from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.utils.strings import to_safe_word

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService

logger = logging.getLogger(__name__)

class DynamicTool(Tool):
    """A dynamic tool implementation compatible with beeai-framework."""
    
    class DynamicToolInput(BaseModel):
        text: str = Field(description="Input text to process")
    
    def __init__(self, name: str, description: str, code_snippet: str, options: Optional[Dict[str, Any]] = None):
        # Initialize with options
        super().__init__(options=options or {})
        
        # Store tool attributes
        self._name_value = name
        self._description_value = description
        
        # Process the code snippet to ensure it's executable
        self.code_snippet = self._process_code_snippet(code_snippet)
        
        # Initialize emitter
        self.emitter = Emitter.root().child(
            namespace=["tool", "dynamic", to_safe_word(self._name_value)],
            creator=self,
        )
    
    def _process_code_snippet(self, code_snippet: str) -> str:
        """Process the code snippet to ensure it's executable."""
        # Log the original code for debugging
        logger.debug(f"Original code snippet:\n{code_snippet}")
        
        # Remove markdown code blocks
        # First, try to extract Python code between ```python and ``` tags
        python_blocks = re.findall(r"```python(.*?)```", code_snippet, re.DOTALL)
        if python_blocks:
            # Use the first Python code block found
            code_snippet = python_blocks[0]
        else:
            # Remove any generic code blocks
            code_blocks = re.findall(r"```(.*?)```", code_snippet, re.DOTALL)
            if code_blocks:
                # Use the first code block found
                code_snippet = code_blocks[0]
                
        # Clean up the code - remove leading/trailing whitespace
        code_snippet = code_snippet.strip()
        
        # Ensure the code has proper input handling
        if "input" not in code_snippet:
            logger.warning(f"Code snippet doesn't reference input parameter. Adding fallback input handling.")
            code_snippet = "# Input is available as 'input' variable\n" + code_snippet
        
        # Ensure the code sets a 'result' variable
        if "result =" not in code_snippet and not code_snippet.strip().endswith("result = "):
            logger.warning(f"Code snippet doesn't set 'result' variable. Adding fallback result assignment.")
            code_snippet += "\n\n# Ensure result is set\nif 'result' not in locals():\n    result = 'Tool executed but did not set a result'"
        
        # Log the processed code
        logger.debug(f"Processed code snippet:\n{code_snippet}")
        
        return code_snippet
    
    @property
    def name(self) -> str:
        return self._name_value
        
    @property
    def description(self) -> str:
        return self._description_value
    
    @property
    def input_schema(self) -> Type[BaseModel]:
        return self.DynamicToolInput
    
    def _run(self, input: Any, options: Optional[Dict[str, Any]] = None) -> Any:
        """Run the tool with the provided input."""
        try:
            # Create a local scope with the input
            local_scope = {"input": input.text, "result": None}
            
            # Log the code being executed
            logger.debug(f"Executing code snippet for {self.name}:\n{self.code_snippet}")
            
            # Execute the code with the local scope
            exec(self.code_snippet, {"print": print}, local_scope)
            
            # Return the result as a StringToolOutput
            if "result" in local_scope and local_scope["result"] is not None:
                result_value = local_scope["result"]
                if isinstance(result_value, dict) or isinstance(result_value, list):
                    import json
                    return StringToolOutput(json.dumps(result_value, indent=2))
                return StringToolOutput(str(result_value))
            else:
                return StringToolOutput("Tool executed but did not produce a result")
        except Exception as e:
            error_msg = f"Error executing tool {self.name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return StringToolOutput(f"Execution error: {str(e)}")

class ToolFactory:
    """
    Factory for creating and executing tools from library records.
    """
    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService):
        self.library = smart_library
        self.llm = llm_service
        self.active_tools = {}
    
    async def create_tool(
        self, 
        record: Dict[str, Any],
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
        logger.info(f"Creating tool {record['name']} from record {record['id']}")
        
        try:
            # Create the dynamic tool
            tool = DynamicTool(
                name=record["name"],
                description=record["description"],
                code_snippet=record["code_snippet"]
            )
            
            # Store in active tools
            self.active_tools[record["name"]] = {
                "record": record,
                "instance": tool,
                "type": "TOOL"
            }
            
            return tool
        except Exception as e:
            logger.error(f"Error creating tool {record['name']}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Create a fallback tool that returns the error
            fallback_code = f"""
            # Error occurred during tool creation
            
            def fallback_function(input):
                return "Error creating tool: {str(e)}"
                
            result = fallback_function(input)
            """
            
            fallback_tool = DynamicTool(
                name=record["name"],
                description=f"FALLBACK: {record['description']}",
                code_snippet=fallback_code
            )
            
            # Store fallback in active tools
            self.active_tools[record["name"]] = {
                "record": record,
                "instance": fallback_tool,
                "type": "TOOL"
            }
            
            return fallback_tool
    
    async def execute_tool(self, tool_instance: Tool, input_text: str) -> str:
        """
        Execute a tool with input text.
        
        Args:
            tool_instance: The tool instance
            input_text: Input text
            
        Returns:
            Tool output as string
        """
        logger.info(f"Executing tool with input: {input_text[:50]}...")
        
        try:
            # Create input for the tool
            input_obj = tool_instance.input_schema(text=input_text)
            
            # Execute the tool directly
            result = tool_instance._run(input=input_obj, options=None)
            
            # Convert result to string
            if hasattr(result, "get_text_content"):
                return result.get_text_content()
            elif hasattr(result, "result"):
                return result.result
            else:
                return str(result)
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_instance.name}: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Execution error: {str(e)}"