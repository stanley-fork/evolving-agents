# evolving_agents/acp/tool.py

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from abc import abstractmethod

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

logger = logging.getLogger(__name__)

@runtime_checkable
class ACPToolInterface(Protocol):
    """Protocol defining the interface for ACP-compatible tools."""
    
    @abstractmethod
    async def process_acp_message(
        self,
        message: Dict[str, Any],
        context: Optional[RunContext] = None
    ) -> Dict[str, Any]:
        """
        Process an ACP message.
        
        Args:
            message: The ACP message to process
            context: Optional execution context
            
        Returns:
            Response message
        """
        ...

class ACPTool(Tool):
    """
    Base implementation of an ACP-compatible tool.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        options: Optional[Dict[str, Any]] = None
    ):
        """Initialize the ACP tool."""
        super().__init__(options=options or {})
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "acp", self._name],
            creator=self,
        )
    
    async def process_acp_message(
        self,
        message: Dict[str, Any],
        context: Optional[RunContext] = None
    ) -> Dict[str, Any]:
        """
        Process an ACP message.
        
        Args:
            message: The ACP message to process
            context: Optional execution context
            
        Returns:
            Response message
        """
        message_type = message.get("type", "unknown")
        
        if message_type == "text_input":
            # Parse text input as tool input
            content = message.get("content", "")
            input_obj = self.input_schema(text=content)
            
            # Execute the tool
            result = await self._run(input_obj, None, context)
            
            # Format the response
            response = {
                "type": "text_output",
                "content": result.get_text_content() if hasattr(result, "get_text_content") else str(result),
                "message_id": str(uuid.uuid4()),
                "in_response_to": message.get("message_id")
            }
            
        elif message_type == "json_input":
            # Parse JSON input as tool input
            content = message.get("content", {})
            input_obj = self.input_schema(**content)
            
            # Execute the tool
            result = await self._run(input_obj, None, context)
            
            # Format the response
            response = {
                "type": "json_output",
                "content": json.loads(result.get_text_content()) if hasattr(result, "get_text_content") else result,
                "message_id": str(uuid.uuid4()),
                "in_response_to": message.get("message_id")
            }
            
        else:
            # Unknown message type
            response = {
                "type": "error",
                "content": f"Unsupported message type: {message_type}",
                "message_id": str(uuid.uuid4()),
                "in_response_to": message.get("message_id")
            }
        
        return response