# evolving_agents/tools/smart_library/search_component_tool.py

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import json

from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

from evolving_agents.smart_library.smart_library import SmartLibrary

class SearchInput(BaseModel):
    """Input schema for the SearchComponentTool."""
    query: str = Field(description="Query string to search for")
    record_type: Optional[str] = Field(
        None, 
        description="Type of record to search for (AGENT or TOOL)"
    )
    domain: Optional[str] = Field(
        None, 
        description="Domain to search within"
    )
    limit: int = Field(
        5, 
        description="Maximum number of results to return"
    )
    threshold: float = Field(
        0.0, 
        description="Minimum similarity threshold (0.0 to 1.0)"
    )
    with_recommendation: bool = Field(
        True,
        description="Include a recommendation on whether to reuse, evolve, or create based on similarity score"
    )

class SearchComponentTool(Tool[SearchInput, None, StringToolOutput]):
    """
    Tool for searching components in the Smart Library by query, similarity, or name.
    Uses semantic search to find the most relevant components and provides recommendations
    based on similarity scores.
    """
    name = "SearchComponentTool"
    description = "Search for agents and tools in the library using natural language queries and get recommendations based on similarity"
    input_schema = SearchInput
    
    def __init__(self, smart_library: SmartLibrary, options: Optional[Dict[str, Any]] = None):
        super().__init__(options=options or {})
        self.library = smart_library
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "library", "search"],
            creator=self,
        )
    
    async def _run(self, input: SearchInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Search the Smart Library for components matching the query.
        
        Args:
            input: The search input parameters
        
        Returns:
            StringToolOutput containing the search results in JSON format with recommendations
        """
        try:
            # Perform semantic search
            search_results = await self.library.semantic_search(
                query=input.query,
                record_type=input.record_type,
                domain=input.domain,
                limit=input.limit,
                threshold=input.threshold
            )
            
            # Format results with recommendations
            formatted_results = []
            for i, (record, similarity) in enumerate(search_results):
                result = {
                    "rank": i + 1,
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["record_type"],
                    "domain": record.get("domain", "general"),
                    "description": record["description"],
                    "similarity_score": similarity,
                    "version": record.get("version", "1.0.0")
                }
                
                # Add recommendation based on similarity score
                if input.with_recommendation:
                    result["recommendation"] = self._get_recommendation(similarity)
                    result["recommendation_reason"] = self._get_recommendation_reason(similarity)
                
                formatted_results.append(result)
            
            # Provide an overall recommendation if no results found
            recommendation = {}
            if len(formatted_results) == 0 and input.with_recommendation:
                recommendation = {
                    "overall_recommendation": "create",
                    "reason": "No similar components found. Consider creating a new component."
                }
            elif len(formatted_results) > 0 and input.with_recommendation:
                # Get the recommendation from the highest ranked result
                top_result = formatted_results[0]
                recommendation = {
                    "overall_recommendation": top_result["recommendation"],
                    "reason": top_result["recommendation_reason"],
                    "based_on": f"{top_result['name']} (similarity: {top_result['similarity_score']:.2f})"
                }
            
            response = {
                "query": input.query,
                "result_count": len(formatted_results),
                "results": formatted_results,
            }
            
            # Add recommendation if requested
            if input.with_recommendation:
                response["recommendation"] = recommendation
            
            return StringToolOutput(json.dumps(response, indent=2))
            
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({
                "error": f"Error searching components: {str(e)}",
                "details": traceback.format_exc()
            }, indent=2))
    
    def _get_recommendation(self, similarity: float) -> str:
        """
        Get a recommendation based on similarity score.
        
        Args:
            similarity: Similarity score (0.0 to 1.0)
            
        Returns:
            Recommendation ("reuse", "evolve", or "create")
        """
        if similarity >= 0.8:
            return "reuse"
        elif similarity >= 0.4:
            return "evolve"
        else:
            return "create"
    
    def _get_recommendation_reason(self, similarity: float) -> str:
        """
        Get a reason for the recommendation.
        
        Args:
            similarity: Similarity score (0.0 to 1.0)
            
        Returns:
            Reason for the recommendation
        """
        if similarity >= 0.8:
            return f"The component is highly similar (score: {similarity:.2f}) to the request. Reuse as-is for efficiency."
        elif similarity >= 0.4:
            return f"The component is moderately similar (score: {similarity:.2f}) to the request. Evolve it to better match the requirements."
        else:
            return f"No sufficiently similar component found (best score: {similarity:.2f}). Create a new component."