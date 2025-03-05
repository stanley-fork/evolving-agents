# evolving_agents/smart_library/smart_library.py

import json
import os
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from evolving_agents.utils.embeddings import embedding_service

logger = logging.getLogger(__name__)

class SmartLibrary:
    """
    Unified library that stores all agents, tools, and firmware as simple dictionary records.
    """
    def __init__(self, storage_path: str = "smart_library.json"):
        self.storage_path = storage_path
        self.records = []
        self._load_library()
    
    def _load_library(self):
        """Load the library from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
                logger.info(f"Loaded {len(self.records)} records from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading library: {str(e)}")
                self.records = []
        else:
            logger.info(f"No existing library found at {self.storage_path}. Creating new library.")
            self.records = []
    
    def _save_library(self):
        """Save the library to storage."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)
        logger.info(f"Saved {len(self.records)} records to {self.storage_path}")
    
    async def save_record(self, record: Dict[str, Any]) -> str:
        """
        Save a record to the library.
        
        Args:
            record: Dictionary containing record data
            
        Returns:
            ID of the saved record
        """
        # Update existing or add new
        idx = next((i for i, r in enumerate(self.records) if r["id"] == record["id"]), -1)
        if idx >= 0:
            self.records[idx] = record
            logger.info(f"Updated record {record['id']} ({record['name']})")
        else:
            self.records.append(record)
            logger.info(f"Added new record {record['id']} ({record['name']})")
        
        self._save_library()
        return record["id"]
    
    async def find_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Find a record by ID."""
        return next((r for r in self.records if r["id"] == record_id), None)
    
    async def find_record_by_name(self, name: str, record_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find a record by name.
        
        Args:
            name: Record name to find
            record_type: Optional record type filter (AGENT, TOOL, FIRMWARE)
            
        Returns:
            The record if found, None otherwise
        """
        if record_type:
            return next((r for r in self.records if r["name"] == name and r["record_type"] == record_type), None)
        return next((r for r in self.records if r["name"] == name), None)
    
    async def find_records_by_domain(self, domain: str, record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find records by domain.
        
        Args:
            domain: Domain to search for
            record_type: Optional record type filter
            
        Returns:
            List of matching records
        """
        if record_type:
            return [r for r in self.records if r.get("domain") == domain and r["record_type"] == record_type]
        return [r for r in self.records if r.get("domain") == domain]
    
    async def semantic_search(
        self, 
        query: str, 
        record_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for records semantically similar to the query using OpenAI embeddings.
        
        Args:
            query: The search query
            record_type: Optional record type filter
            domain: Optional domain filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (record, similarity) tuples sorted by similarity
        """
        # Filter records by type and domain if specified
        filtered_records = self.records
        if record_type:
            filtered_records = [r for r in filtered_records if r["record_type"] == record_type]
        if domain:
            filtered_records = [r for r in filtered_records if r.get("domain") == domain]
            
        # Filter only active records
        active_records = [r for r in filtered_records if r.get("status", "active") == "active"]
        
        if not active_records:
            logger.info(f"No active records found for search: {query}")
            return []
        
        # Get embedding for the query
        query_embedding = await embedding_service.generate_embedding(query)
        
        # For each record, calculate similarity
        results = []
        for record in active_records:
            # Create a combined text representation of the record
            description = record.get("description", "")
            code_snippet = record.get("code_snippet", "")
            name = record.get("name", "")
            
            # Generate embedding for the record
            record_text = f"{name} {description} {code_snippet}"
            record_embedding = await embedding_service.generate_embedding(record_text)
            
            # Compute similarity
            similarity = embedding_service.compute_similarity(query_embedding, record_embedding)
            
            if similarity >= threshold:
                results.append((record, similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def create_record(
        self,
        name: str,
        record_type: str,  # "AGENT", "TOOL", or "FIRMWARE"
        domain: str,
        description: str,
        code_snippet: str,
        version: str = "1.0.0",
        status: str = "active",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new record in the library.
        
        Args:
            name: Record name
            record_type: Record type (AGENT, TOOL, FIRMWARE)
            domain: Domain of the record
            description: Description of the record
            code_snippet: Code snippet or content
            version: Version string
            status: Status of the record
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            The created record
        """
        record_id = str(uuid.uuid4())
        
        record = {
            "id": record_id,
            "name": name,
            "record_type": record_type,
            "domain": domain,
            "description": description,
            "code_snippet": code_snippet,
            "version": version,
            "usage_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "tags": tags or [],
            "metadata": metadata or {}
        }
        
        await self.save_record(record)
        logger.info(f"Created new {record_type} record: {name}")
        return record
    
    async def update_usage_metrics(self, record_id: str, success: bool = True) -> None:
        """
        Update usage metrics for a record.
        
        Args:
            record_id: ID of the record to update
            success: Whether the usage was successful
        """
        record = await self.find_record_by_id(record_id)
        if record:
            record["usage_count"] = record.get("usage_count", 0) + 1
            if success:
                record["success_count"] = record.get("success_count", 0) + 1
            else:
                record["fail_count"] = record.get("fail_count", 0) + 1
            
            record["last_updated"] = datetime.utcnow().isoformat()
            await self.save_record(record)
            logger.info(f"Updated usage metrics for {record['name']} (success={success})")
        else:
            logger.warning(f"Attempted to update metrics for non-existent record: {record_id}")
    
    async def get_firmware(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get the active firmware for a domain.
        
        Args:
            domain: Domain to get firmware for
            
        Returns:
            Firmware record if found, None otherwise
        """
        # Find active firmware for the domain, or general if not found
        firmware = next(
            (r for r in self.records 
             if r["record_type"] == "FIRMWARE" 
             and r.get("domain") == domain 
             and r.get("status") == "active"),
            None
        )
        
        if not firmware:
            # Fall back to general firmware
            firmware = next(
                (r for r in self.records 
                 if r["record_type"] == "FIRMWARE" 
                 and r.get("domain") == "general" 
                 and r.get("status") == "active"),
                None
            )
            
        return firmware
    
    async def evolve_record(
        self,
        parent_id: str,
        new_code_snippet: str,
        description: Optional[str] = None,
        new_version: Optional[str] = None,
        status: str = "active"
    ) -> Dict[str, Any]:
        """
        Create an evolved version of an existing record.
        
        Args:
            parent_id: ID of the parent record
            new_code_snippet: New code snippet for the evolved record
            description: Optional new description
            new_version: Optional version override (otherwise incremented)
            status: Status for the new record
            
        Returns:
            The newly created record
        """
        parent = await self.find_record_by_id(parent_id)
        if not parent:
            raise ValueError(f"Parent record not found: {parent_id}")
            
        # Increment version if not specified
        if new_version is None:
            new_version = self._increment_version(parent["version"])
        
        # Create new record with parent's metadata
        new_record = {
            "id": str(uuid.uuid4()),
            "name": parent["name"],
            "record_type": parent["record_type"],
            "domain": parent["domain"],
            "description": description or parent["description"],
            "code_snippet": new_code_snippet,
            "version": new_version,
            "usage_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "parent_id": parent_id,
            "tags": parent.get("tags", []).copy(),
            "metadata": {
                **(parent.get("metadata", {}).copy()),
                "evolved_at": datetime.utcnow().isoformat(),
                "evolved_from": parent_id,
                "previous_version": parent["version"]
            }
        }
        
        # Save and return
        await self.save_record(new_record)
        logger.info(f"Evolved record {parent['name']} from {parent['version']} to {new_version}")
        
        return new_record
    
    def _increment_version(self, version: str) -> str:
        """
        Increment the version number.
        
        Args:
            version: Current version string (e.g., "1.0.0")
            
        Returns:
            Incremented version string
        """
        parts = version.split(".")
        if len(parts) < 3:
            parts += ["0"] * (3 - len(parts))
            
        # Increment patch version
        try:
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except (ValueError, IndexError):
            # If version format is invalid, just append .1
            return f"{version}.1"