import json
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

from evolving_agents.smart_library.record import LibraryRecord, RecordType, RecordStatus
from evolving_agents.smart_library.semantic_search import semantic_search
from evolving_agents.utils.embeddings import generate_embedding

logger = logging.getLogger(__name__)

class SmartLibrary:
    """
    Central repository for all agents, tools, and firmware.
    Handles storage, retrieval, semantic search, and metrics tracking.
    """
    def __init__(self, storage_path: str = "smart_library.json"):
        self.storage_path = storage_path
        self.records: List[LibraryRecord] = []
        self._load_library()
        
    def _load_library(self) -> None:
        """Load the library from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.records = [LibraryRecord.from_dict(item) for item in data]
                logger.info(f"Loaded {len(self.records)} records from {self.storage_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading library: {e}")
                self.records = []
        else:
            logger.info(f"No existing library found at {self.storage_path}. Creating new library.")
            self.records = []
            
    def _save_library(self) -> None:
        """Save the library to storage."""
        data = [record.to_dict() for record in self.records]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.records)} records to {self.storage_path}")
            
    async def save_record(self, record: LibraryRecord) -> str:
        """
        Save a record to the library.
        
        Args:
            record: The record to save
            
        Returns:
            The ID of the saved record
        """
        # Update existing or add new
        idx = next((i for i, r in enumerate(self.records) if r.id == record.id), -1)
        if idx >= 0:
            self.records[idx] = record
            logger.info(f"Updated record {record.id} ({record.name})")
        else:
            self.records.append(record)
            logger.info(f"Added new record {record.id} ({record.name})")
            
        self._save_library()
        return record.id
        
    async def find_record_by_id(self, record_id: str) -> Optional[LibraryRecord]:
        """Find a record by ID."""
        return next((r for r in self.records if r.id == record_id), None)
    
    async def find_record_by_name(self, name: str, record_type: Optional[RecordType] = None) -> Optional[LibraryRecord]:
        """
        Find a record by name.
        
        Args:
            name: Record name to find
            record_type: Optional filter by record type
            
        Returns:
            The record if found, None otherwise
        """
        if record_type:
            return next((r for r in self.records if r.name == name and r.record_type == record_type), None)
        return next((r for r in self.records if r.name == name), None)
        
    async def find_records_by_domain(self, domain: str, record_type: Optional[RecordType] = None) -> List[LibraryRecord]:
        """
        Find records by domain.
        
        Args:
            domain: Domain to search for
            record_type: Optional filter by record type
            
        Returns:
            List of matching records
        """
        if record_type:
            return [r for r in self.records if r.domain == domain and r.record_type == record_type]
        return [r for r in self.records if r.domain == domain]
        
    async def find_records_by_tags(self, tags: List[str], record_type: Optional[RecordType] = None) -> List[LibraryRecord]:
        """
        Find records by tags.
        
        Args:
            tags: List of tags to search for (any match)
            record_type: Optional filter by record type
            
        Returns:
            List of matching records
        """
        filtered = []
        for r in self.records:
            if record_type and r.record_type != record_type:
                continue
            if any(tag in r.tags for tag in tags):
                filtered.append(r)
        return filtered
        
    async def semantic_search(
        self, 
        query: str, 
        record_type: Optional[RecordType] = None,
        domain: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[LibraryRecord, float]]:
        """
        Search for records semantically similar to the query.
        
        Args:
            query: The search query
            record_type: Optional filter by record type
            domain: Optional filter by domain
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (record, similarity) tuples sorted by similarity
        """
        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        
        # Filter records by type and domain if specified
        filtered_records = self.records
        if record_type:
            filtered_records = [r for r in filtered_records if r.record_type == record_type]
        if domain:
            filtered_records = [r for r in filtered_records if r.domain == domain]
            
        # Filter only active records
        active_records = [r for r in filtered_records if r.status == RecordStatus.ACTIVE]
        
        # Perform semantic search
        search_results = await semantic_search(query_embedding, active_records, limit, threshold)
        
        # Return (record, similarity) tuples
        return [(result.record, result.similarity) for result in search_results]
    
    async def update_usage_metrics(self, record_id: str, success: bool = True) -> None:
        """
        Update usage metrics for a record.
        
        Args:
            record_id: ID of the record to update
            success: Whether the usage was successful
        """
        record = await self.find_record_by_id(record_id)
        if record:
            record.increment_usage(success)
            await self.save_record(record)
            logger.info(f"Updated usage metrics for {record.name} (success={success})")
        else:
            logger.warning(f"Attempted to update metrics for non-existent record: {record_id}")
    
    async def get_firmware(self, domain: str) -> Optional[LibraryRecord]:
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
             if r.record_type == RecordType.FIRMWARE 
             and r.domain == domain 
             and r.status == RecordStatus.ACTIVE),
            None
        )
        
        if not firmware:
            # Fall back to general firmware
            firmware = next(
                (r for r in self.records 
                 if r.record_type == RecordType.FIRMWARE 
                 and r.domain == "general" 
                 and r.status == RecordStatus.ACTIVE),
                None
            )
            
        return firmware
    
    async def list_domains(self) -> List[str]:
        """
        Get a list of all domains in the library.
        
        Returns:
            List of unique domain names
        """
        return list(set(r.domain for r in self.records))
    
    async def evolve_record(
        self,
        parent_id: str,
        new_code_snippet: str,
        description: Optional[str] = None,
        new_version: Optional[str] = None,
        status: RecordStatus = RecordStatus.ACTIVE
    ) -> LibraryRecord:
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
            new_version = self._increment_version(parent.version)
            
        # Generate embedding for the new snippet
        embedding = await generate_embedding(new_code_snippet)
        
        # Create new record with parent's metadata
        new_record = LibraryRecord(
            name=parent.name,
            record_type=parent.record_type,
            domain=parent.domain,
            description=description or parent.description,
            code_snippet=new_code_snippet,
            version=new_version,
            usage_count=0,
            success_count=0,
            fail_count=0,
            embedding=embedding,
            status=status,
            metadata=parent.metadata.copy(),  # Copy metadata
            parent_id=parent_id,
            tags=parent.tags.copy()  # Copy tags
        )
        
        # Add evolution metadata
        new_record.metadata["evolved_at"] = datetime.utcnow().isoformat()
        new_record.metadata["evolved_from"] = parent_id
        new_record.metadata["previous_version"] = parent.version
        
        # Save and return
        await self.save_record(new_record)
        logger.info(f"Evolved record {parent.name} from {parent.version} to {new_version}")
        
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