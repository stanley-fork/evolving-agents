from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid

class RecordType(str, Enum):
    """Type of record in the Smart Library."""
    AGENT = "AGENT"
    TOOL = "TOOL"
    FIRMWARE = "FIRMWARE"

class RecordStatus(str, Enum):
    """Status of a record in the Smart Library."""
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"

class LibraryRecord:
    """
    Unified record for the Smart Library that can represent agents, tools, or firmware.
    Includes metrics, metadata, and version history.
    """
    def __init__(
        self,
        name: str,
        record_type: RecordType,
        domain: str,
        description: str,
        code_snippet: str,
        version: str = "1.0.0",
        usage_count: int = 0,
        success_count: int = 0,
        fail_count: int = 0,
        embedding: Optional[List[float]] = None,
        status: RecordStatus = RecordStatus.ACTIVE,
        last_updated: Optional[datetime] = None,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.record_type = record_type
        self.domain = domain
        self.description = description
        self.code_snippet = code_snippet
        self.version = version
        self.usage_count = usage_count
        self.success_count = success_count
        self.fail_count = fail_count
        self.embedding = embedding or []
        self.status = status
        self.last_updated = last_updated or datetime.utcnow()
        self.metadata = metadata or {}
        self.parent_id = parent_id  # For tracking evolutions
        self.tags = tags or []
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate from metrics."""
        if self.usage_count == 0:
            return 0.0
        return float(self.success_count) / float(self.usage_count)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "record_type": self.record_type.value,
            "domain": self.domain,
            "description": self.description,
            "code_snippet": self.code_snippet,
            "version": self.version,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "embedding": self.embedding,
            "status": self.status.value,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "tags": self.tags
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LibraryRecord':
        """Create a record from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            record_type=RecordType(data["record_type"]),
            domain=data["domain"],
            description=data["description"],
            code_snippet=data["code_snippet"],
            version=data["version"],
            usage_count=data["usage_count"],
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            embedding=data["embedding"],
            status=RecordStatus(data["status"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", [])
        )
        
    def increment_usage(self, success: bool = True) -> None:
        """
        Increment usage metrics.
        
        Args:
            success: Whether the usage was successful
        """
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        self.last_updated = datetime.utcnow()