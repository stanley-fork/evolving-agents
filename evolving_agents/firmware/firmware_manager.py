# evolving_agents/firmware/firmware_manager.py

import yaml
import os
import logging
from typing import Dict, Optional, List

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import LibraryRecord, RecordType, RecordStatus

logger = logging.getLogger(__name__)

class FirmwareManager:
    """
    Manages firmware loading and retrieval from the Smart Library.
    """
    def __init__(self, smart_library: SmartLibrary):
        self.library = smart_library

    async def load_firmware_from_yaml(self, yaml_path: str) -> Dict[str, str]:
        """
        Load firmware definitions from a YAML file into the Smart Library.
        
        Args:
            yaml_path: Path to the YAML file containing firmware definitions
            
        Returns:
            Dictionary mapping domain names to firmware IDs
        """
        logger.info(f"Loading firmware from: {yaml_path}")
        
        if not os.path.exists(yaml_path):
            logger.error(f"Firmware file not found: {yaml_path}")
            return {}
            
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading firmware YAML: {str(e)}")
            return {}
            
        firmware_ids = {}
        
        for fw_def in data.get("firmware", []):
            domain = fw_def.get("domain", "general")
            name = fw_def.get("name", f"{domain}_firmware")
            description = fw_def.get("description", f"Firmware rules for {domain} domain")
            
            # Get content from inline or file
            if "content" in fw_def:
                content = fw_def["content"]
            elif "file" in fw_def and os.path.exists(fw_def["file"]):
                with open(fw_def["file"], 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                logger.warning(f"No content found for firmware {name}")
                continue
                
            # Create firmware record
            fw_record = LibraryRecord(
                name=name,
                record_type=RecordType.FIRMWARE,
                domain=domain,
                description=description,
                code_snippet=content,
                version=fw_def.get("version", "1.0.0"),
                status=RecordStatus.ACTIVE,
                tags=[domain, "firmware"]
            )
            
            # Save to library
            record_id = await self.library.save_record(fw_record)
            firmware_ids[domain] = record_id
            logger.info(f"Loaded firmware for domain {domain}: {name}")
            
        return firmware_ids
        
    async def get_firmware_content(self, domain: str) -> str:
        """
        Get firmware content for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Firmware content as string
        """
        fw_record = await self.library.get_firmware(domain)
        if fw_record:
            return fw_record.code_snippet
        
        # Fall back to general domain if specific domain not found
        if domain != "general":
            logger.warning(f"No firmware found for domain {domain}, falling back to general")
            general_fw = await self.library.get_firmware("general")
            if general_fw:
                return general_fw.code_snippet
                
        logger.warning(f"No firmware found for domain {domain}")
        return ""
    
    async def get_domains(self) -> List[str]:
        """
        Get a list of all domains with firmware.
        
        Returns:
            List of domain names
        """
        firmware_records = [r for r in await self.library.list_records() 
                          if r.record_type == RecordType.FIRMWARE and r.status == RecordStatus.ACTIVE]
        return list(set(r.domain for r in firmware_records))