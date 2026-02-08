"""
BSV Blockchain Evidence Registry (Bridge Version)
=================================================
Este archivo conecta tu App con el sistema de Blockchain.
Sincroniza los nombres de las funciones para evitar errores de 'AttributeError'.
"""

import hashlib
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
import numpy as np

# --- CLASES DE DATOS ---

@dataclass
class ImageEvidence:
    image_hash: str
    timestamp: str
    scene_id: str
    location: Optional[Dict] = None
    
    def to_dict(self): return asdict(self)

@dataclass
class AnalysisEvidence:
    metrics_hash: str
    timestamp: str
    scene_id: str
    summary_metrics: Dict
    total_vehicles: int
    incident_count: int
    processing_time_sec: float = 0.0
    
    def to_dict(self): return asdict(self)

@dataclass
class BlockchainTransaction:
    transaction_id: str
    timestamp: str
    scene_id: str
    image_hash: str
    metrics_hash: str
    payload: Dict
    chain_stage: str
    previous_transaction_id: Optional[str] = None
    
    def to_dict(self): return asdict(self)

# --- CLASE PRINCIPAL ---

class BSVEvidenceRegistry:
    def __init__(self, scene_id: str, location: Optional[Dict] = None, output_dir: str = "outputs/blockchain_evidence"):
        self.scene_id = scene_id
        self.location = location
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_evidences: List[ImageEvidence] = []
        self.analysis_evidences: List[AnalysisEvidence] = []
        self.blockchain_transactions: List[BlockchainTransaction] = []
        self.etl_chain: List[str] = []
        
        print(f"✅ BSV Registry iniciado para: {scene_id}")

    def _hash(self, data: Union[bytes, str, Dict, np.ndarray]) -> str:
        """Genera hash SHA-256 consistente"""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, (dict, list)):
            data = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def register_image(self, image_array: np.ndarray) -> ImageEvidence:
        """Registra una imagen (frame)"""
        ev = ImageEvidence(
            image_hash=self._hash(image_array), 
            timestamp=datetime.utcnow().isoformat() + "Z", 
            scene_id=self.scene_id, 
            location=self.location
        )
        self.image_evidences.append(ev)
        return ev

    def register_analysis(self, metrics: Dict, total_vehicles: int, incident_count: int, processing_time_sec: float = 0) -> AnalysisEvidence:
        """Registra el análisis final"""
        ev = AnalysisEvidence(
            metrics_hash=self._hash(metrics), 
            timestamp=datetime.utcnow().isoformat() + "Z", 
            scene_id=self.scene_id, 
            summary_metrics=metrics, 
            total_vehicles=total_vehicles, 
            incident_count=incident_count, 
            processing_time_sec=processing_time_sec
        )
        self.analysis_evidences.append(ev)
        return ev

    # ---------------------------------------------------------
    # 🔥 PUNTOS CLAVE DE CORRECCIÓN (Nombres compatibles)
    # ---------------------------------------------------------

    def create_blockchain_transaction(self, image_evidence: ImageEvidence, analysis_evidence: AnalysisEvidence, chain_stage: str = "processed") -> BlockchainTransaction:
        """
        Crea una transacción vinculada. 
        Nota: main.py llama a 'create_blockchain_transaction', no a 'create_transaction'.
        """
        tx_id = uuid.uuid4().hex
        prev_tx = self.etl_chain[-1] if self.etl_chain else None
        
        payload = {
            "metrics": analysis_evidence.summary_metrics,
            "total_vehicles": analysis_evidence.total_vehicles,
            "incident_count": analysis_evidence.incident_count
        }
        
        tx = BlockchainTransaction(
            transaction_id=tx_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            scene_id=self.scene_id,
            image_hash=image_evidence.image_hash,
            metrics_hash=analysis_evidence.metrics_hash,
            payload=payload,
            chain_stage=chain_stage,
            previous_transaction_id=prev_tx
        )
        
        self.blockchain_transactions.append(tx)
        self.etl_chain.append(tx_id)
        return tx

    def export_evidence_log(self) -> str:
        """
        Exporta el JSON final.
        Nota: main.py llama a 'export_evidence_log', no a 'export'.
        """
        full_log = {
            "scene_id": self.scene_id,
            "location": self.location,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            # IMPORTANTE: statistics es requerido por app.py
            "statistics": {
                "total_images_processed": len(self.image_evidences),
                "total_analyses": len(self.analysis_evidences),
                "total_transactions": len(self.blockchain_transactions)
            },
            "etl_chain": self.etl_chain,
            "blockchain_transactions": [tx.to_dict() for tx in self.blockchain_transactions],
            "image_evidences": [ev.to_dict() for ev in self.image_evidences]
        }
        
        filename = f"evidence_log_{self.scene_id}.json"
        path = self.output_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(full_log, f, indent=2, default=str)
            
        return str(path)

    def export_supply_chain_format(self):
        """Método auxiliar opcional para compatibilidad con main.py"""
        pass 