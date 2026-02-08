import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.spatial import distance
import json
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN OPTIMIZADA
# =============================================================================

@dataclass
class Config:
    """Configuración centralizada del sistema"""
    
    # Directorios
    SOURCE_IMAGES_DIR: str = "data/frames"
    OUTPUT_DIR: str = "outputs"
    MODEL_NAME: str = "yolov8x.pt"  # Modelo más preciso
    MAX_FRAMES: int = 1500
    
    # CALIBRACIÓN FÍSICA
    ALTURA_VUELO_M: float = 120.0
    FOV_HORIZONTAL_GRADOS: float = 84.0
    
    # ZONA DE INTERÉS (ROI) - Excluye parking
    ZONE_POLYGON = np.array([
        [0, 1080],      # Inferior izquierda
        [0, 360],       # Superior izquierda
        [1920, 360],    # Superior derecha
        [1920, 1080]    # Inferior derecha
    ])
    
    # SECTORES PARA ANÁLISIS DETALLADO
    SECTORES = {
        'entrada_norte': np.array([[700, 360], [1200, 360], [1200, 600], [700, 600]]),
        'entrada_sur': np.array([[700, 800], [1200, 800], [1200, 1080], [700, 1080]]),
        'entrada_este': np.array([[1300, 600], [1920, 600], [1920, 800], [1300, 800]]),
        'entrada_oeste': np.array([[0, 600], [600, 600], [600, 800], [0, 800]]),
        'zona_central': np.array([[700, 600], [1200, 600], [1200, 800], [700, 800]])
    }
    
    # PARÁMETROS DE DETECCIÓN MEJORADOS
    CONF_THRESHOLD: float = 0.20  # Bajado para capturar más motos
    IOU_THRESHOLD: float = 0.45
    IMGSZ: int = 1920  # Resolución completa
    
    # PARÁMETROS ESPECÍFICOS POR CLASE
    CLASS_CONF_THRESHOLDS = {
        1: 0.15,   # Bicis - umbral bajo
        2: 0.25,   # Coches - umbral normal
        3: 0.15,   # Motos - umbral bajo (crítico)
        5: 0.30,   # Buses - umbral alto
        7: 0.30    # Camiones - umbral alto
    }
    
    # FILTROS DE TAMAÑO (píxeles)
    CLASS_SIZE_LIMITS = {
        1: (400, 8000),      # Bicis: pequeñas
        2: (1000, 50000),    # Coches: medianas
        3: (300, 6000),      # Motos: muy pequeñas (ajustado)
        5: (4000, 100000),   # Buses: grandes
        7: (3000, 80000)     # Camiones: grandes
    }
    
    # FILTROS DE ASPECTO (W/H)
    CLASS_ASPECT_RATIOS = {
        1: (0.3, 1.2),   # Bicis: verticales o cuadradas
        2: (0.5, 2.5),   # Coches: variado
        3: (0.3, 2.0),   # Motos: delgadas (más permisivo)
        5: (1.0, 3.5),   # Buses: horizontales
        7: (1.0, 3.0)    # Camiones: horizontales
    }
    
    # TRACKING MEJORADO
    TRACK_THRESH: float = 0.15
    TRACK_BUFFER: int = 150  # Frames de memoria
    MATCH_THRESH: float = 0.90
    
    # PARÁMETROS DE INCIDENTES
    VELOCIDAD_MIN_MOVIMIENTO: float = 0.5  # m/s
    TIEMPO_VEHICULO_DETENIDO: int = 120  # frames (~4s a 30fps)
    UMBRAL_FRENADA_BRUSCA: float = 3.0  # m/s²
    DISTANCIA_CONFLICTO: float = 4.0  # metros
    UMBRAL_DENSIDAD_PELIGROSA: float = 50  # veh/km²
    UMBRAL_DENSIDAD_CRITICA: float = 70  # veh/km²
    
    # ANÁLISIS TEMPORAL
    VENTANA_TEMPORAL_FRAMES: int = 30  # 1 segundo a 30fps
    VENTANA_ANALISIS_FRAMES: int = 300  # 10 segundos
    
    def __post_init__(self):
        """Post-inicialización"""
        self.TARGET_CLASSES = [1, 2, 3, 5, 7]
        self.CLASS_NAMES = {
            1: 'Bicicleta', 
            2: 'Turismo', 
            3: 'Motocicleta', 
            5: 'Bus', 
            7: 'Camión'
        }
        
        # Crear directorios
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, 'incidents'), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, 'visualizations'), exist_ok=True)

CONFIG = Config()

# =============================================================================
# SISTEMA DE CALIBRACIÓN Y MÉTRICAS
# =============================================================================

class TrafficEngineer:
    """Cálculo avanzado de métricas de ingeniería de tráfico"""
    
    def __init__(self, image_width_px: int, image_height_px: int):
        # 1. CÁLCULO TRIGONOMÉTRICO 
        # Convertir FOV a radianes
        fov_rad = np.radians(CONFIG.FOV_HORIZONTAL_GRADOS)
        
        # Calcular el ancho real del terreno visible (Trigonometría básica de cámara)
        # Ancho = 2 * Altura_Vuelo * tan(FOV / 2)
        ancho_terreno_m = 2 * CONFIG.ALTURA_VUELO_M * np.tan(fov_rad / 2)
        
        # Calcular alto proporcional
        alto_terreno_m = ancho_terreno_m * (image_height_px / image_width_px)
        
        # 2. CÁLCULO DEL GSD (Ground Sample Distance)
        self.gsd = ancho_terreno_m / image_width_px
        
        # Información del sistema en consola
        print(f"🔧 CALIBRACIÓN DEL SISTEMA")
        print(f"   - Resolución: {image_width_px}x{image_height_px} px")
        print(f"   - GSD (Ground Sample Distance): {self.gsd:.4f} m/px")
        print(f"   - Área de cobertura: {ancho_terreno_m:.1f} x {alto_terreno_m:.1f} m")
        
        # 3. MATRIZ DE HOMOGRAFÍA (Ajuste de perspectiva)
        # Define 4 puntos en la imagen (esquinas)
        src_points = np.float32([
            [0, 0], 
            [image_width_px, 0], 
            [0, image_height_px], 
            [image_width_px, image_height_px]
        ])
        
        # Define las coordenadas reales correspondientes en metros (Top-Down View)
        dst_points = np.float32([
            [0, 0], 
            [ancho_terreno_m, 0], 
            [0, alto_terreno_m], 
            [ancho_terreno_m, alto_terreno_m]
        ])
        
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 4. UMBRALES DE DENSIDAD (HCM 2010)
        self.los_thresholds = {
            'A': (0, 14),      # Flujo libre
            'B': (14, 22),     # Flujo razonablemente libre
            'C': (22, 32),     # Flujo estable
            'D': (32, 45),     # Flujo aproximándose a inestable
            'E': (45, 67),     # Flujo inestable
            'F': (67, float('inf'))  # Flujo forzado
        }

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Transforma coordenadas de píxel a coordenadas métricas reales usando Homografía"""
        # Esta función usa la matriz para corregir perspectiva si la hubiera
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        return transformed[0][0][0], transformed[0][0][1]

    def get_real_coords(self, px: float, py: float) -> Tuple[float, float]:
        """Transforma píxeles a metros reales usando GSD simple (Alternativa rápida)"""
        real_x = px * self.gsd
        real_y = py * self.gsd
        return real_x, real_y
    
    def pixel_to_meters(self, px_distance: float) -> float:
        """Convierte distancia en píxeles a metros"""
        return px_distance * self.gsd
    
    def area_px_to_m2(self, area_px: float) -> float:
        """Convierte área de píxeles a metros cuadrados"""
        return area_px * (self.gsd ** 2)
    
    def calculate_density(self, vehicle_count: int, area_m2: float) -> float:
        """
        Calcula densidad de tráfico
        Densidad (K) = vehículos / área (veh/km²)
        """
        if area_m2 == 0:
            return 0.0
        density = (vehicle_count / area_m2) * 1_000_000  # veh/km²
        return density
    
    def calculate_level_of_service(self, density: float) -> Tuple[str, str]:
        """
        Determina el nivel de servicio (LOS) según HCM
        Retorna: (letra, descripción)
        """
        for los, (min_d, max_d) in self.los_thresholds.items():
            if min_d <= density < max_d:
                descriptions = {
                    'A': 'Libre',
                    'B': 'Estable',
                    'C': 'Limitado',
                    'D': 'Denso',
                    'E': 'Saturado',
                    'F': 'Colapso'
                }
                return los, descriptions[los]
        return 'F', 'Colapso'
    
    def calculate_occupancy(self, vehicle_count: int, max_capacity: int) -> float:
        """
        Calcula tasa de ocupación
        Ocupación = (vehículos / capacidad_máxima) * 100
        """
        if max_capacity == 0:
            return 0.0
        return (vehicle_count / max_capacity) * 100
    
    def estimate_speed_from_trajectory(self, positions: List[np.ndarray], 
                                       timestamps: List[float]) -> Optional[float]:
        """
        Estima velocidad promedio a partir de trayectoria
        Velocidad = distancia_total / tiempo_total (m/s)
        """
        if len(positions) < 2:
            return None
        
        total_distance_m = 0
        for i in range(1, len(positions)):
            dist_px = np.linalg.norm(positions[i] - positions[i-1])
            total_distance_m += self.pixel_to_meters(dist_px)
        
        total_time = timestamps[-1] - timestamps[0]
        if total_time == 0:
            return None
        
        return total_distance_m / total_time

# =============================================================================
# SISTEMA DE TRACKING AVANZADO
# =============================================================================

class VehicleTracker:
    """Seguimiento temporal de vehículos con análisis cinemático"""
    
    def __init__(self, gsd: float):
        self.gsd = gsd
        self.history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=CONFIG.VENTANA_ANALISIS_FRAMES)
        )
        self.stopped_counters: Dict[int, int] = defaultdict(int)
        self.first_detection: Dict[int, int] = {}
        self.last_detection: Dict[int, int] = {}
        self.vehicle_classes: Dict[int, int] = {}
    
    def update(self, tracker_id: int, position: np.ndarray, 
               frame_idx: int, class_id: int):
        """Actualiza historial de vehículo"""
        self.history[tracker_id].append({
            'frame': frame_idx,
            'position': position,
            'timestamp': frame_idx / 30.0,  # Asumiendo 30 fps
            'class_id': class_id
        })
        
        if tracker_id not in self.first_detection:
            self.first_detection[tracker_id] = frame_idx
        self.last_detection[tracker_id] = frame_idx
        self.vehicle_classes[tracker_id] = class_id
    
    def get_velocity(self, tracker_id: int, window_frames: int = 10) -> float:
        """
        Calcula velocidad instantánea
        Usa ventana de frames para suavizar
        """
        if tracker_id not in self.history:
            return 0.0
        
        hist = list(self.history[tracker_id])
        if len(hist) < window_frames:
            window_frames = len(hist)
        
        if window_frames < 2:
            return 0.0
        
        p_start = hist[-window_frames]['position']
        p_end = hist[-1]['position']
        t_start = hist[-window_frames]['timestamp']
        t_end = hist[-1]['timestamp']
        
        dt = t_end - t_start
        if dt == 0:
            return 0.0
        
        dist_px = np.linalg.norm(p_end - p_start)
        dist_m = dist_px * self.gsd
        
        return dist_m / dt  # m/s
    
    def get_acceleration(self, tracker_id: int) -> float:
        """Calcula aceleración (m/s²)"""
        if tracker_id not in self.history:
            return 0.0
        
        hist = list(self.history[tracker_id])
        if len(hist) < 20:
            return 0.0
        
        # Velocidad hace 15 frames
        window = 5
        p1 = hist[-20]['position']
        p2 = hist[-15]['position']
        t1 = hist[-20]['timestamp']
        t2 = hist[-15]['timestamp']
        dt1 = t2 - t1
        
        if dt1 == 0:
            return 0.0
        
        dist1_m = np.linalg.norm(p2 - p1) * self.gsd
        v1 = dist1_m / dt1
        
        # Velocidad actual
        p3 = hist[-window]['position']
        p4 = hist[-1]['position']
        t3 = hist[-window]['timestamp']
        t4 = hist[-1]['timestamp']
        dt2 = t4 - t3
        
        if dt2 == 0:
            return 0.0
        
        dist2_m = np.linalg.norm(p4 - p3) * self.gsd
        v2 = dist2_m / dt2
        
        # Aceleración
        dt_total = t4 - t1
        if dt_total == 0:
            return 0.0
        
        return (v2 - v1) / dt_total
    
    def get_trajectory_length(self, tracker_id: int) -> float:
        """Calcula longitud total de trayectoria en metros"""
        if tracker_id not in self.history:
            return 0.0
        
        hist = list(self.history[tracker_id])
        if len(hist) < 2:
            return 0.0
        
        total_dist_m = 0
        for i in range(1, len(hist)):
            dist_px = np.linalg.norm(hist[i]['position'] - hist[i-1]['position'])
            total_dist_m += dist_px * self.gsd
        
        return total_dist_m
    
    def get_dwell_time(self, tracker_id: int) -> float:
        """Tiempo de permanencia en la escena (segundos)"""
        if tracker_id not in self.first_detection:
            return 0.0
        
        frames = self.last_detection[tracker_id] - self.first_detection[tracker_id]
        return frames / 30.0  # Asumiendo 30 fps

# =============================================================================
# DETECTOR DE INCIDENTES
# =============================================================================

class IncidentDetector:
    """Sistema avanzado de detección de incidentes de seguridad vial"""
    
    def __init__(self, gsd: float):
        self.gsd = gsd
        self.incidents: List[Dict] = []
        self.incident_counters = defaultdict(int)
        
    def detect_stopped_vehicle(self, tracker_id: int, velocity: float, 
                               stopped_counter: int, position: np.ndarray, 
                               frame_idx: int, class_id: int) -> bool:
        """Detecta vehículos anormalmente detenidos"""
        if velocity < CONFIG.VELOCIDAD_MIN_MOVIMIENTO:
            if stopped_counter >= CONFIG.TIEMPO_VEHICULO_DETENIDO:
                # Solo registrar una vez por vehículo
                incident_key = f"stopped_{tracker_id}"
                if self.incident_counters[incident_key] == 0:
                    incident = {
                        'type': 'VEHICULO_DETENIDO',
                        'severity': 'ALTA',
                        'tracker_id': int(tracker_id),
                        'frame': int(frame_idx),
                        'position': position.tolist(),
                        'duration_seconds': float(stopped_counter / 30.0),
                        'vehicle_class': CONFIG.CLASS_NAMES.get(class_id, 'Desconocido'),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.incidents.append(incident)
                    self.incident_counters[incident_key] = 1
                    return True
        else:
            # Reset si el vehículo vuelve a moverse
            incident_key = f"stopped_{tracker_id}"
            self.incident_counters[incident_key] = 0
        
        return False
    
    def detect_harsh_braking(self, tracker_id: int, acceleration: float, 
                            position: np.ndarray, frame_idx: int, 
                            class_id: int, velocity: float) -> bool:
        """Detecta frenadas bruscas"""
        if acceleration < -CONFIG.UMBRAL_FRENADA_BRUSCA and velocity > 2.0:
            incident_key = f"braking_{tracker_id}_{frame_idx//30}"  # Una por segundo
            if self.incident_counters[incident_key] == 0:
                incident = {
                    'type': 'FRENADA_BRUSCA',
                    'severity': 'MEDIA',
                    'tracker_id': int(tracker_id),
                    'frame': int(frame_idx),
                    'position': position.tolist(),
                    'acceleration_m_s2': float(acceleration),
                    'velocity_m_s': float(velocity),
                    'vehicle_class': CONFIG.CLASS_NAMES.get(class_id, 'Desconocido'),
                    'timestamp': datetime.now().isoformat()
                }
                self.incidents.append(incident)
                self.incident_counters[incident_key] = 1
                return True
        
        return False
    
    def detect_conflicts(self, detections: sv.Detections, frame_idx: int) -> List[Dict]:
        """Detecta conflictos espaciales (vehículos peligrosamente cercanos)"""
        conflicts = []
        
        if len(detections) < 2:
            return conflicts
        
        # Calcular centros
        centers = []
        for xyxy in detections.xyxy:
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            centers.append([cx, cy])
        
        centers = np.array(centers)
        
        # Matriz de distancias
        dist_matrix = distance.cdist(centers, centers, 'euclidean')
        
        # Detectar pares cercanos
        for i in range(len(dist_matrix)):
            for j in range(i + 1, len(dist_matrix)):
                dist_px = dist_matrix[i][j]
                dist_m = dist_px * self.gsd
                
                if dist_m < CONFIG.DISTANCIA_CONFLICTO:
                    conflict_key = f"conflict_{min(i,j)}_{max(i,j)}_{frame_idx//15}"
                    if self.incident_counters[conflict_key] == 0:
                        incident = {
                            'type': 'CONFLICTO_ESPACIAL',
                            'severity': 'ALTA',
                            'frame': int(frame_idx),
                            'distance_m': float(dist_m),
                            'vehicle_indices': [int(i), int(j)],
                            'positions': [centers[i].tolist(), centers[j].tolist()],
                            'timestamp': datetime.now().isoformat()
                        }
                        conflicts.append(incident)
                        self.incidents.append(incident)
                        self.incident_counters[conflict_key] = 1
        
        return conflicts
    
    def detect_dangerous_density(self, density: float, frame_idx: int, 
                                 vehicle_count: int) -> Optional[Dict]:
        """Detecta niveles de densidad peligrosos"""
        if density > CONFIG.UMBRAL_DENSIDAD_CRITICA:
            severity = 'CRITICA'
        elif density > CONFIG.UMBRAL_DENSIDAD_PELIGROSA:
            severity = 'ALTA'
        else:
            return None
        
        # Solo registrar cada 60 frames (2 segundos)
        incident_key = f"density_{frame_idx//60}"
        if self.incident_counters[incident_key] == 0:
            incident = {
                'type': 'DENSIDAD_PELIGROSA',
                'severity': severity,
                'frame': int(frame_idx),
                'density_veh_km2': float(density),
                'vehicle_count': int(vehicle_count),
                'timestamp': datetime.now().isoformat()
            }
            self.incidents.append(incident)
            self.incident_counters[incident_key] = 1
            return incident
        
        return None
    
    def get_incident_summary(self) -> Dict:
        """Genera resumen de incidentes"""
        summary = {
            'total': len(self.incidents),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        for incident in self.incidents:
            summary['by_type'][incident['type']] += 1
            summary['by_severity'][incident['severity']] += 1
        
        return {
            'total': summary['total'],
            'by_type': dict(summary['by_type']),
            'by_severity': dict(summary['by_severity'])
        }

# =============================================================================
# SISTEMA DE VALIDACIÓN
# =============================================================================

class ValidationMetrics:
    """Métricas de precisión y validación del modelo"""
    
    def __init__(self):
        self.detections_per_class = defaultdict(int)
        self.confidence_per_class = defaultdict(list)
        self.size_per_class = defaultdict(list)
        self.aspect_ratio_per_class = defaultdict(list)
        self.false_positive_candidates = []
        
    def update(self, detections: sv.Detections):
        """Registra detecciones para análisis"""
        for xyxy, class_id, conf in zip(detections.xyxy, 
                                        detections.class_id, 
                                        detections.confidence):
            self.detections_per_class[class_id] += 1
            self.confidence_per_class[class_id].append(float(conf))
            
            # Calcular tamaño
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            self.size_per_class[class_id].append(float(area))
            
            # Calcular aspect ratio
            if h > 0:
                ar = w / h
                self.aspect_ratio_per_class[class_id].append(float(ar))
            
            # Detectar posibles falsos positivos
            if conf < 0.30:
                self.false_positive_candidates.append({
                    'class_id': int(class_id),
                    'confidence': float(conf),
                    'bbox': xyxy.tolist()
                })
    
    def generate_report(self) -> Dict:
        """Genera reporte completo de validación"""
        report = {
            'summary': {
                'total_detections': sum(self.detections_per_class.values()),
                'unique_classes_detected': len(self.detections_per_class),
                'potential_false_positives': len(self.false_positive_candidates)
            },
            'by_class': {},
            'class_distribution': {}
        }
        
        total = sum(self.detections_per_class.values())
        
        for class_id, count in self.detections_per_class.items():
            class_name = CONFIG.CLASS_NAMES.get(class_id, f'Class_{class_id}')
            confs = self.confidence_per_class[class_id]
            sizes = self.size_per_class[class_id]
            ars = self.aspect_ratio_per_class[class_id]
            
            report['by_class'][class_name] = {
                'count': int(count),
                'percentage': float(count / total * 100) if total > 0 else 0,
                'confidence': {
                    'mean': float(np.mean(confs)),
                    'std': float(np.std(confs)),
                    'min': float(np.min(confs)),
                    'max': float(np.max(confs)),
                    'median': float(np.median(confs))
                },
                'size_px2': {
                    'mean': float(np.mean(sizes)),
                    'std': float(np.std(sizes)),
                    'min': float(np.min(sizes)),
                    'max': float(np.max(sizes))
                },
                'aspect_ratio': {
                    'mean': float(np.mean(ars)) if ars else 0,
                    'std': float(np.std(ars)) if ars else 0
                }
            }
            
            report['class_distribution'][class_name] = int(count)
        
        return report

# =============================================================================
# FILTROS AVANZADOS DE DETECCIÓN
# =============================================================================

class DetectionFilter:
    """Sistema de filtrado multi-criterio para mejorar precisión"""
    
    @staticmethod
    def filter_by_confidence(detections: sv.Detections) -> sv.Detections:
        """Filtra por umbrales de confianza específicos por clase"""
        if len(detections) == 0:
            return detections
        
        valid_indices = []
        for idx, (class_id, conf) in enumerate(zip(detections.class_id, 
                                                   detections.confidence)):
            threshold = CONFIG.CLASS_CONF_THRESHOLDS.get(class_id, CONFIG.CONF_THRESHOLD)
            if conf >= threshold:
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            return sv.Detections.empty()
        
        return detections[valid_indices]
    
    @staticmethod
    def filter_by_size(detections: sv.Detections) -> sv.Detections:
        """Filtra por tamaño razonable según clase"""
        if len(detections) == 0:
            return detections
        
        valid_indices = []
        for idx, (xyxy, class_id) in enumerate(zip(detections.xyxy, 
                                                   detections.class_id)):
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            
            if class_id in CONFIG.CLASS_SIZE_LIMITS:
                min_size, max_size = CONFIG.CLASS_SIZE_LIMITS[class_id]
                if min_size <= area <= max_size:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            return sv.Detections.empty()
        
        return detections[valid_indices]
    
    @staticmethod
    def filter_by_aspect_ratio(detections: sv.Detections) -> sv.Detections:
        """Filtra por relación de aspecto esperada"""
        if len(detections) == 0:
            return detections
        
        valid_indices = []
        for idx, (xyxy, class_id) in enumerate(zip(detections.xyxy, 
                                                   detections.class_id)):
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            
            if h == 0:
                continue
            
            ar = w / h
            
            if class_id in CONFIG.CLASS_ASPECT_RATIOS:
                min_ar, max_ar = CONFIG.CLASS_ASPECT_RATIOS[class_id]
                if min_ar <= ar <= max_ar:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            return sv.Detections.empty()
        
        return detections[valid_indices]
    
    @staticmethod
    def apply_all_filters(detections: sv.Detections) -> sv.Detections:
        """Aplica todos los filtros en secuencia"""
        detections = DetectionFilter.filter_by_confidence(detections)
        detections = DetectionFilter.filter_by_size(detections)
        detections = DetectionFilter.filter_by_aspect_ratio(detections)
        return detections

# =============================================================================
# GENERADOR DE DASHBOARD AVANZADO
# =============================================================================

class DashboardGenerator:
    """Genera dashboard interactivo completo"""
    
    @staticmethod
    def create_advanced_dashboard(df: pd.DataFrame, 
                                 incidents: List[Dict],
                                 validation_report: Dict,
                                 sector_data: Dict,
                                 output_path: str):
        """Crea dashboard con todas las visualizaciones"""
        
        print("📊 Generando Dashboard Avanzado...")
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                # Fila 1
                "📈 Evolución de Densidad Temporal",
                "🚗 Distribución por Tipo de Vehículo", 
                "🚦 Distribución de Nivel de Servicio",
                # Fila 2
                "⚠️ Incidentes por Tipo",
                "📍 Flujo Vehicular por Sector",
                "⚡ Distribución de Velocidades",
                # Fila 3
                "✅ Confianza del Modelo por Clase",
                "📊 Evolución Temporal (Media por Intervalo)",
                "🎯 Precisión de Detecciones",
                # Fila 4
                "🔥 Mapa de Calor: Densidad",
                "📉 Tendencia de Ocupación",
                "⚠️ Indicador de Riesgo Global"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.10
        )
        
        # ==================== FILA 1 ====================
        
        # 1.1 Densidad temporal suavizada
        df['Densidad_Smooth'] = df['Densidad'].rolling(window=30, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df['Frame'],
                y=df['Densidad_Smooth'],
                mode='lines',
                name='Densidad',
                line=dict(color='#00d2d3', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 210, 211, 0.2)'
            ),
            row=1, col=1
        )
        
        # Líneas de referencia LOS
        # --- Umbral E/F (row 1, col 1) ---
        fig.add_shape(
            type="line",
            xref="x domain",
            yref="y",
            x0=0, x1=1,
            y0=45, y1=45,
            line=dict(color="orange", width=2, dash="dash"),
            row=1, col=1
        )

        fig.add_annotation(
            text="Umbral E/F",
            xref="x domain",
            yref="y",
            x=0.02, y=45,
            showarrow=False,
            yshift=10,
            font=dict(color="orange", size=10),
            row=1, col=1
        )

        # --- Colapso (row 1, col 1) ---
        fig.add_shape(
            type="line",
            xref="x domain",
            yref="y",
            x0=0, x1=1,
            y0=67, y1=67,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )

        fig.add_annotation(
            text="Colapso",
            xref="x domain",
            yref="y",
            x=0.02, y=67,
            showarrow=False,
            yshift=10,
            font=dict(color="red", size=10),
            row=1, col=1
        )
        
        # 1.2 Distribución vehicular (pie chart)
        vehicle_counts = {
            'Turismos': int(df['Turismos'].sum()),
            'Motocicletas': int(df['Motos'].sum()),
            'Bicicletas': int(df['Bicis'].sum()),
            'Buses': int(df['Buses'].sum()),
            'Camiones': int(df['Camiones'].sum())
        }
        
        # Filtrar ceros
        vehicle_counts = {k: v for k, v in vehicle_counts.items() if v > 0}
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        fig.add_trace(
            go.Pie(
                labels=list(vehicle_counts.keys()),
                values=list(vehicle_counts.values()),
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 1.3 Nivel de Servicio
        los_counts = df['LOS'].value_counts()
        los_order = ['A', 'B', 'C', 'D', 'E', 'F']
        los_counts = los_counts.reindex(los_order, fill_value=0)
        
        los_colors = {
            'A': '#2ecc71',
            'B': '#27ae60',
            'C': '#f39c12',
            'D': '#e67e22',
            'E': '#e74c3c',
            'F': '#c0392b'
        }
        
        fig.add_trace(
            go.Bar(
                x=los_counts.index,
                y=los_counts.values,
                marker=dict(color=[los_colors.get(x, '#95a5a6') for x in los_counts.index]),
                name='LOS Count',
                text=los_counts.values,
                textposition='outside'
            ),
            row=1, col=3
        )
        
        # ==================== FILA 2 ====================
        
        # 2.1 Incidentes por tipo
        if incidents:
            incident_types = defaultdict(int)
            for inc in incidents:
                incident_types[inc['type']] += 1
            
            incident_colors = {
                'VEHICULO_DETENIDO': '#e74c3c',
                'FRENADA_BRUSCA': '#f39c12',
                'CONFLICTO_ESPACIAL': '#e67e22',
                'DENSIDAD_PELIGROSA': '#c0392b'
            }
            
            types = list(incident_types.keys())
            counts_inc = list(incident_types.values())
            
            fig.add_trace(
                go.Bar(
                    x=types,
                    y=counts_inc,
                    marker=dict(color=[incident_colors.get(t, '#95a5a6') for t in types]),
                    name='Incidentes',
                    text=counts_inc,
                    textposition='outside'
                ),
                row=2, col=1
            )
        
        # 2.2 Flujo por sector
        for sector_name, data in sector_data.items():
            if data:
                frames_s = [d['frame'] for d in data]
                counts_s = [d['count'] for d in data]
                
                fig.add_trace(
                    go.Scatter(
                        x=frames_s,
                        y=counts_s,
                        mode='lines',
                        name=sector_name.replace('_', ' ').title(),
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
        
        # 2.3 Velocidades
        if 'Velocidad_Media' in df.columns and df['Velocidad_Media'].sum() > 0:
            velocidades_kmh = df['Velocidad_Media'] * 3.6  # Convertir a km/h
            velocidades_kmh = velocidades_kmh[velocidades_kmh > 0]
            
            if len(velocidades_kmh) > 0:
                fig.add_trace(
                    go.Box(
                        y=velocidades_kmh,
                        name='Velocidad (km/h)',
                        marker=dict(color='#3498db'),
                        boxmean='sd'
                    ),
                    row=2, col=3
                )
        
        # ==================== FILA 3 ====================
        
        # 3.1 Confianza del modelo
        if validation_report.get('by_class'):
            classes_val = list(validation_report['by_class'].keys())
            confs_val = [validation_report['by_class'][c]['confidence']['mean'] 
                        for c in classes_val]
            
            fig.add_trace(
                go.Bar(
                    x=classes_val,
                    y=confs_val,
                    marker=dict(color='#48dbfb'),
                    name='Confianza Media',
                    text=[f"{c:.2%}" for c in confs_val],
                    textposition='outside'
                ),
                row=3, col=1
            )
            
            # Línea de referencia
            fig.add_shape(
                type="line",
                xref="x domain", yref="y", # Usar el dominio del subplot específico
                x0=0, x1=1,                # De principio a fin del eje X del subplot
                y0=0.5, y1=0.5,            # Altura 0.5
                line=dict(color="orange", width=2, dash="dash"),
                row=3, col=1
            )
            # Añadimos la etiqueta manualmente
            fig.add_annotation(
                text="Umbral Mínimo",
                xref="x domain", yref="y",
                x=0.05, y=0.5,
                showarrow=False,
                yshift=10,
                font=dict(color="orange", size=10),
                row=3, col=1
            )
                         
        
        # 3.2 Evolución temporal
        df['Intervalo'] = (df['Frame'] // 300).astype(int)
        temporal = df.groupby('Intervalo').agg({
            'Total': 'mean',
            'Densidad': 'mean'
        })
        
        fig.add_trace(
            go.Scatter(
                x=temporal.index,
                y=temporal['Total'],
                mode='lines+markers',
                name='Media por Intervalo',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=8)
            ),
            row=3, col=2
        )
        
        # 3.3 Precisión de detecciones
        if validation_report.get('by_class'):
            classes_prec = list(validation_report['by_class'].keys())
            counts_prec = [validation_report['by_class'][c]['count'] 
                          for c in classes_prec]
            
            fig.add_trace(
                go.Bar(
                    x=classes_prec,
                    y=counts_prec,
                    marker=dict(color='#2ecc71'),
                    name='Detecciones Totales',
                    text=counts_prec,
                    textposition='outside'
                ),
                row=3, col=3
            )
        
        # ==================== FILA 4 ====================
        
        # 4.1 Mapa de calor de densidad
        # Crear matriz de densidad por intervalos
        df['Intervalo_10s'] = (df['Frame'] // 300).astype(int)
        heatmap_data = df.pivot_table(
            values='Densidad',
            index='Intervalo_10s',
            aggfunc='mean'
        )
        
        # Crear matriz 2D para heatmap (simular sectores)
        heatmap_matrix = []
        for i in range(min(10, len(heatmap_data))):
            value = float(heatmap_data.iloc[i].values[0])
            row = [value] * 5
            heatmap_matrix.append(row)

        
        if heatmap_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_matrix,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Densidad")
                ),
                row=4, col=1
            )
        
        # 4.2 Tendencia de ocupación
        if 'Total' in df.columns:
            # Calcular ocupación relativa
            max_capacity = df['Total'].quantile(0.95)  # Capacidad estimada
            df['Ocupacion'] = (df['Total'] / max_capacity * 100).clip(0, 100)
            
            fig.add_trace(
                go.Scatter(
                    x=df['Frame'],
                    y=df['Ocupacion'],
                    mode='lines',
                    name='Ocupación %',
                    line=dict(color='#e74c3c', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.2)'
                ),
                row=4, col=2
            )
            
            # Líneas de referencia
            fig.add_shape(
                type="line",
                xref="x domain",
                yref="y",
                x0=0, x1=1,
                y0=80, y1=80,
                line=dict(color="orange", width=2, dash="dash"),
                row=4, col=2
            )

            fig.add_annotation(
                text="80% Capacidad",
                xref="x domain",
                yref="y",
                x=0.02, y=80,
                showarrow=False,
                yshift=10,
                font=dict(color="orange", size=10),
                row=4, col=2
            )
        
        # 4.3 Indicador de riesgo global
        total_incidents = len(incidents)
        
        if total_incidents < 5:
            risk_level = "BAJO"
            risk_color = "#2ecc71"
        elif total_incidents < 15:
            risk_level = "MEDIO"
            risk_color = "#f39c12"
        elif total_incidents < 30:
            risk_level = "ALTO"
            risk_color = "#e67e22"
        else:
            risk_level = "CRÍTICO"
            risk_color = "#e74c3c"
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta+gauge",
                value=total_incidents,
                delta={'reference': 10},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 15], 'color': "lightyellow"},
                        {'range': [15, 30], 'color': "orange"},
                        {'range': [30, 50], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 40
                    }
                },
                title={
                    'text': f"<b>Incidentes Totales</b><br><span style='font-size:0.9em'>Nivel de Riesgo: {risk_level}</span>",
                    'font': {'size': 16}
                },
            ),
            row=4, col=3
        )
        
        # ==================== LAYOUT ====================
        
        fig.update_layout(
            template="plotly_dark",
            title={
                'text': "<b>AeroTrace v5.0 - Dashboard de Análisis Integral de Tráfico UAV</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#ecf0f1'}
            },
            showlegend=True,
            height=1800,
            font=dict(family="Arial, sans-serif", size=11),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#0e0e0e'
        )
        
        # Actualizar ejes
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#2c3e50')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#2c3e50')
        
        # Guardar
        fig.write_html(output_path)
        print(f"✅ Dashboard guardado en: {output_path}")

# =============================================================================
# SISTEMA PRINCIPAL MEJORADO
# =============================================================================

class AeroTraceSystem:
    """Sistema principal de análisis de tráfico con UAV - Versión mejorada"""
    
    def __init__(self):
        print("="*80)
        print("🚀 INICIANDO AEROTRACE v5.0 - SISTEMA AVANZADO DE ANÁLISIS DE TRÁFICO")
        print("="*80)
        
        # Cargar modelo
        print("\n📦 Cargando modelo YOLOv8...")
        self.model = YOLO(CONFIG.MODEL_NAME)
        
        # Mover a GPU si está disponible
        try:
            self.model.to('cuda')
            print("✅ Modelo cargado en GPU")
        except:
            print("⚠️  GPU no disponible, usando CPU")
        
        # Tracker mejorado
        print("🎯 Configurando ByteTrack...")
        self.tracker = sv.ByteTrack(
            track_thresh=CONFIG.TRACK_THRESH,
            track_buffer=CONFIG.TRACK_BUFFER,
            match_thresh=CONFIG.MATCH_THRESH,
            frame_rate=30
        )
        
        # Anotadores
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.ColorPalette.DEFAULT
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=5
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=50,
            position=sv.Position.CENTER
        )
        
        # Mejora de imagen con CLAHE
        print("🎨 Configurando procesamiento de imagen...")
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
        # Almacenamiento de datos
        self.metrics_log = []
        self.sector_metrics = defaultdict(list)
        
        print("✅ Sistema inicializado correctamente\n")

        def callback(image: np.ndarray) -> sv.Detections:
            results = self.model(image, imgsz=640, verbose=False)[0]
            return sv.Detections.from_ultralytics(results)

        self.slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(640, 640),
            overlap_ratio_wh=(0.2, 0.2) # 20% de solape para no cortar coches a la mitad
        )
    
    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Mejora contraste de imagen con CLAHE"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def analyze_sectors(self, detections: sv.Detections, frame_idx: int):
        """Analiza flujo vehicular por sectores"""
        for sector_name, sector_poly in CONFIG.SECTORES.items():
            zone = sv.PolygonZone(
                polygon=sector_poly,
                frame_resolution_wh=(1920, 1080)
            )
            mask = zone.trigger(detections=detections)
            count = int(np.sum(mask))
            
            self.sector_metrics[sector_name].append({
                'frame': frame_idx,
                'count': count
            })
    
    def run(self):
        """Pipeline principal de procesamiento"""
        
        # ==================== CARGA DE DATOS ====================
        
        search_path = os.path.join(CONFIG.SOURCE_IMAGES_DIR, "*.jpg")
        frames = sorted(glob.glob(search_path))
        
        if not frames:
            # Intentar con PNG
            search_path = os.path.join(CONFIG.SOURCE_IMAGES_DIR, "*.png")
            frames = sorted(glob.glob(search_path))
        
        if not frames:
            print(f"❌ ERROR: No se encontraron imágenes en {CONFIG.SOURCE_IMAGES_DIR}")
            print("   Formatos soportados: .jpg, .png")
            return
        
        if CONFIG.MAX_FRAMES and len(frames) > CONFIG.MAX_FRAMES:
            print(f"⚠️  Limitando procesamiento a {CONFIG.MAX_FRAMES} frames")
            frames = frames[:CONFIG.MAX_FRAMES]
        
        print(f"📁 Se procesarán {len(frames)} frames")
        
        # ==================== INICIALIZACIÓN ====================
        
        # Leer primer frame para obtener dimensiones
        first_frame = cv2.imread(frames[0])
        if first_frame is None:
            print(f"❌ ERROR: No se pudo leer el primer frame: {frames[0]}")
            return
        
        h, w, _ = first_frame.shape
        print(f"📐 Resolución de video: {w}x{h} px")
        
        # Inicializar sistemas
        engineer = TrafficEngineer(w, h)
        vehicle_tracker = VehicleTracker(engineer.gsd)
        incident_detector = IncidentDetector(engineer.gsd)
        validation_metrics = ValidationMetrics()
        detection_filter = DetectionFilter()
        
        # Configurar zona de interés
        zone = sv.PolygonZone(
            polygon=CONFIG.ZONE_POLYGON,
            frame_resolution_wh=(w, h)
        )
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color.WHITE,
            thickness=2,
            text_thickness=2,
            text_scale=0.5
        )
        
        # Calcular área de la zona en metros cuadrados
        poly_area_px = cv2.contourArea(CONFIG.ZONE_POLYGON.astype(np.int32))
        road_area_m2 = engineer.area_px_to_m2(poly_area_px)
        
        print(f"📏 Área de análisis: {road_area_m2:.1f} m²")
        print(f"   Equivalente a: {road_area_m2/10000:.4f} hectáreas")
        
        # Configurar video de salida
        video_path = os.path.join(CONFIG.OUTPUT_DIR, "video_final_v5_improved.mp4")
        video_info = sv.VideoInfo(width=w, height=h, fps=30)
        
        print(f"\n🎬 Generando video de salida: {video_path}")
        
        # ==================== PROCESAMIENTO PRINCIPAL ====================
        
        print("\n" + "="*80)
        print("🔄 INICIANDO PROCESAMIENTO DE FRAMES")
        print("="*80 + "\n")
        
        with sv.VideoSink(video_path, video_info) as sink:
            for frame_idx, path in enumerate(tqdm(frames, desc="Procesando frames")):
                
                # Leer frame
                original_frame = cv2.imread(path)
                if original_frame is None:
                    print(f"⚠️  Warning: No se pudo leer frame {frame_idx}: {path}")
                    continue
                
                # Mejorar imagen
                enhanced_frame = self.enhance_image(original_frame)
                
                # ============ INFERENCIA ============
                results = self.model(
                    enhanced_frame,
                    imgsz=CONFIG.IMGSZ,
                    conf=CONFIG.CONF_THRESHOLD,
                    iou=CONFIG.IOU_THRESHOLD,
                    verbose=False,
                    device='cuda' if self.model.device.type == 'cuda' else 'cpu'
                )[0]
                
                # En lugar de model(frame), usamos el slicer
                detections = self.slicer(enhanced_frame)
                
                # ============ FILTRADO ============
                
                # Filtro por zona
                mask_zone = zone.trigger(detections=detections)
                detections = detections[mask_zone]
                
                # Filtro por clases objetivo
                mask_classes = np.isin(detections.class_id, CONFIG.TARGET_CLASSES)
                detections = detections[mask_classes]
                
                # Filtros avanzados
                detections = detection_filter.apply_all_filters(detections)
                
                # ============ TRACKING ============
                detections = self.tracker.update_with_detections(detections)

                # ============ MEJORA 1: CONTINUIDAD TEMPORAL PARA MOTOCICLETAS ============
                # Se prioriza continuidad temporal de motocicletas por su menor tamaño en UAV
                if detections.tracker_id is not None:
                    for i, (tid, cid) in enumerate(zip(detections.tracker_id, detections.class_id)):
                        if cid == 3:  # Motocicleta
                            # Mantener track activo aunque haya fallos puntuales de detección
                            if vehicle_tracker.stopped_counters.get(int(tid), 0) < 5:
                                continue
                
                # ============ VALIDACIÓN ============
                validation_metrics.update(detections)
                
                # ============ ANÁLISIS POR SECTORES ============
                self.analyze_sectors(detections, frame_idx)
                
                # ============ CONTEO POR CLASE ============
                class_counts = {1: 0, 2: 0, 3: 0, 5: 0, 7: 0}
                for class_id in detections.class_id:
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                
                total_vehicles = len(detections)
                
                # ============ MÉTRICAS DE TRÁFICO ============
                density = engineer.calculate_density(total_vehicles, road_area_m2)
                los_letter, los_desc = engineer.calculate_level_of_service(density)
                los_full = f"{los_letter} ({los_desc})"
                
                # ============ ANÁLISIS DE INCIDENTES ============
                
                velocidades = []
                num_stopped = 0
                num_harsh_braking = 0
                
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    for tid, xyxy, cid in zip(detections.tracker_id, 
                                             detections.xyxy, 
                                             detections.class_id):
                        # Calcular centro
                        cx = (xyxy[0] + xyxy[2]) / 2
                        cy = (xyxy[1] + xyxy[3]) / 2
                        center = np.array([cx, cy])
                        
                        # Actualizar historial
                        vehicle_tracker.update(int(tid), center, frame_idx, int(cid))
                        
                        # Calcular cinemática
                        vel = vehicle_tracker.get_velocity(int(tid))
                        acc = vehicle_tracker.get_acceleration(int(tid))
                        
                        velocidades.append(vel)
                        
                        # Contador de vehículos detenidos
                        if vel < CONFIG.VELOCIDAD_MIN_MOVIMIENTO:
                            vehicle_tracker.stopped_counters[int(tid)] += 1
                        else:
                            vehicle_tracker.stopped_counters[int(tid)] = 0
                        
                        # Detectar vehículo detenido
                        if incident_detector.detect_stopped_vehicle(
                            int(tid), vel, 
                            vehicle_tracker.stopped_counters[int(tid)],
                            center, frame_idx, int(cid)
                        ):
                            num_stopped += 1
                        
                        # Detectar frenada brusca
                        if incident_detector.detect_harsh_braking(
                            int(tid), acc, center, frame_idx, int(cid), vel
                        ):
                            num_harsh_braking += 1
                
                # Detectar conflictos espaciales
                conflicts = incident_detector.detect_conflicts(detections, frame_idx)
                num_conflicts = len(conflicts)
                
                # Detectar densidad peligrosa
                incident_detector.detect_dangerous_density(
                    density, frame_idx, total_vehicles
                )
                
                total_incidents = num_stopped + num_harsh_braking + num_conflicts
                
                # ============ LOGGING DE MÉTRICAS ============
                
                vel_media = np.mean(velocidades) if velocidades else 0.0
                
                self.metrics_log.append({
                    'Frame': frame_idx,
                    'Total': total_vehicles,
                    'Densidad': float(density),
                    'LOS': los_letter,
                    'LOS_Descripcion': los_desc,
                    'Turismos': class_counts[2],
                    'Motos': class_counts[3],
                    'Buses': class_counts[5],
                    'Camiones': class_counts[7],
                    'Bicis': class_counts[1],
                    'Velocidad_Media': float(vel_media),
                    'Incidentes_Detenidos': num_stopped,
                    'Incidentes_Frenadas': num_harsh_braking,
                    'Incidentes_Conflictos': num_conflicts,
                    'Incidentes_Total': total_incidents
                })
                
                # ============ VISUALIZACIÓN ============
                
                # Preparar etiquetas
                labels = []
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    for tid, cid in zip(detections.tracker_id, detections.class_id):
                        class_name = CONFIG.CLASS_NAMES.get(int(cid), '?')
                        labels.append(f"#{int(tid)} {class_name}")
                
                # Anotar frame
                annotated_frame = original_frame.copy()
                annotated_frame = zone_annotator.annotate(scene=annotated_frame)
                annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
                annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
                annotated_frame = self.label_annotator.annotate(
                    annotated_frame, detections, labels=labels
                )
                
                # ============ HUD (Heads-Up Display) ============
                
                # Panel de información
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (550, 280), (0, 0, 0), -1)
                annotated_frame = cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0)
                
                # Título
                cv2.putText(
                    annotated_frame,
                    "AeroTrace v5.0 - Analisis Avanzado de Trafico",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Frame
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_idx}/{len(frames)} ({frame_idx/len(frames)*100:.1f}%)",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )
                
                # Conteo de vehículos
                cv2.putText(
                    annotated_frame,
                    f"Turismos: {class_counts[2]:2d} | Motos: {class_counts[3]:2d} | Bicis: {class_counts[1]:2d}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                
                cv2.putText(
                    annotated_frame,
                    f"Buses: {class_counts[5]:2d} | Camiones: {class_counts[7]:2d} | Total: {total_vehicles:2d}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                
                # Métricas de tráfico
                cv2.putText(
                    annotated_frame,
                    f"Densidad: {density:.1f} veh/km² | Nivel: {los_full}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    2
                )
                
                # Color según LOS
                los_color = (0, 255, 0) if los_letter in ['A', 'B'] else \
                           (0, 165, 255) if los_letter in ['C', 'D'] else \
                           (0, 0, 255)
                
                cv2.rectangle(annotated_frame, (510, 145), (540, 175), los_color, -1)
                
                # Velocidad media
                if vel_media > 0:
                    vel_kmh = vel_media * 3.6
                    cv2.putText(
                        annotated_frame,
                        f"Velocidad Media: {vel_kmh:.1f} km/h",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                
                # Incidentes
                inc_color = (0, 255, 0) if total_incidents == 0 else \
                           (0, 200, 255) if total_incidents < 3 else \
                           (0, 100, 255) if total_incidents < 5 else \
                           (0, 0, 255)
                
                cv2.putText(
                    annotated_frame,
                    f"Incidentes: {total_incidents}",
                    (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    inc_color,
                    2
                )
                
                if total_incidents > 0:
                    cv2.putText(
                        annotated_frame,
                        f"(Detenidos: {num_stopped} | Frenadas: {num_harsh_braking} | Conflictos: {num_conflicts})",
                        (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        inc_color,
                        1
                    )
                
                # Escribir frame al video
                sink.write_frame(annotated_frame)
        
        # ==================== GENERACIÓN DE REPORTES ====================
        
        print("\n" + "="*80)
        print("📝 GENERANDO REPORTES FINALES")
        print("="*80 + "\n")
        
        # DataFrame principal
        df = pd.DataFrame(self.metrics_log)
        csv_path = os.path.join(CONFIG.OUTPUT_DIR, 'metrics', 'metricas_completas_v5.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Métricas exportadas: {csv_path}")
        
        # Reporte de validación
        validation_report = validation_metrics.generate_report()
        validation_path = os.path.join(CONFIG.OUTPUT_DIR, 'validation', 'model_validation.json')
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        print(f"✅ Validación del modelo: {validation_path}")
        
        # Reporte de incidentes
        incident_summary = incident_detector.get_incident_summary()
        incidents_path = os.path.join(CONFIG.OUTPUT_DIR, 'incidents', 'incidents_report.json')
        with open(incidents_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': incident_summary,
                'detailed_incidents': incident_detector.incidents
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Reporte de incidentes: {incidents_path}")
        
        # Dashboard interactivo
        dashboard_path = os.path.join(CONFIG.OUTPUT_DIR, 'dashboard_v5_improved.html')
        DashboardGenerator.create_advanced_dashboard(
            df=df,
            incidents=incident_detector.incidents,
            validation_report=validation_report,
            sector_data=dict(self.sector_metrics),
            output_path=dashboard_path
        )
        
        # ==================== RESUMEN FINAL ====================
        
        print("\n" + "="*80)
        print("✅ ANÁLISIS COMPLETADO - AEROTRACE v5.0 MEJORADO")
        print("="*80)
        
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        print(f"   - Frames procesados: {len(frames)}")
        print(f"   - Total de detecciones: {validation_report['summary']['total_detections']}")
        print(f"   - Clases detectadas: {validation_report['summary']['unique_classes_detected']}")
        
        print(f"\n🚗 DISTRIBUCIÓN VEHICULAR:")
        for class_name, data in validation_report['by_class'].items():
            print(f"   - {class_name}: {data['count']} ({data['percentage']:.1f}%)")
            print(f"     └─ Confianza media: {data['confidence']['mean']:.2%}")
        
        print(f"\n⚠️  INCIDENTES DETECTADOS:")
        print(f"   - Total: {incident_summary['total']}")
        for inc_type, count in incident_summary['by_type'].items():
            print(f"   - {inc_type}: {count}")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   📹 Video: {video_path}")
        print(f"   📊 Métricas: {csv_path}")
        print(f"   📈 Dashboard: {dashboard_path}")
        print(f"   ✅ Validación: {validation_path}")
        print(f"   ⚠️  Incidentes: {incidents_path}")
        
        print("\n" + "="*80)
        print("🎉 PROCESO FINALIZADO CON ÉXITO")
        print("="*80 + "\n")

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    try:
        system = AeroTraceSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ ERROR FATAL: {str(e)}")
        import traceback
        traceback.print_exc()
