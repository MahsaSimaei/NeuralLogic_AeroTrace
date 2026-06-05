import glob
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import supervision as sv
from plotly.subplots import make_subplots
from scipy.spatial import distance
from tqdm import tqdm
from ultralytics import YOLO

try:
    from bsv_blockchain import BSVEvidenceRegistry

    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False


# =============================================================================
# CONFIGURACIÓN OPTIMIZADA CON INTEGRACIÓN DATASET UAV
# =============================================================================


@dataclass
class Config:
    """Configuración centralizada del sistema con soporte para dataset UAV"""

    # Directorios
    SOURCE_IMAGES_DIR: str = "data/frames"
    OUTPUT_DIR: str = "outputs"
    MODEL_NAME: str = "yolov8x.pt"
    MAX_FRAMES: int = None  # None = procesar todos los frames

    # CALIBRACIÓN FÍSICA
    ALTURA_VUELO_M: float = 120.0
    FOV_HORIZONTAL_GRADOS: float = 84.0

    # ZONA DE INTERÉS (ROI) - Ajustable dinámicamente
    ZONE_POLYGON: np.ndarray = field(default_factory=lambda: np.array([[0, 1080], [0, 360], [1920, 360], [1920, 1080]]))

    # LÍNEA DE CONTEO - Ajustable desde Streamlit
    LINE_START: sv.Point = field(default_factory=lambda: sv.Point(0, 700))
    LINE_END: sv.Point = field(default_factory=lambda: sv.Point(1920, 700))

    # SECTORES PARA ANÁLISIS DETALLADO (Rotondas, intersecciones)
    SECTORES: Dict = field(
        default_factory=lambda: {
            "entrada_norte": np.array([[700, 360], [1200, 360], [1200, 600], [700, 600]]),
            "entrada_sur": np.array([[700, 800], [1200, 800], [1200, 1080], [700, 1080]]),
            "entrada_este": np.array([[1300, 600], [1920, 600], [1920, 800], [1300, 800]]),
            "entrada_oeste": np.array([[0, 600], [600, 600], [600, 800], [0, 800]]),
            "zona_central": np.array([[700, 600], [1200, 600], [1200, 800], [700, 800]]),
        }
    )

    # PARÁMETROS DE DETECCIÓN OPTIMIZADOS PARA DATASET UAV
    CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    IMGSZ: int = 1280  # Balance entre velocidad y precisión

    # MAPEO DE CLASES YOLO → DATASET UAV
    # Mantenemos compatibilidad con otras clases por si se usan modelos custom
    CLASS_MAPPING: Dict = field(
        default_factory=lambda: {
            0: "persona",  # No relevante para tráfico vehicular
            1: "bicicleta",  # Útil para movilidad
            2: "coche",  # PRINCIPAL en dataset UAV
            3: "motocicleta",  # PRINCIPAL en dataset UAV
            5: "bus",  # Poco frecuente en dataset
            7: "camión",  # Poco frecuente en dataset
        }
    )

    TARGET_CLASSES: List[int] = field(default_factory=lambda: [2, 3, 5, 7])  # coche , motocicleta , bus y camión

    # PARÁMETROS ESPECÍFICOS POR CLASE
    CLASS_CONF_THRESHOLDS: Dict = field(
        default_factory=lambda: {
            1: 0.20,  # Bicis
            2: 0.25,  # Coches (PRINCIPAL)
            3: 0.20,  # Motos (PRINCIPAL)
            5: 0.35,  # Buses (raros en dataset)
            7: 0.35,  # Camiones (raros en dataset)
        }
    )

    # FILTROS DE TAMAÑO (píxeles) - Ajustados para vista aérea UAV
    CLASS_SIZE_LIMITS: Dict = field(
        default_factory=lambda: {
            1: (200, 5000),  # Bicis más pequeñas en vista aérea
            2: (800, 40000),  # Coches: rango amplio por perspectiva
            3: (200, 4000),  # Motos: muy variables en tamaño
            5: (3000, 80000),  # Buses
            7: (2500, 70000),  # Camiones
        }
    )

    # FILTROS DE ASPECTO (W/H) - Más permisivos para vista cenital
    CLASS_ASPECT_RATIOS: Dict = field(
        default_factory=lambda: {
            1: (0.3, 1.5),
            2: (0.4, 3.0),  # Coches: más variación por ángulo
            3: (0.3, 2.5),  # Motos: muy variable
            5: (1.0, 4.0),
            7: (1.0, 3.5),
        }
    )

    # TRACKING OPTIMIZADO
    TRACK_THRESH: float = 0.20  # Más permisivo para mantener IDs
    TRACK_BUFFER: int = 180  # Buffer largo para oclusiones
    MATCH_THRESH: float = 0.85  # Balance entre continuidad y precisión

    # INCIDENTES
    VELOCIDAD_MIN_MOVIMIENTO: float = 0.3  # m/s (muy lento en vista aérea)
    TIEMPO_VEHICULO_DETENIDO: int = 90  # frames (~3 segundos a 30fps)
    UMBRAL_FRENADA_BRUSCA: float = 2.5  # m/s²
    DISTANCIA_CONFLICTO: float = 3.5  # metros
    UMBRAL_DENSIDAD_PELIGROSA: float = 45  # veh/km²
    UMBRAL_DENSIDAD_CRITICA: float = 65  # veh/km²

    # ANÁLISIS TEMPORAL
    VENTANA_TEMPORAL_FRAMES: int = 30
    VENTANA_ANALISIS_FRAMES: int = 300

    def __post_init__(self):
        """Post-inicialización con mapeo de nombres"""
        # Crear directorios de salida
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "validation"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "incidents"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "reports"), exist_ok=True)


CONFIG = Config()

# =============================================================================
# CARGADOR DE DATASET CON SCENES.CSV
# =============================================================================


class DatasetLoader:
    """Cargador inteligente del dataset UAV con soporte para scenes.csv"""

    def __init__(self, scenes_csv_path: str = None):
        self.scenes_df = None
        if scenes_csv_path and os.path.exists(scenes_csv_path):
            try:
                self.scenes_df = pd.read_csv(scenes_csv_path)
                print(f"✅ Loaded scenes.csv: {len(self.scenes_df)} scenes")
                print(f"   Sequences: {', '.join(self.scenes_df['Sequence'].unique())}")
            except Exception as e:
                print(f"⚠️ Error loading scenes.csv: {e}")

    def get_scene_info(self, sequence_name: str) -> Dict:
        """Obtiene información de una escena específica"""
        if self.scenes_df is None:
            return {"name": sequence_name, "type": "Unknown", "lat": None, "long": None}

        scene_row = self.scenes_df[self.scenes_df["Sequence"] == sequence_name]
        if scene_row.empty:
            return {"name": sequence_name, "type": "Unknown", "lat": None, "long": None}

        return {
            "name": sequence_name,
            "type": scene_row.iloc[0]["Scene name"],
            "lat": scene_row.iloc[0]["lat"],
            "long": scene_row.iloc[0]["long"],
        }

    def list_available_scenes(self) -> List[str]:
        """Lista todas las escenas disponibles"""
        if self.scenes_df is None:
            return []
        return self.scenes_df["Sequence"].tolist()

    @staticmethod
    def smart_sort_files(file_list: List[str]) -> List[str]:
        """
        Ordenamiento inteligente de archivos usando números naturales.
        Soporta formatos: frame_1.jpg, img_0001.png, 0000.jpg, etc.
        """

        def extract_number(filename: str) -> int:
            """Extrae el primer número significativo del nombre"""
            basename = os.path.basename(filename)
            # Buscar todos los números en el nombre
            numbers = re.findall(r"\d+", basename)
            if numbers:
                # Usar el número más largo (generalmente el índice del frame)
                return int(max(numbers, key=len))
            return 0

        try:
            sorted_files = sorted(file_list, key=extract_number)
            print(f"📁 Sorted {len(sorted_files)} files")
            if len(sorted_files) > 0:
                print(f"   First: {os.path.basename(sorted_files[0])}")
                print(f"   Last: {os.path.basename(sorted_files[-1])}")
            return sorted_files
        except Exception as e:
            print(f"⚠️ Error sorting files: {e}, using default sort")
            return sorted(file_list)

    @staticmethod
    def validate_image_file(filepath: str) -> bool:
        """Valida que un archivo sea una imagen válida"""
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False
            h, w = img.shape[:2]
            # Verificar dimensiones mínimas
            if h < 100 or w < 100:
                return False
            return True
        except Exception:
            return False


# =============================================================================
# SISTEMA DE CALIBRACIÓN Y MÉTRICAS
# =============================================================================


class TrafficEngineer:
    """Cálculo avanzado de métricas de ingeniería de tráfico"""

    def __init__(self, image_width_px: int, image_height_px: int):
        # Calibración física basada en altura de vuelo y FOV
        fov_rad = np.radians(CONFIG.FOV_HORIZONTAL_GRADOS)
        ancho_terreno_m = 2 * CONFIG.ALTURA_VUELO_M * np.tan(fov_rad / 2)
        alto_terreno_m = ancho_terreno_m * (image_height_px / image_width_px)
        self.gsd = ancho_terreno_m / image_width_px

        print("🔧 CALIBRACIÓN DEL SISTEMA")
        print(f"   - GSD (Ground Sample Distance): {self.gsd:.4f} m/px")
        print(f"   - Área total visible: {ancho_terreno_m:.1f}m x {alto_terreno_m:.1f}m")

        # Matriz de homografía para transformaciones
        src_points = np.float32([[0, 0], [image_width_px, 0], [0, image_height_px], [image_width_px, image_height_px]])
        dst_points = np.float32([[0, 0], [ancho_terreno_m, 0], [0, alto_terreno_m], [ancho_terreno_m, alto_terreno_m]])
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Thresholds de Nivel de Servicio (HCM 2010)
        self.los_thresholds = {
            "A": (0, 14),  # Flujo libre
            "B": (14, 22),  # Flujo razonablemente libre
            "C": (22, 32),  # Flujo estable
            "D": (32, 45),  # Flujo inestable
            "E": (45, 67),  # Flujo forzado
            "F": (67, float("inf")),  # Colapso
        }

    def pixel_to_meters(self, px_distance: float) -> float:
        """Convierte distancia en píxeles a metros"""
        return px_distance * self.gsd

    def area_px_to_m2(self, area_px: float) -> float:
        """Convierte área en píxeles cuadrados a metros cuadrados"""
        return area_px * (self.gsd**2)

    def calculate_density(self, vehicle_count: int, area_m2: float) -> float:
        """
        Calcula densidad de tráfico (vehículos por km²)
        Fórmula: ρ = N / A * 1,000,000
        """
        if area_m2 == 0:
            return 0.0
        return (vehicle_count / area_m2) * 1_000_000

    def calculate_level_of_service(self, density: float) -> Tuple[str, str]:
        """
        Determina el Nivel de Servicio (Level of Service) basado en HCM
        """
        for los, (min_d, max_d) in self.los_thresholds.items():
            if min_d <= density < max_d:
                descriptions = {
                    "A": "Libre",
                    "B": "Estable",
                    "C": "Limitado",
                    "D": "Denso",
                    "E": "Saturado",
                    "F": "Colapso",
                }
                return los, descriptions[los]
        return "F", "Colapso"

    def calculate_occupancy_rate(self, occupied_area_m2: float, total_area_m2: float) -> float:
        """Calcula porcentaje de ocupación del espacio vial"""
        if total_area_m2 == 0:
            return 0.0
        return (occupied_area_m2 / total_area_m2) * 100


# =============================================================================
# SISTEMA DE TRACKING AVANZADO CON CINEMÁTICA
# =============================================================================


class VehicleTracker:
    """Seguimiento temporal de vehículos con análisis cinemático"""

    def __init__(self, gsd: float):
        self.gsd = gsd  # Ground Sample Distance para conversión px→m
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=CONFIG.VENTANA_ANALISIS_FRAMES))
        self.stopped_counters: Dict[int, int] = defaultdict(int)

    def update(self, tracker_id: int, position: np.ndarray, frame_idx: int, class_id: int):
        """Actualiza historial de posiciones de un vehículo"""
        self.history[tracker_id].append(
            {
                "frame": frame_idx,
                "position": position,
                "timestamp": frame_idx / 30.0,  # Asumiendo 30 FPS
                "class_id": class_id,
            }
        )

    def get_velocity(self, tracker_id: int, window_frames: int = 10) -> float:
        """
        Calcula velocidad instantánea (m/s) usando ventana móvil
        """
        if tracker_id not in self.history:
            return 0.0

        hist = list(self.history[tracker_id])
        if len(hist) < window_frames:
            window_frames = len(hist)
        if window_frames < 2:
            return 0.0

        p_start = hist[-window_frames]["position"]
        p_end = hist[-1]["position"]
        t_start = hist[-window_frames]["timestamp"]
        t_end = hist[-1]["timestamp"]

        dt = t_end - t_start
        if dt == 0:
            return 0.0

        dist_m = np.linalg.norm(p_end - p_start) * self.gsd
        return dist_m / dt  # m/s

    def get_acceleration(self, tracker_id: int) -> float:
        """
        Calcula aceleración (m/s²) comparando dos ventanas de velocidad
        """
        if tracker_id not in self.history:
            return 0.0

        hist = list(self.history[tracker_id])
        if len(hist) < 20:
            return 0.0

        # Velocidad en ventana antigua
        p1, t1 = hist[-20]["position"], hist[-20]["timestamp"]
        p2, t2 = hist[-15]["position"], hist[-15]["timestamp"]
        v1 = (np.linalg.norm(p2 - p1) * self.gsd) / (t2 - t1) if (t2 - t1) > 0 else 0

        # Velocidad en ventana reciente
        p3, t3 = hist[-5]["position"], hist[-5]["timestamp"]
        p4, t4 = hist[-1]["position"], hist[-1]["timestamp"]
        v2 = (np.linalg.norm(p4 - p3) * self.gsd) / (t4 - t3) if (t4 - t3) > 0 else 0

        dt_total = t4 - t1
        return (v2 - v1) / dt_total if dt_total > 0 else 0

    def get_trajectory_length(self, tracker_id: int) -> float:
        """Calcula longitud total de trayectoria (metros)"""
        if tracker_id not in self.history:
            return 0.0

        hist = list(self.history[tracker_id])
        if len(hist) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(hist)):
            p1 = hist[i - 1]["position"]
            p2 = hist[i]["position"]
            dist_px = np.linalg.norm(p2 - p1)
            total_distance += dist_px * self.gsd

        return total_distance


# =============================================================================
# DETECTOR DE INCIDENTES CRÍTICOS
# =============================================================================


class IncidentDetector:
    """Detección avanzada de incidentes y situaciones de riesgo"""

    def __init__(self, gsd: float):
        self.gsd = gsd
        self.incidents: List[Dict] = []
        self.incident_counters = defaultdict(int)  # Evita duplicados

    def detect_stopped_vehicle(
        self,
        tracker_id: int,
        velocity: float,
        stopped_counter: int,
        position: np.ndarray,
        frame_idx: int,
        class_id: int,
    ) -> bool:
        """Detecta vehículo detenido prolongadamente"""
        if velocity < CONFIG.VELOCIDAD_MIN_MOVIMIENTO:
            if stopped_counter >= CONFIG.TIEMPO_VEHICULO_DETENIDO:
                incident_key = f"stopped_{tracker_id}"
                if self.incident_counters[incident_key] == 0:
                    self._add_incident(
                        "VEHICULO_DETENIDO",
                        "ALTA",
                        tracker_id,
                        frame_idx,
                        position,
                        class_id,
                        details={"stopped_frames": stopped_counter, "velocity_ms": velocity},
                    )
                    self.incident_counters[incident_key] = 1
                    return True
        else:
            # Reset si el vehículo se mueve
            self.incident_counters[f"stopped_{tracker_id}"] = 0
        return False

    def detect_harsh_braking(
        self, tracker_id: int, acceleration: float, position: np.ndarray, frame_idx: int, class_id: int, velocity: float
    ) -> bool:
        """Detecta frenada brusca (aceleración negativa pronunciada)"""
        if acceleration < -CONFIG.UMBRAL_FRENADA_BRUSCA and velocity > 1.0:
            incident_key = f"braking_{tracker_id}_{frame_idx//30}"  # 1 por segundo
            if self.incident_counters[incident_key] == 0:
                self._add_incident(
                    "FRENADA_BRUSCA",
                    "MEDIA",
                    tracker_id,
                    frame_idx,
                    position,
                    class_id,
                    details={"acceleration_ms2": acceleration, "velocity_ms": velocity},
                )
                self.incident_counters[incident_key] = 1
                return True
        return False

    def detect_conflicts(self, detections: sv.Detections, frame_idx: int) -> List[Dict]:
        """Detecta conflictos espaciales (vehículos muy cercanos)"""
        conflicts = []
        if len(detections) < 2:
            return conflicts

        centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
        dist_matrix = distance.cdist(centers, centers, "euclidean")

        for i in range(len(dist_matrix)):
            for j in range(i + 1, len(dist_matrix)):
                dist_m = dist_matrix[i][j] * self.gsd
                if dist_m < CONFIG.DISTANCIA_CONFLICTO:
                    conflict_key = f"conflict_{min(i, j)}_{max(i, j)}_{frame_idx // 15}"
                    if self.incident_counters[conflict_key] == 0:
                        incident = self._create_incident_dict("CONFLICTO_ESPACIAL", "ALTA", frame_idx)
                        incident["distance_m"] = float(dist_m)
                        incident["vehicles_involved"] = 2
                        self.incidents.append(incident)
                        conflicts.append(incident)
                        self.incident_counters[conflict_key] = 1

        return conflicts

    def detect_dangerous_density(self, density: float, frame_idx: int, vehicle_count: int):
        """Detecta densidad peligrosa de tráfico"""
        severity = None
        if density > CONFIG.UMBRAL_DENSIDAD_CRITICA:
            severity = "CRITICA"
        elif density > CONFIG.UMBRAL_DENSIDAD_PELIGROSA:
            severity = "ALTA"

        if severity:
            incident_key = f"density_{frame_idx//60}"  # Cada 2 segundos
            if self.incident_counters[incident_key] == 0:
                incident = self._create_incident_dict("DENSIDAD_PELIGROSA", severity, frame_idx)
                incident["density_veh_km2"] = float(density)
                incident["vehicle_count"] = vehicle_count
                self.incidents.append(incident)
                self.incident_counters[incident_key] = 1

    def _add_incident(
        self,
        incident_type: str,
        severity: str,
        tracker_id: int,
        frame: int,
        pos: np.ndarray,
        class_id: int,
        details: Dict = None,
    ):
        """Añade incidente con información completa"""
        inc = self._create_incident_dict(incident_type, severity, frame)
        inc["tracker_id"] = int(tracker_id)
        inc["position"] = pos.tolist()
        inc["vehicle_class"] = CONFIG.CLASS_MAPPING.get(class_id, "Desconocido")
        if details:
            inc.update(details)
        self.incidents.append(inc)

    def _create_incident_dict(self, incident_type: str, severity: str, frame: int) -> Dict:
        """Crea diccionario base de incidente"""
        return {
            "type": incident_type,
            "severity": severity,
            "frame": int(frame),
            "timestamp": datetime.now().isoformat(),
        }

    def get_incident_summary(self) -> Dict:
        """Genera resumen estadístico de incidentes"""
        summary = {"total": len(self.incidents), "by_type": defaultdict(int), "by_severity": defaultdict(int)}
        for incident in self.incidents:
            summary["by_type"][incident["type"]] += 1
            summary["by_severity"][incident["severity"]] += 1
        return dict(summary)

    def export_incidents_json(self, output_path: str):
        """Exporta incidentes a JSON estructurado"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"summary": self.get_incident_summary(), "incidents": self.incidents},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"✅ Incidentes exportados: {output_path}")
        except Exception as e:
            print(f"❌ Error exportando incidentes: {e}")


# =============================================================================
# FILTROS DE DETECCIÓN
# =============================================================================


class DetectionFilter:
    """Filtros avanzados para mejorar precisión de detecciones"""

    @staticmethod
    def filter_by_confidence(detections: sv.Detections) -> sv.Detections:
        """Aplica thresholds de confianza específicos por clase"""
        if len(detections) == 0:
            return detections

        valid_indices = []
        for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
            threshold = CONFIG.CLASS_CONF_THRESHOLDS.get(class_id, CONFIG.CONF_THRESHOLD)
            if conf >= threshold:
                valid_indices.append(i)

        if not valid_indices:
            return sv.Detections.empty()

        return detections[valid_indices]

    @staticmethod
    def filter_by_size(detections: sv.Detections) -> sv.Detections:
        """Filtra por tamaño de bounding box"""
        if len(detections) == 0:
            return detections

        valid_indices = []
        for i, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            area = width * height

            if class_id in CONFIG.CLASS_SIZE_LIMITS:
                min_area, max_area = CONFIG.CLASS_SIZE_LIMITS[class_id]
                if min_area <= area <= max_area:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        if not valid_indices:
            return sv.Detections.empty()

        return detections[valid_indices]

    @staticmethod
    def filter_by_aspect_ratio(detections: sv.Detections) -> sv.Detections:
        """Filtra por relación de aspecto"""
        if len(detections) == 0:
            return detections

        valid_indices = []
        for i, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]

            if height == 0:
                continue

            aspect_ratio = width / height

            if class_id in CONFIG.CLASS_ASPECT_RATIOS:
                min_ar, max_ar = CONFIG.CLASS_ASPECT_RATIOS[class_id]
                if min_ar <= aspect_ratio <= max_ar:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        if not valid_indices:
            return sv.Detections.empty()

        return detections[valid_indices]

    @staticmethod
    def apply_all_filters(detections: sv.Detections) -> sv.Detections:
        """Aplica todos los filtros en cascada"""
        detections = DetectionFilter.filter_by_confidence(detections)
        detections = DetectionFilter.filter_by_size(detections)
        detections = DetectionFilter.filter_by_aspect_ratio(detections)
        return detections


# Clase auxiliar para convertir números de NumPy a nativos de Python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# =============================================================================
# GENERADOR DE DASHBOARD INTERACTIVO
# =============================================================================


class DashboardGenerator:
    """Generador de dashboard HTML interactivo con Plotly"""

    @staticmethod
    def create_advanced_dashboard(
        df: pd.DataFrame,
        incidents: List[Dict],
        validation_report: Dict,
        sector_data: Dict,
        output_path: str,
        scene_info: Dict = None,
    ):
        """
        Crea dashboard completo con múltiples visualizaciones
        """
        print("📊 Generando Dashboard Interactivo...")

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=(
                "📈 Flujo Acumulado vs Ocupación Instantánea",
                "🚗 Distribución por Clase (Acumulada)",
                "🚦 Nivel de Servicio (LOS)",
                "⚠️ Incidentes por Tipo",
                "📍 Densidad Temporal",
                "⚡ Estadísticas de Velocidad",
                "🎯 Detecciones por Frame",
                "📊 Ocupación Media",
                "🔥 Tendencia de Flujo",
                "📉 Comparativa Clases",
                "⏱️ Timeline de Incidentes",
                "🎯 KPIs Principales",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        # 1.1 Flujo Acumulado vs Ocupación (GRÁFICA PRINCIPAL)
        fig.add_trace(
            go.Scatter(
                x=df["Frame"],
                y=df["Flujo_Acumulado"],
                mode="lines",
                name="Flujo Acumulado Total",
                line=dict(color="#2ecc71", width=3),
                fill="tozeroy",
                fillcolor="rgba(46, 204, 113, 0.1)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Frame"],
                y=df["Ocupacion_Actual"],
                mode="lines",
                name="Ocupación Instantánea",
                line=dict(color="#e74c3c", width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # 1.2 Pie chart de distribución por clase (Acumulados)
        last_row = df.iloc[-1] if not df.empty else pd.Series()
        vehicle_counts = {
            "Coches": last_row.get("Acum_Coches", 0),
            "Motocicletas": last_row.get("Acum_Motos", 0),
            "Bicicletas": last_row.get("Acum_Bicis", 0),
        }
        fig.add_trace(
            go.Pie(
                labels=list(vehicle_counts.keys()),
                values=list(vehicle_counts.values()),
                hole=0.4,
                marker=dict(colors=["#3498db", "#e74c3c", "#f39c12"]),
            ),
            row=1,
            col=2,
        )

        # 1.3 Level of Service (LOS)
        los_counts = df["LOS"].value_counts().reindex(["A", "B", "C", "D", "E", "F"], fill_value=0)
        colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
        fig.add_trace(
            go.Bar(
                x=los_counts.index,
                y=los_counts.values,
                marker_color=colors,
                text=los_counts.values,
                textposition="auto",
            ),
            row=1,
            col=3,
        )

        # 2.1 Incidentes por tipo
        if incidents:
            incident_types = {}
            for inc in incidents:
                inc_type = inc.get("type", "Unknown")
                incident_types[inc_type] = incident_types.get(inc_type, 0) + 1

            fig.add_trace(
                go.Bar(
                    x=list(incident_types.keys()),
                    y=list(incident_types.values()),
                    marker_color="#e74c3c",
                    text=list(incident_types.values()),
                    textposition="auto",
                ),
                row=2,
                col=1,
            )

        # 2.2 Densidad temporal
        fig.add_trace(
            go.Scatter(
                x=df["Frame"],
                y=df["Densidad"],
                mode="lines",
                name="Densidad (veh/km²)",
                line=dict(color="#9b59b6", width=2),
                fill="tozeroy",
            ),
            row=2,
            col=2,
        )

        # 3.1 Detecciones por frame
        fig.add_trace(
            go.Scatter(
                x=df["Frame"],
                y=df["Inst_Coches"] + df["Inst_Motos"] + df["Inst_Bicis"],
                mode="markers",
                name="Total Detecciones",
                marker=dict(size=3, color="#3498db"),
            ),
            row=3,
            col=1,
        )

        # 3.2 Ocupación media
        if not df.empty:
            mean_occupancy = df["Ocupacion_Actual"].mean()
            fig.add_trace(
                go.Bar(
                    x=["Media"],
                    y=[mean_occupancy],
                    marker_color="#1abc9c",
                    text=[f"{mean_occupancy:.1f}"],
                    textposition="auto",
                ),
                row=3,
                col=2,
            )

        # 3.3 Tendencia de flujo
        if len(df) > 10:
            window = min(30, len(df) // 10)
            df["Flujo_Smooth"] = df["Flujo_Acumulado"].rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["Frame"],
                    y=df["Flujo_Smooth"],
                    mode="lines",
                    name="Tendencia Suavizada",
                    line=dict(color="#16a085", width=2),
                ),
                row=3,
                col=3,
            )

        # 4.3 KPI Indicator (Total de incidentes)
        fig.add_trace(
            go.Indicator(
                mode="number+delta+gauge",
                value=len(incidents),
                title={"text": "⚠️ Incidentes Detectados"},
                delta={"reference": 0},
                gauge={
                    "axis": {"range": [0, max(len(incidents), 10)]},
                    "bar": {"color": "#e74c3c"},
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": len(incidents) * 0.8,
                    },
                },
            ),
            row=4,
            col=3,
        )

        # Layout general
        title_text = "AeroTrace - Dashboard de Análisis de Tráfico UAV"
        if scene_info:
            title_text += f" | Escena: {scene_info.get('name', 'N/A')} ({scene_info.get('type', 'N/A')})"

        fig.update_layout(
            template="plotly_dark", height=1600, title_text=title_text, showlegend=True, font=dict(size=10)
        )

        # Guardar
        fig.write_html(output_path)
        print(f"✅ Dashboard guardado: {output_path}")

        # También crear un resumen en JSON
        summary_path = output_path.replace(".html", "_summary.json")
        summary_data = {
            "scene_info": scene_info,
            "total_frames": len(df),
            "total_flow": int(df["Flujo_Acumulado"].iloc[-1]) if not df.empty else 0,
            "avg_occupancy": float(df["Ocupacion_Actual"].mean()) if not df.empty else 0,
            "max_density": float(df["Densidad"].max()) if not df.empty else 0,
            "total_incidents": len(incidents),
            "vehicle_distribution": vehicle_counts,
            "los_distribution": los_counts.to_dict(),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"✅ Resumen guardado: {summary_path}")


# =============================================================================
# SISTEMA PRINCIPAL - AEROTRACE
# =============================================================================


class AeroTraceSystem:
    """
    Sistema principal de análisis de tráfico con UAV
    Optimizado para dataset de imágenes aéreas con formato YOLO
    """

    def __init__(self, model_name: str = None):
        print("=" * 60)
        print("🚀 INICIANDO AEROTRACE v2.0 - HACKATHON EDITION")
        print("=" * 60)

        # Cargar modelo YOLO
        model_path = model_name if model_name else CONFIG.MODEL_NAME
        print(f"📦 Cargando modelo: {model_path}")
        self.model = YOLO(model_path)

        # Tracker ByteTrack
        self.tracker = sv.ByteTrack(
            track_activation_threshold=CONFIG.TRACK_THRESH,  # Antes: track_thresh
            lost_track_buffer=CONFIG.TRACK_BUFFER,  # Antes: track_buffer
            minimum_matching_threshold=CONFIG.MATCH_THRESH,  # Antes: match_thresh
            frame_rate=30,
        )  # TRACKING OPTIMIZADO
        print("🎯 Tracker inicializado: ByteTrack")

        # Sistema de conteo por línea (CRÍTICO para flujo)
        self.line_zone = sv.LineZone(start=CONFIG.LINE_START, end=CONFIG.LINE_END)
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2, text_thickness=1, text_scale=0.5, color=sv.Color.WHITE
        )

        # Contador acumulativo persistente (por clase)
        self.cumulative_counts = defaultdict(int)

        # Anotadores visuales
        self.box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.CLASS)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

        # Logs de métricas
        self.metrics_log = []
        self.sector_metrics = defaultdict(list)

        # Slicer para inferencia en imágenes grandes
        def callback(image: np.ndarray) -> sv.Detections:
            results = self.model(image, imgsz=CONFIG.IMGSZ, verbose=False, conf=CONFIG.CONF_THRESHOLD)[0]
            return sv.Detections.from_ultralytics(results)

        self.slicer = sv.InferenceSlicer(
            callback=callback, slice_wh=(640, 640), overlap_wh=(0.2, 0.2), iou_threshold=CONFIG.IOU_THRESHOLD
        )

        print("✅ Sistema inicializado correctamente")
        print("=" * 60)

    def run(
        self,
        source_path: str = None,
        progress_callback: Callable = None,
        display_callback: Callable = None,
        scene_name: str = None,
        scenes_csv_path: str = None,
        enable_blockchain: bool = False,
    ) -> Tuple[str, pd.DataFrame, List[Dict], Optional[str]]:
        """
        Ejecuta el análisis completo del sistema

        Args:
            source_path: Ruta a directorio de imágenes o video
            progress_callback: Callback para actualizar progreso (Streamlit)
            display_callback: Callback para mostrar frames (Streamlit)
            scene_name: Nombre de la escena (para metadata)
            scenes_csv_path: Ruta al archivo scenes.csv

        Returns:
            Tuple: (video_output_path, dataframe_metrics, incidents_list)
        """

        # Cargar información de escenas
        dataset_loader = DatasetLoader(scenes_csv_path)
        scene_info = dataset_loader.get_scene_info(scene_name) if scene_name else {}

        print(f"\n{'='*60}")
        print("🎬 INICIANDO PROCESAMIENTO")
        # --- BLOQUE BLOCKCHAIN 1: INICIALIZACIÓN ---
        bsv_registry = None
        if enable_blockchain and BLOCKCHAIN_AVAILABLE:
            print("🔗 Inicializando Registro Blockchain BSV...")
            loc = None
            if scene_info.get("lat") and scene_info.get("long"):
                loc = {"lat": scene_info["lat"], "long": scene_info["long"]}

            try:
                bsv_registry = BSVEvidenceRegistry(
                    scene_id=scene_name or "unknown",
                    location=loc,
                    output_dir=os.path.join(CONFIG.OUTPUT_DIR, "blockchain_evidence"),
                )
            except Exception as e:
                print(f"⚠️ Error iniciando blockchain: {e}")
        # -------------------------------------------
        if scene_info.get("type"):
            print(f"📍 Escena: {scene_info['name']} - {scene_info['type']}")
            print(f"🌍 Coordenadas: {scene_info['lat']}, {scene_info['long']}")
        print(f"{'='*60}\n")

        # Determinar fuente de datos
        src = source_path if source_path else CONFIG.SOURCE_IMAGES_DIR

        if not os.path.exists(src):
            raise FileNotFoundError(f"❌ Fuente no encontrada: {src}")

        # Determinar si es video o directorio de imágenes
        is_video = False
        total_frames = 0

        if os.path.isdir(src):
            # Directorio de imágenes
            print("📁 Modo: Directorio de imágenes")
            files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                files.extend(glob.glob(os.path.join(src, ext)))

            if not files:
                raise ValueError(f"❌ No se encontraron imágenes en: {src}")

            # ORDENAMIENTO CRÍTICO
            files = dataset_loader.smart_sort_files(files)

            # Validar primeros archivos
            print("🔍 Validando archivos...")
            valid_files = []
            for f in tqdm(files[: min(10, len(files))], desc="Validando muestra"):
                if dataset_loader.validate_image_file(f):
                    valid_files.append(f)

            if not valid_files:
                raise ValueError("❌ No se encontraron imágenes válidas")

            print(f"✅ Archivos válidos encontrados: {len(files)}")
            total_frames = len(files)

            # Leer primera imagen para dimensiones
            first_frame = cv2.imread(files[0])
            if first_frame is None:
                raise ValueError(f"❌ No se pudo leer la primera imagen: {files[0]}")

            h, w = first_frame.shape[:2]

        else:
            # Video
            print("🎥 Modo: Video")
            if not os.path.isfile(src):
                raise ValueError(f"❌ Archivo no encontrado: {src}")

            try:
                video_info = sv.VideoInfo.from_video_path(src)
                w, h = video_info.width, video_info.height
                total_frames = video_info.total_frames
                is_video = True
                files = sv.get_video_frames_generator(src)
                print(f"✅ Video cargado: {w}x{h}, {total_frames} frames")
            except Exception as e:
                raise ValueError(f"❌ Error cargando video: {e}")

        print(f"📐 Dimensiones: {w}x{h}")
        print(f"🎞️  Total de frames: {total_frames}")

        # Inicializar componentes
        engineer = TrafficEngineer(w, h)
        vehicle_tracker = VehicleTracker(engineer.gsd)
        incident_detector = IncidentDetector(engineer.gsd)

        # Zona ROI
        zone = sv.PolygonZone(polygon=CONFIG.ZONE_POLYGON, triggering_anchors=[sv.Position.CENTER])

        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, color=sv.Color.from_hex("#00FF00"), thickness=2, text_thickness=1, text_scale=0.5
        )
        road_area_m2 = engineer.area_px_to_m2(cv2.contourArea(CONFIG.ZONE_POLYGON.astype(np.int32)))
        print(f"🛣️  Área de análisis: {road_area_m2:.2f} m²")

        # Configurar iterador
        if CONFIG.MAX_FRAMES and CONFIG.MAX_FRAMES < total_frames:
            print(f"⚠️  Limitando a {CONFIG.MAX_FRAMES} frames (config)")
            total_frames = CONFIG.MAX_FRAMES

        iterator = files if not is_video else files
        if not is_video:
            iterator = tqdm(files, total=total_frames, desc="Procesando")
        else:
            iterator = tqdm(files, total=total_frames, desc="Procesando")

        # Video de salida
        video_output_path = os.path.join(
            CONFIG.OUTPUT_DIR, f"video_processed_{scene_name if scene_name else 'output'}.mp4"
        )

        print(f"\n🎬 Iniciando procesamiento de {total_frames} frames...")
        print(f"💾 Output: {video_output_path}\n")

        # Pipeline principal
        with sv.VideoSink(video_output_path, sv.VideoInfo(width=w, height=h, fps=30)) as sink:

            for frame_idx, frame_data in enumerate(iterator):

                # Limitar frames si está configurado
                if CONFIG.MAX_FRAMES and frame_idx >= CONFIG.MAX_FRAMES:
                    break

                # Leer frame
                if is_video:
                    frame = frame_data
                else:
                    frame = cv2.imread(frame_data)

                if frame is None:
                    print(f"⚠️ Frame {frame_idx} no válido, saltando...")
                    continue
                # --- BLOQUE BLOCKCHAIN 2: REGISTRO DE FRAME ---
                if bsv_registry and frame_idx % 30 == 0:  # Cada 30 frames
                    try:
                        bsv_registry.register_image(image_array=frame)
                    except Exception as e:
                        print(f"⚠️ Error hashing frame: {e}")
                # ----------------------------------------------

                # =================================================================
                # PASO 1: DETECCIÓN CON YOLO + SLICER
                # =================================================================
                detections = self.slicer(frame)

                # Filtrar por zona ROI
                detections = detections[zone.trigger(detections=detections)]

                # Filtrar por clases objetivo (solo cars y motorcycles para dataset UAV)
                detections = detections[np.isin(detections.class_id, CONFIG.TARGET_CLASSES)]

                # Aplicar filtros de calidad
                detections = DetectionFilter.apply_all_filters(detections)

                # =================================================================
                # PASO 2: TRACKING
                # =================================================================
                detections = self.tracker.update_with_detections(detections)

                # =================================================================
                # PASO 3: CONTEO DE FLUJO (CRÍTICO)
                # =================================================================
                # LineZone.trigger devuelve (crossed_in, crossed_out)
                crossed_in, crossed_out = self.line_zone.trigger(detections=detections)

                # Actualizar contadores acumulativos
                for idx, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
                    if is_in or is_out:  # Cualquier cruce cuenta
                        class_id = detections.class_id[idx]
                        self.cumulative_counts[class_id] += 1

                # =================================================================
                # PASO 4: MÉTRICAS INSTANTÁNEAS
                # =================================================================
                # Ocupación (vehículos en pantalla ahora)
                occupancy_counts = defaultdict(int)
                for cid in detections.class_id:
                    occupancy_counts[cid] += 1

                total_occupancy = len(detections)
                total_cumulative = sum(self.cumulative_counts.values())

                # Densidad y Level of Service
                density = engineer.calculate_density(total_occupancy, road_area_m2)
                los, los_desc = engineer.calculate_level_of_service(density)

                # =================================================================
                # PASO 5: DETECCIÓN DE INCIDENTES
                # =================================================================
                num_incidents_frame = 0

                if detections.tracker_id is not None and len(detections) > 0:
                    for tid, xyxy, cid in zip(detections.tracker_id, detections.xyxy, detections.class_id):
                        # Centro del vehículo
                        center = np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])

                        # Actualizar tracking
                        vehicle_tracker.update(int(tid), center, frame_idx, int(cid))

                        # Calcular cinemática
                        vel = vehicle_tracker.get_velocity(int(tid))
                        acc = vehicle_tracker.get_acceleration(int(tid))

                        # Detectar vehículo detenido
                        if vel < CONFIG.VELOCIDAD_MIN_MOVIMIENTO:
                            vehicle_tracker.stopped_counters[int(tid)] += 1
                        else:
                            vehicle_tracker.stopped_counters[int(tid)] = 0

                        # Incidentes
                        if incident_detector.detect_stopped_vehicle(
                            int(tid), vel, vehicle_tracker.stopped_counters[int(tid)], center, frame_idx, int(cid)
                        ):
                            num_incidents_frame += 1

                        if incident_detector.detect_harsh_braking(int(tid), acc, center, frame_idx, int(cid), vel):
                            num_incidents_frame += 1

                # Detectar conflictos espaciales
                conflicts = incident_detector.detect_conflicts(detections, frame_idx)
                num_incidents_frame += len(conflicts)

                # Detectar densidad peligrosa
                incident_detector.detect_dangerous_density(density, frame_idx, total_occupancy)

                # =================================================================
                # PASO 6: LOGGING DE MÉTRICAS
                # =================================================================
                self.metrics_log.append(
                    {
                        "Frame": frame_idx,
                        "Ocupacion_Actual": total_occupancy,
                        "Flujo_Acumulado": total_cumulative,
                        "Densidad": density,
                        "LOS": los,
                        # Instantáneos
                        "Inst_Coches": occupancy_counts[2],
                        "Inst_Motos": occupancy_counts[3],
                        "Inst_Bicis": occupancy_counts[1],
                        # Acumulados
                        "Acum_Coches": self.cumulative_counts[2],
                        "Acum_Motos": self.cumulative_counts[3],
                        "Acum_Bicis": self.cumulative_counts[1],
                        "Acum_Buses": self.cumulative_counts[5],
                        "Acum_Camiones": self.cumulative_counts[7],
                        "Incidentes_Total": len(incident_detector.incidents),
                        "Incidentes_Frame": num_incidents_frame,
                    }
                )

                # =================================================================
                # PASO 7: ANOTACIONES VISUALES
                # =================================================================
                # Labels con ID y clase
                labels = []
                if detections.tracker_id is not None:
                    for tid, cid in zip(detections.tracker_id, detections.class_id):
                        class_name = CONFIG.CLASS_MAPPING.get(cid, "?")
                        labels.append(f"#{tid} {class_name}")

                # Anotar zona ROI
                frame = zone_annotator.annotate(scene=frame)

                # Anotar línea de conteo
                frame = self.line_zone_annotator.annotate(frame, line_counter=self.line_zone)

                # Anotar trazas
                frame = self.trace_annotator.annotate(frame, detections)

                # Anotar bounding boxes
                frame = self.box_annotator.annotate(frame, detections)

                # Anotar labels
                frame = self.label_annotator.annotate(frame, detections, labels=labels)

                # Texto con métricas principales
                metrics_text = [
                    f"Frame: {frame_idx}/{total_frames}",
                    f"Flujo Total: {total_cumulative}",
                    f"Ocupacion: {total_occupancy}",
                    f"Densidad: {density:.1f} veh/km2",
                    f"LOS: {los} ({los_desc})",
                    f"Incidentes: {len(incident_detector.incidents)}",
                ]

                y_offset = 30
                for text in metrics_text:
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25

                # Escribir frame en video
                sink.write_frame(frame)

                # =================================================================
                # PASO 8: CALLBACKS PARA STREAMLIT
                # =================================================================
                if progress_callback:
                    progress_value = (frame_idx + 1) / total_frames
                    progress_callback(frame_idx, progress_value)

                if display_callback and frame_idx % 3 == 0:  # Cada 3 frames
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    display_callback(frame_rgb, total_cumulative, density, len(incident_detector.incidents))

        print(f"\n✅ Procesamiento completado: {frame_idx + 1} frames")

        # =================================================================
        # PASO 9: GENERACIÓN DE SALIDAS FINALES
        # =================================================================
        print("\n📊 Generando reportes y visualizaciones...")

        # DataFrame de métricas
        df = pd.DataFrame(self.metrics_log)
        metrics_csv_path = os.path.join(
            CONFIG.OUTPUT_DIR, "metrics", f'metrics_{scene_name if scene_name else "output"}.csv'
        )
        df.to_csv(metrics_csv_path, index=False)
        print(f"✅ Métricas guardadas: {metrics_csv_path}")

        # Dashboard HTML interactivo
        dashboard_path = os.path.join(CONFIG.OUTPUT_DIR, f'dashboard_{scene_name if scene_name else "output"}.html')
        DashboardGenerator.create_advanced_dashboard(
            df, incident_detector.incidents, {}, dict(self.sector_metrics), dashboard_path, scene_info
        )

        # Exportar incidentes
        incidents_json_path = os.path.join(
            CONFIG.OUTPUT_DIR, "incidents", f'incidents_{scene_name if scene_name else "output"}.json'
        )
        incident_detector.export_incidents_json(incidents_json_path)

        # Resumen final
        print(f"\n{'='*60}")
        print("📈 RESUMEN FINAL")
        print(f"{'='*60}")
        print(f"✅ Flujo Total: {total_cumulative} vehículos")
        print(f"✅ Ocupación Promedio: {df['Ocupacion_Actual'].mean():.1f} vehículos")
        print(f"✅ Densidad Máxima: {df['Densidad'].max():.1f} veh/km²")
        print(f"✅ Incidentes Detectados: {len(incident_detector.incidents)}")
        print(f"   - Por tipo: {incident_detector.get_incident_summary()['by_type']}")
        print("✅ Distribución de vehículos:")
        print(f"   - Coches: {self.cumulative_counts[2]}")
        print(f"   - Motocicletas: {self.cumulative_counts[3]}")
        print(f"   - Bicicletas: {self.cumulative_counts[1]}")
        print(f"{'='*60}\n")

        # --- BLOQUE BLOCKCHAIN 3: EXPORTACIÓN FINAL ---

        blockchain_log_path = None
        if bsv_registry:
            print("\n🔗 Generando evidencias blockchain...")
            try:
                summary_metrics = {
                    "total_flow": int(total_cumulative),
                    "max_density": float(df["Densidad"].max()) if not df.empty else 0,
                    "total_incidents": len(incident_detector.incidents),
                }

                an_ev = bsv_registry.register_analysis(
                    metrics=summary_metrics,
                    total_vehicles=int(total_cumulative),
                    incident_count=len(incident_detector.incidents),
                    processing_time_sec=0,
                )

                for img_ev in bsv_registry.image_evidences:
                    bsv_registry.create_blockchain_transaction(
                        image_evidence=img_ev, analysis_evidence=an_ev, chain_stage="processed_data"
                    )

                blockchain_log_path = bsv_registry.export_evidence_log()
                bsv_registry.export_supply_chain_format()
                print(f"✅ Log Blockchain generado: {blockchain_log_path}")

            except Exception as e:
                print(f"❌ Error generando log blockchain: {e}")

        # ----------------------------------------------

        # RETORNO ACTUALIZADO (Devuelve 4 valores para que app.py reciba el log)
        return video_output_path, df, incident_detector.incidents, blockchain_log_path


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    """Modo standalone para testing"""

    print("🚀 Modo Standalone - Testing AeroTrace")

    # Configuración de ejemplo
    CONFIG.SOURCE_IMAGES_DIR = "data/frames"  # Ajustar según tu estructura
    CONFIG.MAX_FRAMES = 500  # Limitar para testing

    # Instanciar sistema
    system = AeroTraceSystem()

    # Ejecutar
    try:
        video_path, metrics_df, incidents = system.run(
            scene_name="test_scene", scenes_csv_path="scenes.csv"  # Si existe
        )

        print("\n✅ Ejecución completada exitosamente")
        print(f"📹 Video: {video_path}")
        print(f"📊 Métricas: {len(metrics_df)} frames procesados")
        print(f"⚠️ Incidentes: {len(incidents)}")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
