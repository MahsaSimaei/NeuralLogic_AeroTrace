import json
import os
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
# ==========================================

# Importar sistema principal (asumiendo que main.py está disponible)
try:
    from main import CONFIG, AeroTraceSystem, DatasetLoader

    MAIN_AVAILABLE = True
except ImportError:
    MAIN_AVAILABLE = False
    st.error("⚠️ No se pudo importar main.py. Asegúrate de que esté en el mismo directorio.")

import plotly.express as px
import plotly.graph_objects as go
import supervision as sv

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="AeroTrace - UAV Traffic Analytics",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AeroTrace v2.0 - Sistema de Análisis de Tráfico con UAV para Hackathon"},
)

# =============================================================================
# ESTILOS CSS PERSONALIZADOS
# =============================================================================

st.markdown(
    """
    <style>
    /* Tema oscuro principal */
    .main {
        background-color: #0e1117;
    }

    /* Tarjetas de métricas */
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1e1e1e;
    }

    /* Botones principales */
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #27ae60;
        box-shadow: 0 4px 8px rgba(46, 204, 113, 0.4);
    }

    /* Alertas */
    .stAlert {
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }

    /* Info boxes */
    .info-box {
        background-color: #1e3a5f;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }

    .success-box {
        background-color: #1e4d2b;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 10px 0;
    }

    .warning-box {
        background-color: #5d4e1a;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def save_uploaded_file(uploaded_file, suffix: str = "") -> Optional[str]:
    """
    Guarda archivo subido en directorio temporal de forma segura

    Args:
        uploaded_file: Archivo de Streamlit
        suffix: Sufijo opcional para el nombre del archivo

    Returns:
        str: Ruta del archivo temporal o None si falla
    """
    try:
        # Validar tamaño del archivo (máx 5000MB)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 5000:
            st.error(f"❌ Archivo demasiado grande: {file_size_mb:.1f} MB (máximo: 5000 MB)")
            return None

        # Crear archivo temporal
        file_extension = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"{suffix}{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        st.success(f"✅ Archivo guardado: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        return temp_path

    except Exception as e:
        st.error(f"❌ Error guardando archivo: {str(e)}")
        return None


def validate_video_file(file_path: str) -> Dict:
    """
    Valida que un archivo de video sea legible

    Args:
        file_path: Ruta al archivo de video

    Returns:
        Dict con información del video o None si es inválido
    """
    try:
        video_info = sv.VideoInfo.from_video_path(file_path)
        return {
            "valid": True,
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "total_frames": video_info.total_frames,
            "duration_sec": video_info.total_frames / video_info.fps,
        }
    except Exception as e:
        st.error(f"❌ Error validando video: {str(e)}")
        return {"valid": False, "error": str(e)}


def validate_zip_structure(zip_path: str) -> Dict:
    """
    Valida la estructura de un archivo ZIP con imágenes

    Args:
        zip_path: Ruta al archivo ZIP

    Returns:
        Dict con información de validación
    """
    try:
        image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            all_files = zip_ref.namelist()
            image_files = [f for f in all_files if Path(f).suffix in image_extensions]

            if not image_files:
                return {
                    "valid": False,
                    "error": "No se encontraron imágenes en el ZIP",
                    "total_files": len(all_files),
                    "image_files": 0,
                }

            return {
                "valid": True,
                "total_files": len(all_files),
                "image_files": len(image_files),
                "first_image": image_files[0] if image_files else None,
                "last_image": image_files[-1] if image_files else None,
            }

    except Exception as e:
        return {"valid": False, "error": f"Error leyendo ZIP: {str(e)}"}


def load_scenes_data(scenes_csv_path: str = "scenes.csv") -> Optional[pd.DataFrame]:
    """
    Carga el archivo scenes.csv si existe

    Args:
        scenes_csv_path: Ruta al archivo scenes.csv

    Returns:
        DataFrame o None si no existe
    """
    if os.path.exists(scenes_csv_path):
        try:
            df = pd.read_csv(scenes_csv_path)
            return df
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar scenes.csv: {e}")
            return None
    return None


def create_quick_stats_viz(df: pd.DataFrame) -> go.Figure:
    """
    Crea visualización rápida de estadísticas principales

    Args:
        df: DataFrame con métricas

    Returns:
        Figura de Plotly
    """
    fig = go.Figure()

    # Flujo acumulado
    fig.add_trace(
        go.Scatter(
            x=df["Frame"],
            y=df["Flujo_Acumulado"],
            mode="lines",
            name="Flujo Acumulado",
            line=dict(color="#2ecc71", width=3),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.1)",
        )
    )

    # Ocupación
    fig.add_trace(
        go.Scatter(
            x=df["Frame"],
            y=df["Ocupacion_Actual"],
            mode="lines",
            name="Ocupación",
            line=dict(color="#e74c3c", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Flujo Acumulado vs Ocupación",
        xaxis_title="Frame",
        yaxis_title="Vehículos",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )

    return fig


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================


def main():
    """Función principal de la aplicación Streamlit"""

    # Header
    st.title("🚁 AeroTrace: Análisis de Tráfico Aéreo con IA")
    st.markdown(
        """
    <div class="info-box">
    <strong>Sistema avanzado para análisis de tráfico con drones UAV</strong><br>
    ✅ Detección y clasificación vehicular con YOLOv8<br>
    ✅ Cálculo de flujo, densidad y ocupación<br>
    ✅ Detección automática de incidentes críticos<br>
    ✅ Análisis de nivel de servicio (LOS)
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Verificar disponibilidad del backend
    if not MAIN_AVAILABLE:
        st.error("❌ El módulo principal (main.py) no está disponible. No se puede continuar.")
        return

    # =============================================================================
    # SIDEBAR: CONFIGURACIÓN
    # =============================================================================

    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")

        # Cargar scenes.csv si existe
        scenes_df = load_scenes_data("scenes.csv")

        # Selector de escena (si scenes.csv está disponible)
        selected_scene = None
        if scenes_df is not None:
            st.subheader("📍 Selección de Escena")
            scene_options = ["Ninguna"] + scenes_df["Sequence"].tolist()
            selected_scene_name = st.selectbox(
                "Escena del Dataset", scene_options, help="Selecciona una escena del dataset UAV"
            )

            if selected_scene_name != "Ninguna":
                selected_scene = selected_scene_name
                scene_info = scenes_df[scenes_df["Sequence"] == selected_scene_name].iloc[0]
                st.info(f"**{scene_info['Scene name']}**\n\n📍 {scene_info['lat']}, {scene_info['long']}")

        st.markdown("---")

        # Parámetros del modelo
        st.subheader("🎯 Detección")
        model_conf = st.slider(
            "Confianza del Modelo",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Umbral de confianza para las detecciones",
        )
        CONFIG.CONF_THRESHOLD = model_conf

        # Límite de frames
        st.subheader("🎞️ Procesamiento")
        max_frames_option = st.selectbox(
            "Frames a procesar",
            ["Todos", "20", "100", "200", "500", "1000", "1500", "3000"],
            index=0,
            help="Limitar frames para testing rápido",
        )

        if max_frames_option == "Todos":
            CONFIG.MAX_FRAMES = None
        else:
            CONFIG.MAX_FRAMES = int(max_frames_option)

        st.markdown("---")

        st.subheader("📐 Configuración de Zona (ROI)")

        # 1. Configuración de la ZONA (Rectángulo)
        col_roi_1, col_roi_2 = st.columns(2)
        with col_roi_1:
            roi_left = st.number_input("X Mín (Izq)", 0, 1920, 0, step=50)
            roi_top = st.number_input("Y Mín (Arriba)", 0, 1080, 360, step=50)
        with col_roi_2:
            roi_right = st.number_input("X Máx (Der)", 0, 1920, 1920, step=50)
            roi_bottom = st.number_input("Y Máx (Abajo)", 0, 1080, 1080, step=50)

        # Actualizar el Polígono en la Configuración Global
        # Orden: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        new_polygon = np.array(
            [[roi_left, roi_top], [roi_right, roi_top], [roi_right, roi_bottom], [roi_left, roi_bottom]]
        )
        CONFIG.ZONE_POLYGON = new_polygon

        # 2. Configuración de la LÍNEA (Debe estar dentro de la zona idealmente)
        st.markdown("---")
        st.subheader("📏 Línea de Conteo")
        line_pos_y = st.slider(
            "Posición Y Línea",
            min_value=roi_top,  # Restringir a la zona
            max_value=roi_bottom,  # Restringir a la zona
            value=int((roi_top + roi_bottom) / 2),  # Por defecto al centro de la zona
            step=10,
            help="Posición vertical de la línea de conteo",
        )

        # Actualizar Línea en Configuración Global
        CONFIG.LINE_START = sv.Point(roi_left, line_pos_y)
        CONFIG.LINE_END = sv.Point(roi_right, line_pos_y)
        st.markdown("---")
        st.subheader("🔒 Auditoría Blockchain")
        use_bsv = st.checkbox(
            "Activar Registro BSV",
            value=True,
            disabled=not BLOCKCHAIN_AVAILABLE,
            help="Generar evidencias criptográficas inmutables (SHA-256)",
        )
        if use_bsv and not BLOCKCHAIN_AVAILABLE:
            st.warning("⚠️ Módulo bsv_blockchain.py no encontrado")
    # =============================================================================
    # ENTRADA DE DATOS
    # =============================================================================

    st.header("📁 Fuente de Datos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎥 Subir Video")
        video_file = st.file_uploader(
            "Selecciona archivo de video",
            type=["mp4", "avi", "mov", "MP4", "AVI", "MOV"],
            help="Formatos soportados: MP4, AVI, MOV",
        )

        if video_file:
            st.success(f"✅ Video cargado: {video_file.name}")

            # Mostrar preview
            st.video(video_file)

    with col2:
        st.subheader("📦 Subir ZIP de Imágenes")
        zip_file = st.file_uploader(
            "Selecciona archivo ZIP", type=["zip", "ZIP"], help="ZIP con imágenes JPG, JPEG o PNG del dataset UAV"
        )

        if zip_file:
            st.success(f"✅ ZIP cargado: {zip_file.name}")

            # Validar estructura del ZIP
            with st.spinner("Validando ZIP..."):
                temp_zip = save_uploaded_file(zip_file, "_temp")
                if temp_zip:
                    validation = validate_zip_structure(temp_zip)

                    if validation["valid"]:
                        st.success(f"""
                        ✅ ZIP válido:
                        - Total de archivos: {validation['total_files']}
                        - Imágenes encontradas: {validation['image_files']}
                        """)
                    else:
                        st.error(f"❌ {validation['error']}")
                        zip_file = None

                    os.unlink(temp_zip)

    # =============================================================================
    # PROCESAMIENTO
    # =============================================================================

    # Determinar fuente de entrada
    input_path = None
    temp_dir = None
    source_type = None

    if video_file and not zip_file:
        source_type = "video"
        input_path = save_uploaded_file(video_file, "_video")

        # Validar video
        if input_path:
            with st.spinner("Validando video..."):
                validation = validate_video_file(input_path)

                if validation["valid"]:
                    st.success(f"""
                    ✅ Video válido:
                    - Resolución: {validation['width']}x{validation['height']}
                    - FPS: {validation['fps']}
                    - Frames totales: {validation['total_frames']}
                    - Duración: {validation['duration_sec']:.1f} segundos
                    """)
                else:
                    st.error(f"❌ Video inválido: {validation.get('error', 'Desconocido')}")
                    input_path = None

    elif zip_file and not video_file:
        source_type = "images"

        # Extraer ZIP
        with st.spinner("Extrayendo imágenes del ZIP..."):
            temp_zip_path = save_uploaded_file(zip_file, "_images")

            if temp_zip_path:
                temp_dir = tempfile.mkdtemp()

                try:
                    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)

                    input_path = temp_dir
                    st.success("✅ Imágenes extraídas en directorio temporal")

                    # Mostrar primera imagen como preview
                    image_files = []
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                        import glob

                        image_files.extend(glob.glob(os.path.join(temp_dir, ext)))

                    if image_files:
                        # Ordenar naturalmente
                        image_files = DatasetLoader.smart_sort_files(image_files)

                        first_img = cv2.imread(image_files[0])
                        if first_img is not None:
                            # --- NUEVO CÓDIGO DE VISUALIZACIÓN ---
                            # Crear una copia para dibujar
                            preview_img = first_img.copy()

                            # 1. Dibujar Zona (Polígono Verde)
                            pts = new_polygon.reshape((-1, 1, 2))
                            cv2.polylines(preview_img, [pts], isClosed=True, color=(0, 255, 0), thickness=4)

                            # 2. Dibujar Línea (Azul)
                            cv2.line(preview_img, (roi_left, line_pos_y), (roi_right, line_pos_y), (255, 0, 0), 4)

                            # 3. Texto descriptivo
                            cv2.putText(
                                preview_img,
                                "ZONA DE ANALISIS",
                                (roi_left + 10, roi_top + 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )

                            # Mostrar en Streamlit
                            first_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                            st.image(
                                first_img_rgb,
                                caption="Preview: Zona de Detección (Verde) y Línea (Azul)",
                                use_container_width=True,
                            )
                            # -------------------------------------

                except Exception as e:
                    st.error(f"❌ Error extrayendo ZIP: {e}")
                    input_path = None

                finally:
                    if temp_zip_path and os.path.exists(temp_zip_path):
                        os.unlink(temp_zip_path)

    elif video_file and zip_file:
        st.warning("⚠️ Por favor, selecciona solo una fuente de datos (video O ZIP)")

    # =============================================================================
    # BOTÓN DE EJECUCIÓN
    # =============================================================================

    st.markdown("---")

    if input_path:

        # Botón centrado con estilo
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_button = st.button(
                "🚀 INICIAR ANÁLISIS",
                type="primary",
                use_container_width=True,
                help="Comenzar el procesamiento del video/imágenes",
            )

        if run_button:

            # Contenedores para la UI en tiempo real
            st.markdown("---")
            st.header("📊 Procesamiento en Tiempo Real")

            # Layout de 2 columnas
            col_video, col_stats = st.columns([2, 1])

            with col_video:
                st.subheader("🎬 Visualización")
                image_placeholder = st.empty()

            with col_stats:
                st.subheader("📈 Métricas en Vivo")

                # Contenedores de métricas
                metric_cols = st.columns(2)
                with metric_cols[0]:
                    metric_flow = st.empty()
                    metric_density = st.empty()

                with metric_cols[1]:
                    metric_occupancy = st.empty()
                    metric_incidents = st.empty()

                # Barra de progreso
                st.markdown("**Progreso:**")
                progress_bar = st.progress(0)
                progress_text = st.empty()

            # =============================================================================
            # CALLBACKS PARA STREAMLIT
            # =============================================================================

            def update_progress(frame_idx: int, progress_value: float):
                """Actualiza la barra de progreso"""
                progress_bar.progress(min(progress_value, 1.0))
                progress_text.text(f"Frame {frame_idx} - {progress_value*100:.1f}%")

            def update_display(frame_rgb, total_flow, density, incidents_count):
                """Actualiza la visualización en tiempo real"""
                # Mostrar imagen
                image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Actualizar métricas
                metric_flow.metric(
                    "🚗 Flujo Total",
                    f"{total_flow}",
                    delta="+1" if total_flow > 0 else None,
                    help="Vehículos totales que han cruzado la línea",
                )

                metric_density.metric(
                    "📊 Densidad", f"{density:.1f}", delta="veh/km²", help="Densidad de tráfico en el área de análisis"
                )

                metric_occupancy.metric("👥 Ocupación", "En proceso", help="Vehículos actualmente en pantalla")

                metric_incidents.metric(
                    "⚠️ Incidentes", f"{incidents_count}", delta_color="inverse", help="Total de incidentes detectados"
                )

            # =============================================================================
            # EJECUTAR SISTEMA
            # =============================================================================

            try:
                with st.spinner("⚙️ Inicializando sistema AeroTrace..."):
                    system = AeroTraceSystem()

                st.info(
                    "🎬 Procesamiento en curso... Esto puede tardar varios minutos "
                    "dependiendo del tamaño del video y tu GPU."
                )

                # Ejecutar con callbacks
                video_out, df_metrics, incidents, blockchain_log = system.run(
                    source_path=input_path,
                    progress_callback=update_progress,
                    display_callback=update_display,
                    scene_name=selected_scene if selected_scene else "output",
                    scenes_csv_path="scenes.csv" if scenes_df is not None else None,
                    enable_blockchain=use_bsv,
                )

                # =============================================================================
                # RESULTADOS
                # =============================================================================

                st.markdown("---")
                st.success("✅ ¡Análisis Completado con Éxito!")

                # Mostrar métricas finales
                st.header("📊 Resultados Finales")

                # KPIs principales
                kpi_cols = st.columns(4)

                with kpi_cols[0]:
                    total_flow = df_metrics["Flujo_Acumulado"].iloc[-1] if not df_metrics.empty else 0
                    st.metric("🚗 Flujo Total", f"{total_flow}")

                with kpi_cols[1]:
                    avg_occupancy = df_metrics["Ocupacion_Actual"].mean() if not df_metrics.empty else 0
                    st.metric("👥 Ocupación Promedio", f"{avg_occupancy:.1f}")

                with kpi_cols[2]:
                    max_density = df_metrics["Densidad"].max() if not df_metrics.empty else 0
                    st.metric("📊 Densidad Máxima", f"{max_density:.1f} veh/km²")

                with kpi_cols[3]:
                    st.metric("⚠️ Incidentes", f"{len(incidents)}")

                # =============================================================================
                # TABS DE RESULTADOS
                # =============================================================================

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["📈 Visualizaciones", "📋 Datos Detallados", "⚠️ Incidentes", "💾 Descargas", "🔗 Blockchain"]
                )

                # TAB 1: VISUALIZACIONES
                with tab1:
                    st.subheader("📈 Análisis Gráfico")

                    # Gráfica principal
                    if not df_metrics.empty:
                        fig_main = create_quick_stats_viz(df_metrics)
                        st.plotly_chart(fig_main, use_container_width=True)

                    # Dashboard HTML (si existe)
                    dashboard_path = os.path.join(
                        CONFIG.OUTPUT_DIR, f'dashboard_{selected_scene if selected_scene else "output"}.html'
                    )

                    if os.path.exists(dashboard_path):
                        st.subheader("🎯 Dashboard Interactivo Completo")

                        try:
                            with open(dashboard_path, "r", encoding="utf-8") as f:
                                html_content = f.read()

                            st.components.v1.html(html_content, height=1200, scrolling=True)

                        except Exception as e:
                            st.error(f"❌ Error cargando dashboard: {e}")
                    else:
                        st.warning("⚠️ Dashboard HTML no disponible")

                    # Distribución por clase
                    if not df_metrics.empty:
                        st.subheader("🚗 Distribución de Vehículos")

                        vehicle_dist = {
                            "Coches": df_metrics["Acum_Coches"].iloc[-1],
                            "Motocicletas": df_metrics["Acum_Motos"].iloc[-1],
                            "Bicicletas": df_metrics["Acum_Bicis"].iloc[-1],
                        }

                        fig_pie = px.pie(
                            values=list(vehicle_dist.values()),
                            names=list(vehicle_dist.keys()),
                            title="Distribución por Tipo de Vehículo",
                            color_discrete_sequence=["#3498db", "#e74c3c", "#f39c12"],
                        )
                        fig_pie.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_pie, use_container_width=True)

                # TAB 2: DATOS DETALLADOS
                with tab2:
                    st.subheader("📋 Tabla de Métricas Completa")

                    # Filtros
                    col_filter1, col_filter2 = st.columns(2)

                    with col_filter1:
                        show_rows = st.slider(
                            "Mostrar filas",
                            min_value=10,
                            max_value=min(len(df_metrics), 1000),
                            value=min(100, len(df_metrics)),
                            step=10,
                        )

                    with col_filter2:
                        columns_to_show = st.multiselect(
                            "Columnas a mostrar",
                            df_metrics.columns.tolist(),
                            default=["Frame", "Flujo_Acumulado", "Ocupacion_Actual", "Densidad", "LOS"],
                        )

                    # Mostrar dataframe
                    if columns_to_show:
                        st.dataframe(df_metrics[columns_to_show].head(show_rows), use_container_width=True, height=400)
                    else:
                        st.dataframe(df_metrics.head(show_rows), use_container_width=True, height=400)

                    # Estadísticas descriptivas
                    st.subheader("📊 Estadísticas Descriptivas")
                    st.dataframe(df_metrics.describe(), use_container_width=True)

                # TAB 3: INCIDENTES
                with tab3:
                    st.subheader("⚠️ Registro de Incidentes")

                    if incidents:
                        # Resumen de incidentes
                        st.metric("Total de Incidentes", len(incidents))

                        # Por tipo
                        incident_types = {}
                        for inc in incidents:
                            inc_type = inc.get("type", "Unknown")
                            incident_types[inc_type] = incident_types.get(inc_type, 0) + 1

                        st.subheader("📊 Por Tipo")
                        for inc_type, count in incident_types.items():
                            st.write(f"**{inc_type}:** {count}")

                        # Mostrar tabla de incidentes
                        st.subheader("📋 Detalle de Incidentes")

                        # Convertir a DataFrame
                        incidents_df = pd.DataFrame(incidents)
                        st.dataframe(incidents_df, use_container_width=True, height=400)

                        # Gráfica de incidentes en el tiempo
                        if "frame" in incidents_df.columns:
                            fig_incidents = px.scatter(
                                incidents_df,
                                x="frame",
                                y="type",
                                color="severity",
                                title="Timeline de Incidentes",
                                color_discrete_map={"ALTA": "#e74c3c", "MEDIA": "#f39c12", "CRITICA": "#c0392b"},
                            )
                            fig_incidents.update_layout(template="plotly_dark")
                            st.plotly_chart(fig_incidents, use_container_width=True)

                    else:
                        st.info("✅ No se detectaron incidentes críticos")

                # TAB 4: DESCARGAS
                with tab4:
                    st.subheader("💾 Archivos Disponibles para Descarga")

                    download_cols = st.columns(2)

                    # Video procesado
                    with download_cols[0]:
                        st.markdown("### 🎥 Video Procesado")

                        if os.path.exists(video_out):
                            file_size_mb = os.path.getsize(video_out) / (1024 * 1024)
                            st.info(f"Tamaño: {file_size_mb:.1f} MB")

                            with open(video_out, "rb") as file:
                                st.download_button(
                                    label="⬇️ Descargar Video",
                                    data=file,
                                    file_name=f"aerotrace_{selected_scene if selected_scene else 'output'}.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                )
                        else:
                            st.error("❌ Video no disponible")

                    # CSV de métricas
                    with download_cols[1]:
                        st.markdown("### 📊 CSV de Métricas")

                        csv_data = df_metrics.to_csv(index=False).encode("utf-8")
                        csv_size_kb = len(csv_data) / 1024
                        st.info(f"Tamaño: {csv_size_kb:.1f} KB")

                        st.download_button(
                            label="⬇️ Descargar CSV",
                            data=csv_data,
                            file_name=f"metrics_{selected_scene if selected_scene else 'output'}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    # JSON de incidentes
                    st.markdown("### ⚠️ JSON de Incidentes")

                    incidents_json_path = os.path.join(
                        CONFIG.OUTPUT_DIR,
                        "incidents",
                        f'incidents_{selected_scene if selected_scene else "output"}.json',
                    )

                    if os.path.exists(incidents_json_path):
                        with open(incidents_json_path, "r", encoding="utf-8") as f:
                            incidents_json = f.read()

                        json_size_kb = len(incidents_json.encode("utf-8")) / 1024
                        st.info(f"Tamaño: {json_size_kb:.1f} KB")

                        st.download_button(
                            label="⬇️ Descargar JSON de Incidentes",
                            data=incidents_json,
                            file_name=f"incidents_{selected_scene if selected_scene else 'output'}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    else:
                        st.warning("⚠️ Archivo de incidentes no disponible")

                    # Dashboard HTML
                    st.markdown("### 📊 Dashboard HTML")

                    if os.path.exists(dashboard_path):
                        with open(dashboard_path, "r", encoding="utf-8") as f:
                            dashboard_html = f.read()

                        html_size_kb = len(dashboard_html.encode("utf-8")) / 1024
                        st.info(f"Tamaño: {html_size_kb:.1f} KB")

                        st.download_button(
                            label="⬇️ Descargar Dashboard HTML",
                            data=dashboard_html,
                            file_name=f"dashboard_{selected_scene if selected_scene else 'output'}.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                    else:
                        st.warning("⚠️ Dashboard HTML no disponible")

                with tab5:
                    st.subheader("🔗 Auditoría de Integridad (BSV Blockchain)")

                    if blockchain_log and os.path.exists(blockchain_log):
                        st.success("✅ Evidencias criptográficas generadas correctamente")

                        # Leer el log
                        with open(blockchain_log, "r", encoding="utf-8") as f:
                            bsv_data = json.load(f)

                        # Mostrar Resumen
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            st.info(f"**Scene ID:** {bsv_data.get('scene_id')}")
                            st.info(f"**Timestamp:** {bsv_data.get('generated_at')}")
                        with col_b2:
                            stats = bsv_data.get("statistics", {})
                            st.metric("Transacciones Generadas", stats.get("total_transactions", 0))
                            st.metric("Imágenes Hasheadas", stats.get("total_images_processed", 0))

                        # Visor de JSON
                        st.markdown("### 📜 Log de Evidencias (JSON)")
                        st.json(bsv_data, expanded=False)

                        # Botón de descarga
                        with open(blockchain_log, "rb") as f:
                            st.download_button(
                                label="⬇️ Descargar Evidencia Blockchain (.json)",
                                data=f,
                                file_name=os.path.basename(blockchain_log),
                                mime="application/json",
                                use_container_width=True,
                            )
                    else:
                        if not use_bsv:
                            st.warning("⚠️ El registro Blockchain estaba desactivado durante el análisis.")
                        else:
                            st.error("❌ No se encontró el archivo de log de Blockchain.")

            except Exception as e:
                st.error(f"❌ Error durante la ejecución: {str(e)}")

                with st.expander("🔍 Ver detalles del error"):
                    st.code(traceback.format_exc())

            finally:
                # Limpieza de archivos temporales
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        st.info("🧹 Archivos temporales eliminados")
                    except Exception as e:
                        st.warning(f"⚠️ No se pudieron eliminar archivos temporales: {e}")

                if input_path and os.path.isfile(input_path) and source_type == "video":
                    try:
                        os.unlink(input_path)
                    except Exception:
                        pass

    else:
        # Mensaje de instrucciones si no hay archivo
        st.markdown(
            """
        <div class="warning-box">
        <h3>📌 Instrucciones de Uso</h3>
        <ol>
            <li><strong>Sube un archivo:</strong> Video (MP4, AVI, MOV) o ZIP con imágenes del dataset UAV</li>
            <li><strong>Configura parámetros:</strong> Ajusta confianza y posición de línea en el sidebar</li>
            <li><strong>Selecciona escena:</strong> Si tienes scenes.csv, elige una escena específica</li>
            <li><strong>Inicia el análisis:</strong> Presiona el botón "Iniciar Análisis"</li>
            <li><strong>Revisa resultados:</strong> Explora gráficas, datos e incidentes detectados</li>
            <li><strong>Descarga resultados:</strong> Video procesado, CSV de métricas y más</li>
        </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Dataset info
        if scenes_df is not None:
            st.markdown("---")
            st.subheader("📊 Dataset UAV - Información")

            st.dataframe(scenes_df, use_container_width=True)

            st.info(f"""
            **Escenas disponibles:** {len(scenes_df)}

            **Tipos de escenas:**
            - Regional road
            - Roundabout (far/near)
            - Rural road
            - Split roundabout
            - Urban intersection
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>AeroTrace v2.0</strong> - Hackathon Edition</p>
    <p>Sistema de Análisis de Tráfico con UAV | Powered by YOLOv8 + ByteTrack + Supervision</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    main()
