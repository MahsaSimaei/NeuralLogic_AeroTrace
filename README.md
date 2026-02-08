# 🚁 AeroTrace v2.0
## Sistema Inteligente de Análisis de Tráfico Rodado mediante Imágenes Aéreas

> **Hackathon NeuralHack Smart Cities & Blockchain** > **Equipo:** NeuralLogic  
> **Miembro:** Mahsa Simaei  
> **Fecha de entrega:** 08/02/2026

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8x-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Blockchain](https://img.shields.io/badge/Blockchain-BSV-orange)

---

## 📖 Resumen Ejecutivo

**AeroTrace** es una solución integral de ingeniería de tráfico diseñada para procesar imágenes capturadas por Vehículos Aéreos No Tripulados (UAV/Drones). El sistema combina Visión Artificial avanzada, cálculos de física vectorial y tecnología Blockchain para ofrecer una herramienta de auditoría de tráfico inmutable.

En el contexto de las **Smart Cities**, AeroTrace resuelve la falta de datos granulares mediante:
1.  **Flexibilidad Espacial:** Análisis bajo demanda sin infraestructura fija.
2.  **Precisión Extrema:** Uso de *Slicing Inference* para detectar vehículos pequeños desde 120m de altura.
3.  **Confianza Criptográfica:** Registro de evidencias en Blockchain BSV.



---

## ⚙️ Arquitectura del Sistema

El sistema sigue una arquitectura modular de tubería (*pipeline*) de datos:

`Input (Video/Imágenes)` → `Detección AI (YOLOv8x + SAHI)` → `Tracking (ByteTrack)` → `Métricas (TrafficEngineer)` → `Blockchain (BSV Registry)` → `Visualización (Streamlit)`



[Image of System Architecture Diagram]


### Stack Tecnológico

| Componente | Tecnología | Función Principal |
| :--- | :--- | :--- |
| **Core AI** | Ultralytics YOLOv8x | Detección de objetos (Modelo Extra Large). |
| **Inferencia** | Supervision (SAHI) | *Slicing* para imágenes de alta resolución (4K). |
| **Tracking** | ByteTrack | Seguimiento persistente con resistencia a oclusiones. |
| **Física** | TrafficEngineer (Custom) | Cálculo de GSD, densidad y LOS (HCM 2010). |
| **Integridad** | BSV Blockchain Lib | Hashing SHA-256 y cadena de custodia ETL. |
| **Frontend** | Streamlit + Plotly | Interfaz interactiva y dashboards. |

---

## 🧠 Módulo de Visión Artificial

### Selección del Modelo: YOLOv8x
Hemos seleccionado la versión **Extra Large** de YOLOv8. Aunque computacionalmente más costosa (4.88 ms), es necesaria para vistas aéreas donde un vehículo puede ocupar apenas **40x40 píxeles**.

### Estrategia de Inferencia: Slicing (SAHI)
Para evitar la pérdida de información al redimensionar imágenes 4K a 640p, implementamos **Slicing Inference**:
* La imagen se divide en recortes (*slices*) con un solapamiento del 20%.
* Permite detectar pequeños objetos que desaparecerían en una inferencia estándar.

### Filtros y Clasificación
Mapeamos las clases COCO a categorías de movilidad urbana con filtros geométricos estrictos:
* **Coches:** Área 800 - 40,000 px.
* **Motos:** Aspecto 0.3 - 2.5.
* **Pesados (Bus/Camión):** Umbral de confianza > 0.35.

---

## 📊 Métricas de Movilidad y Física

El sistema no solo cuenta, **mide**. Utilizamos un motor de física personalizado:

1.  **Calibración Espacial (GSD):** Conversión de píxeles a metros basada en la altura de vuelo (120m) y FOV (84°).
2.  **Nivel de Servicio (LOS):** Clasificación automática de la vía (A-F) según el estándar *Highway Capacity Manual 2010*.
3.  **Detección de Incidentes:**
    * 🛑 **Vehículo Detenido:** V < 0.3 m/s por > 3 seg.
    * ⚠️ **Frenada Brusca:** Aceleración < -2.5 m/s².
    * 💥 **Conflicto Espacial:** Distancia entre vehículos < 3.5 m.

---

## 🔗 Integración Blockchain BSV

Para garantizar la **auditabilidad** en contratos públicos y multas automatizadas, implementamos `bsv_blockchain.py`:

* **Hashing:** Cada 30 frames se genera un hash SHA-256 de la imagen.
* **Cadena ETL:** Cada transacción contiene el hash de la anterior, creando una cadena local inmutable.
* **Salida:** Generación de logs JSON compatibles con la red BSV (Bitcoin SV) listos para indexación (TAAL/Gorillapool).

---

## 🚀 Guía de Instalación y Despliegue

### Requisitos Previos
* Python 3.9+
* S.O.: Windows, Linux o macOS.
* Recomendado: GPU NVIDIA (CUDA) para aceleración.

### Pasos de Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/MahsaSimaei/NeuralLogic_AeroTrace](https://github.com/MahsaSimaei/NeuralLogic_AeroTrace)
    cd C:\Repositorios\NeuralLogic_AeroTrace\BLOCKCHAIN
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```

### Estructura del Repositorio
```text
/NeuralLogic_AeroTrace
/NeuralLogic_AeroTrace/BLOCKCHAIN
├── app.py                 # Punto de entrada Frontend (Streamlit)
├── main.py                # Lógica del Backend y Visión Artificial
├── bsv_blockchain.py      # Módulo de registro de evidencias
├── requirements.txt       # Dependencias
└── outputs/               # Directorio generado automáticamente
