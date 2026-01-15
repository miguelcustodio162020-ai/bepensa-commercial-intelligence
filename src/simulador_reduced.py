#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BEPENSA DOMINICANA COMMERCIAL INTELLIGENCE SIMULATOR
=====================================================
Version: 2.0 (Portfolio Production Release)
Author: [User Name]
Description:
    High-fidelity transactional data simulator for FMCG (Fast-Moving Consumer Goods).
    Generates a complete Data Warehouse (Star Schema) with:
    - 20 Dimension Tables (Product, Client, Geo, Route, etc.)
    - 12 Fact Tables (Sales, Inventory, Logistics, Finance, etc.)
    
    Architecture:
    - OOM-Safe (Out-of-Memory) design using Polars and DuckDB.
    - Vectorized generation for high performance.
    - Realistic business logic (seasonality, price elasticity, channel behavior).
"""

# # 🚀 FASE 1: ARQUITECTURA, MAESTROS Y NÚCLEO TRANSACCIONAL (OOM-SAFE)
# 
# # ====================================================================
# # CONFIGURACIÓN MAESTRA Y ENTORNO
# # ====================================================================
# 
# # Propósito: Configurar el entorno, librerías, directorios y cargar datos maestros.



import os
import sys
# Force UTF-8 encoding for stdout/stderr to avoid crashes with emojis on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import math
import random
import logging
import gc
import glob
import unicodedata
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from matplotlib.pylab import sample

# --- Librerías de Datos ---
import numpy as np
import polars as pl
from polars import expr as px
import shutil
import duckdb  # Motor OLAP para agregaciones OOM
from faker import Faker
from tqdm import tqdm
from polars import Schema
from geopy.distance import geodesic  # para calcular distancias en km


# --- Configuración de Logging ---
LOG_LEVEL = os.getenv("SIM_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SimuladorBepensa")

# --- Configuración de Directorios (Pathlib) ---
BASE_DIR = Path(r"C:\DE")
DIRS = {
    "RAW": BASE_DIR / "raw_data",
    "OUTPUT": BASE_DIR / "output",
    "PARTS": BASE_DIR / "output" / "FactVentasParticionada", # Nueva ruta particionada
    "LOGS": BASE_DIR / "logs"
}

for key, path in DIRS.items():
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Directorio verificado: {path}")

# --- Configuración Global ---
SEED_VAL = 420
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
Faker.seed(SEED_VAL)
fake = Faker('es_ES')
rng = np.random.default_rng(SEED_VAL)

# --- Parámetros del Proyecto ---
FECHA_INICIO_PROYECTO = date(2021, 1, 1)
FECHA_FIN_PROYECTO = date(2021, 12, 31) # Baseline run limited to 1 year

# --- ESQUEMAS OPTIMIZADOS (Downcasting) ---
# Definimos tipos de datos para minimizar uso de RAM
SCHEMAS = {
    "DimProducto": {
        "ID_Producto_SKU": pl.Categorical, "Nombre_Producto": pl.Utf8, "Categoria": pl.Categorical,
        "Precio_Lista_DOP": pl.Float32, "Costo_Prod_DOP": pl.Float32, "Volumen_Litros": pl.Float32,
        "Peso_Venta": pl.Float32
    },
    "DimCliente": {
        "ID_Cliente": pl.Categorical, "ID_Provincia": pl.Categorical, "ID_Canal_Segmento": pl.Categorical,
        "Cluster_ID": pl.UInt8
    },
    "FactVentas": { # Esquema para escritura parquet
        "ID_Venta_Transaccion": pl.Utf8, "ID_Factura": pl.Categorical,
        "ID_Cliente": pl.Categorical, "ID_Producto_SKU": pl.Categorical, "ID_Vendedor": pl.Categorical,
        "ID_Vehiculo": pl.Categorical, "ID_Ruta": pl.Categorical,
        "Cantidad_Unidades": pl.Int32, "Ingreso_Neto_DOP": pl.Float32, "Costo_Venta_Total_DOP": pl.Float32
    },
    "FactEmpleado": {
        "ID_Tiempo_Mes": pl.Utf8, "ID_Empleado": pl.Categorical,
        "Salario_Base_Mensual_DOP": pl.Float32, "Monto_Comisiones_DOP": pl.Float32,
        "Monto_Bonos_DOP": pl.Float32, "Monto_Deducciones_DOP": pl.Float32,
        "Salario_Neto_DOP": pl.Float32, "Horas_Normales_Trabajadas": pl.Int16,
        "Horas_Extras_Trabajadas": pl.Int16, "Puntuacion_Desempeno_Mes": pl.UInt8
    },
    "FactContabilidad": {
        "ID_Asiento": pl.Utf8, "ID_Tiempo_Contable": pl.Date, "Tipo_Transaccion": pl.Categorical,
        "Modulo_Origen": pl.Categorical, "ID_Documento_Origen": pl.Utf8,
        "Cuenta_Contable": pl.Categorical, "Descripcion_Asiento": pl.Utf8,
        "Monto_Debito_DOP": pl.Float32, "Monto_Credito_DOP": pl.Float32, "Centro_Costo": pl.Categorical
    }
}

# --- Helpers ---
def guardar_parquet(df: pl.DataFrame, nombre_archivo: str):
    """Guarda DataFrame en formato Parquet estándar."""
    try:
        ruta = DIRS["OUTPUT"] / f"{nombre_archivo}.parquet"
        df.write_parquet(ruta, compression="zstd")
        logger.info(f"💾 Archivo guardado: {ruta} ({df.height:,} filas)")
    except Exception as e:
        logger.error(f"❌ Error guardando {nombre_archivo}: {e}")
        raise

logger.info("🚀 Entorno OOM-Safe inicializado.")


# --- Helpers de Validación ---
def validar_pesos(lista: List[Dict], llave_peso: str, nombre_entidad: str, tolerancia: float = 0.02):
    """Valida que la suma de pesos de una lista de diccionarios sea aprox 1.0."""
    total = sum(item.get(llave_peso, 0) for item in lista)
    if not math.isclose(total, 1.0, abs_tol=tolerancia):
        logger.warning(f"⚠️ [Integridad] Pesos de '{nombre_entidad}' suman {total:.4f}, se esperaba 1.0. (Tol: {tolerancia})")
    else:
        logger.info(f"✅ [Integridad] Pesos de '{nombre_entidad}' validados correctamente.")


# # ====================================================================
# # ====================================================================



# 📊 BLOQUE 0. Metas Anuales (Target de Ventas) y Facturas Estimada
# ============================================================
# 0. Metas Anuales (Target de Ventas) y Facturas Estimadas

logger.info("✅ Maestros cargados en memoria con estimaciones completas.")

#=========================================================================================================
# 7. Costos de Promociones (Estimado)
# Presupuesto anual de marketing estimado como % de las ventas brutas (aprox 5%).
COSTO_PROMOCIONES_ANUAL_ESTIMADO = {
    2021: 150_000_000, 2022: 170_000_000, 2023: 180_000_000,
    2024: 200_000_000, 2025: 220_000_000
}

# Volumen de cajas (MCU) estimado por año
VOLUMEN_CAJAS_MCU_POR_ANO: Dict[int, int] = {
    2021: 66_490_000,
    2022: 72_800_000,
    2023: 72_800_000,
    2024: 80_040_000,
    2025: 90_000_000
}
VOLUMEN_CAJAS_MCU_TOTAL: int = sum(VOLUMEN_CAJAS_MCU_POR_ANO.values())

def normalizar_pesos(lista: list[dict], columna: str = "Peso", columna_salida: str = "Peso_Normalizado") -> None:
    """
    Normaliza los pesos sobre una lista de diccionarios IN PLACE.
    Modifica la lista agregando la columna de pesos normalizados.
    Solo loguea si ocurre un error o la suma se desvía de 1.
    """
    suma_total = sum(item.get(columna, 0) for item in lista)
    if suma_total == 0:
        logger.error("❌ No se puede normalizar: la suma total de pesos es 0 para la columna '%s'.", columna)
        raise ValueError("No se puede normalizar: la suma total de pesos es 0.")

    if not math.isclose(suma_total, 1.0, abs_tol=0.03):
        logger.warning("⚠️ Suma de pesos = %.6f (column '%s'), se normaliza forzadamente.", suma_total, columna)
    
    # Normalización
    for item in lista:
        item[columna_salida] = round(item[columna] / suma_total, 6)




# BLOQUE 1. PROVINCIAS_MAESTRA
# ==============================

# Estimación basada en población y actividad económica de RD.
# Estructura: (Provincia, Zona, Peso, Lat, Lon)
PROVINCIAS_MAESTRA = [
    # Zona Metropolitana (Ozama) - ~51% del volumen
    {"ID_Provincia": "SANTO01", "Provincia": "Santo Domingo", "Region": "Ozama", "Peso": 0.29, "Lat": 18.48, "Lon": -69.94},
    {"ID_Provincia": "DISTR02", "Provincia": "Distrito Nacional", "Region": "Ozama", "Peso": 0.171, "Lat": 18.46, "Lon": -69.95},
    {"ID_Provincia": "MONT03", "Provincia": "Monte Plata", "Region": "Ozama", "Peso": 0.008, "Lat": 18.80, "Lon": -69.80},

    # Cibao Norte - ~15%
    {"ID_Provincia": "SANTI04", "Provincia": "Santiago", "Region": "Cibao Norte", "Peso": 0.09, "Lat": 19.45, "Lon": -70.70},
    {"ID_Provincia": "PUERTO05", "Provincia": "Puerto Plata", "Region": "Cibao Norte", "Peso": 0.02, "Lat": 19.79, "Lon": -70.68},
    {"ID_Provincia": "ESPAI06", "Provincia": "Espaillat", "Region": "Cibao Norte", "Peso": 0.001, "Lat": 19.50, "Lon": -70.30},
    
    # Yuma (Este) - ~16%
    {"ID_Provincia": "LAALTA07", "Provincia": "La Altagracia", "Region": "Yuma", "Peso": 0.02, "Lat": 18.60, "Lon": -68.58},
    {"ID_Provincia": "LAROMA08", "Provincia": "La Romana", "Region": "Yuma", "Peso": 0.06, "Lat": 18.42, "Lon": -68.97},
    {"ID_Provincia": "ELSEIB09", "Provincia": "El Seibo", "Region": "Yuma", "Peso": 0.005, "Lat": 18.75, "Lon": -69.03},

    # Valdesia (Sur Cercano) - ~6%
    {"ID_Provincia": "SANCRI10", "Provincia": "San Cristóbal", "Region": "Valdesia", "Peso": 0.008, "Lat": 18.41, "Lon": -70.10},
    {"ID_Provincia": "PERAVI11", "Provincia": "Peravia", "Region": "Valdesia", "Peso": 0.05, "Lat": 18.28, "Lon": -70.36},
    {"ID_Provincia": "AZUA12", "Provincia": "Azua", "Region": "Valdesia", "Peso": 0.009, "Lat": 18.32, "Lon": -70.73},
    {"ID_Provincia": "SJOCOA13", "Provincia": "San José de Ocoa", "Region": "Valdesia", "Peso": 0.005, "Lat": 18.55, "Lon": -70.50},

    # Cibao Nordeste - ~6%
    {"ID_Provincia":"SAMANA14", "Provincia": "Samana", "Region": "Cibao Nordeste", "Peso": 0.006, "Lat": 19.20, "Lon": -69.33},
    {"ID_Provincia":"DUART15", "Provincia": "Duarte", "Region": "Cibao Nordeste", "Peso": 0.003, "Lat": 19.20, "Lon": -70.20},
    {"ID_Provincia":"MTRINS16", "Provincia": "María Trinidad Sánchez", "Region": "Cibao Nordeste", "Peso": 0.005, "Lat": 19.45, "Lon": -69.95},
    {"ID_Provincia":"HERMIR17", "Provincia": "Hermanas Mirabal", "Region": "Cibao Nordeste", "Peso": 0.005, "Lat": 19.33, "Lon": -70.30},

    # Resto (Higuamo, Cibao Sur, Enriquillo, El Valle, Cibao Noroeste) - Pesos menores
    {"ID_Provincia":"SANPED18", "Provincia": "San Pedro de Macorís", "Region": "Higuamo", "Peso": 0.06, "Lat": 18.46, "Lon": -69.30},
    {"ID_Provincia":"LAVEGA19", "Provincia": "La Vega", "Region": "Cibao Sur", "Peso": 0.007, "Lat": 19.22, "Lon": -70.52},
    {"ID_Provincia":"MONNOU20", "Provincia": "Monseñor Nouel", "Region": "Cibao Sur", "Peso": 0.010, "Lat": 18.93, "Lon": -70.40},
    {"ID_Provincia":"HATOMAY21", "Provincia": "Hato Mayor", "Region": "Higuamo", "Peso": 0.007, "Lat": 18.75, "Lon": -69.38},
    {"ID_Provincia":"SANJUA22", "Provincia": "San Juan", "Region": "El Valle", "Peso": 0.050, "Lat": 18.80, "Lon": -71.22},
    {"ID_Provincia":"BARAHO23", "Provincia": "Barahona", "Region": "Enriquillo", "Peso": 0.060, "Lat": 18.20, "Lon": -71.10},
    {"ID_Provincia":"VALVER24", "Provincia": "Valverde", "Region": "Cibao Noroeste", "Peso": 0.047, "Lat": 19.55, "Lon": -71.00},
    {"ID_Provincia":"SANRAM25", "Provincia": "Sánchez Ramírez", "Region": "Cibao Sur", "Peso": 0.005, "Lat": 19.10, "Lon": -70.15},
    {"ID_Provincia":"MONCRI26", "Provincia": "Monte Cristi", "Region": "Cibao Noroeste", "Peso": 0.003, "Lat": 19.80, "Lon": -71.65},
    {"ID_Provincia":"SANTRO27", "Provincia": "Santiago Rodríguez", "Region": "Cibao Noroeste", "Peso": 0.003, "Lat": 19.45, "Lon": -71.35},
    {"ID_Provincia":"DAJABO28", "Provincia": "Dajabón", "Region": "Cibao Noroeste", "Peso": 0.001, "Lat": 19.55, "Lon": -71.70},
    {"ID_Provincia":"BAHORU29", "Provincia": "Bahoruco", "Region": "Enriquillo", "Peso": 0.001, "Lat": 18.47, "Lon": -71.40},
    {"ID_Provincia":"INDEPE30", "Provincia": "Independencia", "Region": "Enriquillo", "Peso": 0.001, "Lat": 18.30, "Lon": -71.60},
    {"ID_Provincia":"PEDERN31", "Provincia": "Pedernales", "Region": "Enriquillo", "Peso": 0.001, "Lat": 17.98, "Lon": -71.74},
    {"ID_Provincia":"ELIPIN32", "Provincia": "Elías Piña", "Region": "El Valle", "Peso": 0.001, "Lat": 18.90, "Lon": -71.45}
]

ids = [prov["ID_Provincia"] for prov in PROVINCIAS_MAESTRA]
if len(ids) == len(set(ids)):
    logger.info("✅ Todos los IDs de provincia son únicos.")
else:
    logger.warning("⚠️ ¡Hay IDs duplicados en las provincias!")

###################################################################
# 2) Normalizar
normalizar_pesos(PROVINCIAS_MAESTRA, "Peso", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
validar_pesos(PROVINCIAS_MAESTRA, "Peso_Normalizado", "DimGeografia — Peso normalizado")




# BLOQUE 2.  PRODUCTOS_MAESTRA
# ============================================================
# 2. DimProducto (Catálogo SKU con Precios y Costos Estimados)
# Precios y costos estimados en DOP basados en mercado local y márgenes típicos de la industria.
# Estructura: (Codigo_ProductoSKU, Nombre_Producto, Categoria, Precio_Venta,Peso, Costo_Prod_Unid_DOP, Volumen_Litros, Tipo_Envase, Unidades_Por_Caja, Sabor, Marca)
PRODUCTOS_MAESTRA = [
    {"Codigo_Producto_SKU": "REF-CC-001", "Nombre_Producto": "Coca Cola 2L", "Categoria": "Refrescos", "Precio_Lista_DOP": 90.0, "Peso_Venta": 0.01133011, "Costo_Prod_DOP": 16.25, "Volumen_Litros": 2.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Cola", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_1"},
    {"Codigo_Producto_SKU": "REF-CC-002", "Nombre_Producto": "Coca Cola 1.25L", "Categoria": "Refrescos", "Precio_Lista_DOP": 65.0, "Peso_Venta": 0.02732626, "Costo_Prod_DOP": 13.15, "Volumen_Litros": 1.25, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Cola", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_2"},
    {"Codigo_Producto_SKU": "REF-CC-003", "Nombre_Producto": "Coca Cola 0.5L", "Categoria": "Refrescos", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.02216256, "Costo_Prod_DOP": 6.05, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Cola", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_3"},
    {"Codigo_Producto_SKU": "REF-CC-004", "Nombre_Producto": "Coca Cola Sin Azúcar 2L", "Categoria": "Refrescos", "Precio_Lista_DOP": 95.0, "Peso_Venta": 0.02118272, "Costo_Prod_DOP": 16.46, "Volumen_Litros": 2.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Cola Sin Azúcar", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_4"},
    {"Codigo_Producto_SKU": "REF-CC-005", "Nombre_Producto": "Coca Cola Light 2L", "Categoria": "Refrescos", "Precio_Lista_DOP": 95.0, "Peso_Venta": 0.01601042, "Costo_Prod_DOP": 16.00, "Volumen_Litros": 2.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Cola Light", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_5"},
    {"Codigo_Producto_SKU": "REF-CC-006", "Nombre_Producto": "Coca Cola Lata 355ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.01116204, "Costo_Prod_DOP": 6.25, "Volumen_Litros": 0.355, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Cola", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_6"},
    {"Codigo_Producto_SKU": "REF-CC-007", "Nombre_Producto": "Coca Cola 3L", "Categoria": "Refrescos", "Precio_Lista_DOP": 120.0, "Peso_Venta": 0.02664184, "Costo_Prod_DOP": 22.00, "Volumen_Litros": 3.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 4, "Sabor": "Cola", "Marca": "Coca-Cola", "Enlace_Web_Imagen": "Imagen_7"},
    {"Codigo_Producto_SKU": "REF-SP-001", "Nombre_Producto": "Sprite 2L", "Categoria": "Refrescos", "Precio_Lista_DOP": 95.0, "Peso_Venta": 0.02008788, "Costo_Prod_DOP": 17.5, "Volumen_Litros": 2.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Limón-Lima", "Marca": "Sprite", "Enlace_Web_Imagen": "Imagen_8"},
    {"Codigo_Producto_SKU": "REF-SP-002", "Nombre_Producto": "Sprite 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 33.0, "Peso_Venta": 0.0216202, "Costo_Prod_DOP": 5.5, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Limón-Lima", "Marca": "Sprite", "Enlace_Web_Imagen": "Imagen_9"},
    {"Codigo_Producto_SKU": "REF-SP-003", "Nombre_Producto": "Sprite Cero 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.01418185, "Costo_Prod_DOP": 6.0, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Limón-Lima Sin Azúcar", "Marca": "Sprite", "Enlace_Web_Imagen": "Imagen_10"},
    {"Codigo_Producto_SKU": "REF-FA-001", "Nombre_Producto": "Fanta Naranja 2L", "Categoria": "Refrescos", "Precio_Lista_DOP": 80.0, "Peso_Venta": 0.01081285, "Costo_Prod_DOP": 15.0, "Volumen_Litros": 2.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Naranja", "Marca": "Fanta", "Enlace_Web_Imagen": "Imagen_11"},
    {"Codigo_Producto_SKU": "REF-FA-002", "Nombre_Producto": "Fanta Naranja 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 30.0, "Peso_Venta": 0.01364829, "Costo_Prod_DOP": 5.25, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Naranja", "Marca": "Fanta", "Enlace_Web_Imagen": "Imagen_12"},
    {"Codigo_Producto_SKU": "REF-FA-003", "Nombre_Producto": "Fanta Uva 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 31.0, "Peso_Venta": 0.01124713, "Costo_Prod_DOP": 5.30, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Uva", "Marca": "Fanta", "Enlace_Web_Imagen": "Imagen_13"},
    {"Codigo_Producto_SKU": "REF-FA-004", "Nombre_Producto": "Fanta Piña 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 32.0, "Peso_Venta": 0.02655622, "Costo_Prod_DOP": 5.50, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Piña", "Marca": "Fanta", "Enlace_Web_Imagen": "Imagen_14"},
    {"Codigo_Producto_SKU": "REF-MM-001", "Nombre_Producto": "Mundet Manzana 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 33.0, "Peso_Venta": 0.02517977, "Costo_Prod_DOP": 5.75, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Manzana", "Marca": "Mundet", "Enlace_Web_Imagen": "Imagen_15"},
    {"Codigo_Producto_SKU": "REF-CC-008", "Nombre_Producto": "Country Club Fresa 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 32.0, "Peso_Venta": 0.01273933, "Costo_Prod_DOP": 5.65, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Fresa", "Marca": "Country Club", "Enlace_Web_Imagen": "Imagen_16"},
    {"Codigo_Producto_SKU": "REF-CC-009", "Nombre_Producto": "Country Club Uva 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 32.0, "Peso_Venta": 0.02281339, "Costo_Prod_DOP": 5.65, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Uva", "Marca": "Country Club", "Enlace_Web_Imagen": "Imagen_17"},
    {"Codigo_Producto_SKU": "REF-CC-010", "Nombre_Producto": "Country Club Menta 500ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 32.0, "Peso_Venta": 0.01248171, "Costo_Prod_DOP": 5.65, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Menta", "Marca": "Country Club", "Enlace_Web_Imagen": "Imagen_18"},
    {"Codigo_Producto_SKU": "ISO-PW-001", "Nombre_Producto": "Powerade Azul 500ml", "Categoria": "Isotónicos", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.02014838, "Costo_Prod_DOP": 5.90, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Azul", "Marca": "Powerade", "Enlace_Web_Imagen": "Imagen_19"},
    {"Codigo_Producto_SKU": "ISO-PW-002", "Nombre_Producto": "Powerade Roja 500ml", "Categoria": "Isotónicos", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.02499986, "Costo_Prod_DOP": 5.90, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Rojo", "Marca": "Powerade", "Enlace_Web_Imagen": "Imagen_20"},
    {"Codigo_Producto_SKU": "ISO-PW-003", "Nombre_Producto": "Powerade Verde 500ml", "Categoria": "Isotónicos", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.01688083, "Costo_Prod_DOP": 5.90, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Verde", "Marca": "Powerade", "Enlace_Web_Imagen": "Imagen_21"},
    {"Codigo_Producto_SKU": "AGU-DS-001", "Nombre_Producto": "Dasani 1.5L", "Categoria": "Agua", "Precio_Lista_DOP": 55.0, "Peso_Venta": 0.01358217, "Costo_Prod_DOP": 3.40, "Volumen_Litros": 1.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Natural", "Marca": "Dasani", "Enlace_Web_Imagen": "Imagen_22"},
    {"Codigo_Producto_SKU": "AGU-DS-002", "Nombre_Producto": "Dasani 600ml", "Categoria": "Agua", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.0126829, "Costo_Prod_DOP": 1.80, "Volumen_Litros": 0.591, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Natural", "Marca": "Dasani", "Enlace_Web_Imagen": "Imagen_23"},
    {"Codigo_Producto_SKU": "AGU-DS-003", "Nombre_Producto": "Dasani Garrafón 20L", "Categoria": "Agua", "Precio_Lista_DOP": 225.0, "Peso_Venta": 0.01551624, "Costo_Prod_DOP": 18.0, "Volumen_Litros": 20.0, "Tipo_Envase": "HDPE", "Unidades_Por_Caja": 1, "Sabor": "Natural", "Marca": "Dasani", "Enlace_Web_Imagen": "Imagen_24"},
    {"Codigo_Producto_SKU": "AGU-DS-004", "Nombre_Producto": "Dasani Kids 250ml", "Categoria": "Agua", "Precio_Lista_DOP": 18.0, "Peso_Venta": 0.02562369, "Costo_Prod_DOP": 0.95, "Volumen_Litros": 0.25, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Natural", "Marca": "Dasani", "Enlace_Web_Imagen": "Imagen_25"},
    {"Codigo_Producto_SKU": "AGU-DS-005", "Nombre_Producto": "Agua Dasani Saborizada Fresa", "Categoria": "Agua", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.01868844, "Costo_Prod_DOP": 1.90, "Volumen_Litros": 0.591, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Fresa", "Marca": "Dasani", "Enlace_Web_Imagen": "Imagen_26"},
    {"Codigo_Producto_SKU": "ENR-BN-001", "Nombre_Producto": "Burn Energizante 1.25L", "Categoria": "Energizantes", "Precio_Lista_DOP": 150.0, "Peso_Venta": 0.02444934, "Costo_Prod_DOP": 13.0, "Volumen_Litros": 1.25, "Tipo_Envase": "PET", "Unidades_Por_Caja": 12, "Sabor": "Energy", "Marca": "Burn", "Enlace_Web_Imagen": "Imagen_27"},
    {"Codigo_Producto_SKU": "NRG-MN-001", "Nombre_Producto": "Monster Original 500ml", "Categoria": "Energizantes", "Precio_Lista_DOP": 150.0, "Peso_Venta": 0.0132488, "Costo_Prod_DOP": 9.0, "Volumen_Litros": 0.5, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Original", "Marca": "Monster", "Enlace_Web_Imagen": "Imagen_28"},
    {"Codigo_Producto_SKU": "NRG-MN-002", "Nombre_Producto": "Monster Ultra 500ml", "Categoria": "Energizantes", "Precio_Lista_DOP": 150.0, "Peso_Venta": 0.02728018, "Costo_Prod_DOP": 9.0, "Volumen_Litros": 0.5, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Ultra", "Marca": "Monster", "Enlace_Web_Imagen": "Imagen_29"},
    {"Codigo_Producto_SKU": "NRG-MN-003", "Nombre_Producto": "Monster Mango 500ml", "Categoria": "Energizantes", "Precio_Lista_DOP": 150.0, "Peso_Venta": 0.01195057, "Costo_Prod_DOP": 9.0, "Volumen_Litros": 0.5, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Mango", "Marca": "Monster", "Enlace_Web_Imagen": "Imagen_30"},
    {"Codigo_Producto_SKU": "NRG-MN-004", "Nombre_Producto": "Monster Green 500ml", "Categoria": "Energizantes", "Precio_Lista_DOP": 150.0, "Peso_Venta": 0.02797871, "Costo_Prod_DOP": 9.0, "Volumen_Litros": 0.5, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Green", "Marca": "Monster", "Enlace_Web_Imagen": "Imagen_31"},
    {"Codigo_Producto_SKU": "JUG-DV-002", "Nombre_Producto": "Del Valle Manzana 1L", "Categoria": "Jugos", "Precio_Lista_DOP": 70.0, "Peso_Venta": 0.01130713, "Costo_Prod_DOP": 6.75, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Manzana", "Marca": "Del Valle", "Enlace_Web_Imagen": "Imagen_32"},
    {"Codigo_Producto_SKU": "JUG-DV-003", "Nombre_Producto": "Del Valle Durazno 1L", "Categoria": "Jugos", "Precio_Lista_DOP": 70.0, "Peso_Venta": 0.01822007, "Costo_Prod_DOP": 6.75, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Durazno", "Marca": "Del Valle", "Enlace_Web_Imagen": "Imagen_33"},
    {"Codigo_Producto_SKU": "JUG-DV-004", "Nombre_Producto": "Del Valle Mango 1L", "Categoria": "Jugos", "Precio_Lista_DOP": 70.0, "Peso_Venta": 0.02492681, "Costo_Prod_DOP": 6.75, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Mango", "Marca": "Del Valle", "Enlace_Web_Imagen": "Imagen_34"},
    {"Codigo_Producto_SKU": "JUG-DV-005", "Nombre_Producto": "Del Valle Naranja 1L", "Categoria": "Jugos", "Precio_Lista_DOP": 70.0, "Peso_Venta": 0.01404963, "Costo_Prod_DOP": 6.75, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Naranja", "Marca": "Del Valle", "Enlace_Web_Imagen": "Imagen_35"},
    {"Codigo_Producto_SKU": "JUG-DV-006", "Nombre_Producto": "Del Valle Mini Pack 200ml", "Categoria": "Jugos", "Precio_Lista_DOP": 19.0, "Peso_Venta": 0.02281335, "Costo_Prod_DOP": 2.30, "Volumen_Litros": 0.2, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 6, "Sabor": "Multi", "Marca": "Del Valle", "Enlace_Web_Imagen": "Imagen_36"},
    {"Codigo_Producto_SKU": "LAC-SC-001", "Nombre_Producto": "Santa Clara Entera 1L", "Categoria": "Lácteos", "Precio_Lista_DOP": 60.0, "Peso_Venta": 0.01158699, "Costo_Prod_DOP": 6.50, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Entera", "Marca": "Santa Clara", "Enlace_Web_Imagen": "Imagen_37"},
    {"Codigo_Producto_SKU": "LAC-SC-002", "Nombre_Producto": "Santa Clara Deslactosada 1L", "Categoria": "Lácteos", "Precio_Lista_DOP": 65.0, "Peso_Venta": 0.01416766, "Costo_Prod_DOP": 6.80, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Deslactosada", "Marca": "Santa Clara", "Enlace_Web_Imagen": "Imagen_38"},
    {"Codigo_Producto_SKU": "LAC-SC-003", "Nombre_Producto": "Santa Clara Mini 200ml", "Categoria": "Lácteos", "Precio_Lista_DOP": 15.0, "Peso_Venta": 0.01532279, "Costo_Prod_DOP": 2.00, "Volumen_Litros": 0.2, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Mini", "Marca": "Santa Clara", "Enlace_Web_Imagen": "Imagen_39"},
    {"Codigo_Producto_SKU": "AGU-AQ-001", "Nombre_Producto": "Aquarius Naranja 500ml", "Categoria": "Agua", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.0089225, "Costo_Prod_DOP": 1.75, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Naranja", "Marca": "Aquarius", "Enlace_Web_Imagen": "Imagen_40"},
    {"Codigo_Producto_SKU": "AGU-AQ-002", "Nombre_Producto": "Aquarius Limón 500ml", "Categoria": "Agua", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.01968623, "Costo_Prod_DOP": 1.75, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Limón", "Marca": "Aquarius", "Enlace_Web_Imagen": "Imagen_41"},
    {"Codigo_Producto_SKU": "RTD-VEG-001", "Nombre_Producto": "AdeS Soya Original 1L", "Categoria": "Bebida Vegetal RTD", "Precio_Lista_DOP": 105.0, "Peso_Venta": 0.01438655, "Costo_Prod_DOP": 9.5, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Soya Original", "Marca": "AdeS", "Enlace_Web_Imagen": "Imagen_42"},
    {"Codigo_Producto_SKU": "RTD-VEG-002", "Nombre_Producto": "AdeS Soya Chocolate 1L", "Categoria": "Bebida Vegetal RTD", "Precio_Lista_DOP": 105.0, "Peso_Venta": 0.01234501, "Costo_Prod_DOP": 9.5, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Soya Chocolate", "Marca": "AdeS", "Enlace_Web_Imagen": "Imagen_43"},
    {"Codigo_Producto_SKU": "RTD-VEG-003", "Nombre_Producto": "AdeS Almendra 1L", "Categoria": "Bebida Vegetal RTD", "Precio_Lista_DOP": 110.0, "Peso_Venta": 0.02352518, "Costo_Prod_DOP": 9.84, "Volumen_Litros": 1.0, "Tipo_Envase": "Tetra", "Unidades_Por_Caja": 12, "Sabor": "Almendra", "Marca": "AdeS", "Enlace_Web_Imagen": "Imagen_44"},
    {"Codigo_Producto_SKU": "MAL-MM-001", "Nombre_Producto": "Malta Morena 355ml", "Categoria": "Malta", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.01283459, "Costo_Prod_DOP": 2.64, "Volumen_Litros": 0.355, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Morena", "Marca": "Malta Morena", "Enlace_Web_Imagen": "Imagen_45"},
    {"Codigo_Producto_SKU": "MAL-MM-002", "Nombre_Producto": "Malta Morena 1L", "Categoria": "Malta", "Precio_Lista_DOP": 75.0, "Peso_Venta": 0.01082049, "Costo_Prod_DOP": 5.76, "Volumen_Litros": 1.0, "Tipo_Envase": "PET", "Unidades_Por_Caja": 12, "Sabor": "Morena", "Marca": "Malta Morena", "Enlace_Web_Imagen": "Imagen_46"},
    {"Codigo_Producto_SKU": "ISO-SO-001", "Nombre_Producto": "Suero Oral Uva 500ml", "Categoria": "Isotónicos", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.01280359, "Costo_Prod_DOP": 3.30, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Uva", "Marca": "Suero Oral", "Enlace_Web_Imagen": "Imagen_47"},
    {"Codigo_Producto_SKU": "ISO-SO-002", "Nombre_Producto": "Suero Oral Naranja 500ml", "Categoria": "Isotónicos", "Precio_Lista_DOP": 35.0, "Peso_Venta": 0.02366913, "Costo_Prod_DOP": 3.30, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 24, "Sabor": "Naranja", "Marca": "Suero Oral", "Enlace_Web_Imagen": "Imagen_48"},
    {"Codigo_Producto_SKU": "AGU-KA-001", "Nombre_Producto": "Kinley Agua Tónica 250ml", "Categoria": "Agua", "Precio_Lista_DOP": 19.0, "Peso_Venta": 0.02503783, "Costo_Prod_DOP": 1.10, "Volumen_Litros": 0.25, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Tónica", "Marca": "Kinley", "Enlace_Web_Imagen": "Imagen_49"},
    {"Codigo_Producto_SKU": "REF-SCH-001", "Nombre_Producto": "Schweppes Ginger Ale 355ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.02399029, "Costo_Prod_DOP": 6.00, "Volumen_Litros": 0.355, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Ginger Ale", "Marca": "Schweppes", "Enlace_Web_Imagen": "Imagen_50"},
    {"Codigo_Producto_SKU": "REF-SCH-002", "Nombre_Producto": "Schweppes Soda 355ml", "Categoria": "Refrescos", "Precio_Lista_DOP": 40.0, "Peso_Venta": 0.02304446, "Costo_Prod_DOP": 6.00, "Volumen_Litros": 0.355, "Tipo_Envase": "Lata", "Unidades_Por_Caja": 24, "Sabor": "Soda", "Marca": "Schweppes", "Enlace_Web_Imagen": "Imagen_51"},
    {"Codigo_Producto_SKU": "TEA-FZ-001", "Nombre_Producto": "Fuze Tea Limón 500ml", "Categoria": "Té", "Precio_Lista_DOP": 65.0, "Peso_Venta": 0.01305472, "Costo_Prod_DOP": 2.66, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Limón", "Marca": "Fuze Tea", "Enlace_Web_Imagen": "Imagen_52"},
    {"Codigo_Producto_SKU": "TEA-FZ-002", "Nombre_Producto": "Fuze Tea Durazno 500ml", "Categoria": "Té", "Precio_Lista_DOP": 65.0, "Peso_Venta": 0.0180872, "Costo_Prod_DOP": 2.66, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Durazno", "Marca": "Fuze Tea", "Enlace_Web_Imagen": "Imagen_53"},
    {"Codigo_Producto_SKU": "TEA-FZ-003", "Nombre_Producto": "Fuze Tea Frambuesa 500ml", "Categoria": "Té", "Precio_Lista_DOP": 65.0, "Peso_Venta": 0.02053763, "Costo_Prod_DOP": 2.66, "Volumen_Litros": 0.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Frambuesa", "Marca": "Fuze Tea", "Enlace_Web_Imagen": "Imagen_54"},
    {"Codigo_Producto_SKU": "AGU-CI-001", "Nombre_Producto": "Ciel Agua 1.5L", "Categoria": "Agua", "Precio_Lista_DOP": 45.0, "Peso_Venta": 0.02464545, "Costo_Prod_DOP": 2.20, "Volumen_Litros": 1.5, "Tipo_Envase": "PET", "Unidades_Por_Caja": 6, "Sabor": "Natural", "Marca": "Ciel", "Enlace_Web_Imagen": "Imagen_55"}
]

ids = [prov["Codigo_Producto_SKU"] for prov in PRODUCTOS_MAESTRA]
if len(ids) == len(set(ids)):
    logger.info("✅ Todos los IDs de producto son únicos.")
else:
    logger.warning("⚠️ ¡Hay IDs duplicados en los productos!")

#################################################################
normalizar_pesos(PRODUCTOS_MAESTRA, "Peso_Venta", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
validar_pesos(PRODUCTOS_MAESTRA, "Peso_Normalizado", "DimPoducto — Peso normalizado")




# BLOQUE 3. CANALES_MAESTRA y sus pesos (Si no estaba en el txt, lo definimos aquí para integridad)
# 3. Canales
# Distribución estimada del mercado de bebidas en RD
CANALES_MAESTRA = [
    {"Canal_Venta": "Colmado", "Peso": 0.55},
    {"Canal_Venta": "Supermercado", "Peso": 0.25},
    {"Canal_Venta": "Tienda", "Peso": 0.05},
    {"Canal_Venta": "Mayorista", "Peso": 0.08},
    {"Canal_Venta": "Farmacia", "Peso": 0.02},
    {"Canal_Venta": "Horeca (Hotel/Rest/Café)", "Peso": 0.12},
    {"Canal_Venta": "Institucional", "Peso": 0.02},
    {"Canal_Venta": "Venta Directa", "Peso": 0.01}
]

# 2) Normalizar
normalizar_pesos(CANALES_MAESTRA, "Peso", "Peso_Normalizado")

# 3) Validar después de normalizar (opcional)
validar_pesos(CANALES_MAESTRA, "Peso_Normalizado", "DimCanales — Peso normalizado")

SEGMENTOS_CLIENTES = ["A", "B", "C+", "C-", "D", "E"]
PESO_SEGMENTACION_CANAL = {
    "Supermercado": {"A": 0.4, "B": 0.3, "C+": 0.2, "C-": 0.05, "D": 0.05, "E": 0.0},
    "Colmado": {"A": 0.05, "B": 0.2, "C+": 0.3, "C-": 0.25, "D": 0.15, "E": 0.05},
    "Tienda": {"A": 0.1, "B": 0.2, "C+": 0.3, "C-": 0.2, "D": 0.1, "E": 0.1},
    "Mayorista": {"A": 0.15, "B": 0.2, "C+": 0.25, "C-": 0.15, "D": 0.15, "E": 0.1},
    "Farmacia": {"A": 0.3, "B": 0.4, "C+": 0.2, "C-": 0.05, "D": 0.05, "E": 0.0},
    "Horeca (Hotel/Rest/Café)": {"A": 0.4, "B": 0.3, "C+": 0.2, "C-": 0.1, "D": 0.0, "E": 0.0},
    "Institucional": {"A": 0.5, "B": 0.3, "C+": 0.1, "C-": 0.05, "D": 0.05, "E": 0.0},
    "Venta Directa": {"A": 0.0, "B": 0.1, "C+": 0.2, "C-": 0.3, "D": 0.2, "E": 0.2},
}




# BLOQUE 4 .  FLOTA_VEHICULOS 
# =================================================================================================

# Flota estimada para cubrir la demanda nacional.
FLOTA_VEHICULOS = {
    "Hyundai HD-78": {
        "Tipo_Logistico": "Camion_Interurbano",
        "Capacidad_ton_min": 4.0,
        "Capacidad_ton_max": 5.0,
        "Capacidad_Ton": 5.5,
        "KM_anual_min": 35000,
        "KM_anual_max": 50000,
        "Rutas": "Santo Domingo, Santiago, Este, Sur",
        "Combustible": "Diesel",
        "Total": 62,
        "Rendimiento_KMLitro": 4.5,
        "Costo_Fijo_Diario_DOP": 600.00,
        "Segmento_Operacion": "Interurbana"
    },
    "Fuso Canter FE85": {
        "Tipo_Logistico": "Camion_Interurbano",
        "Capacidad_ton_min": 3.0,
        "Capacidad_ton_max": 4.0,
        "Capacidad_Ton": 5.5,
        "KM_anual_min": 45000,
        "KM_anual_max": 60000,
        "Rutas": "Zonas urbanas y semiurbanas",
        "Combustible": "Diesel",
        "Total": 61,
        "Rendimiento_KMLitro": 4.5,
        "Costo_Fijo_Diario_DOP": 600.00,
        "Segmento_Operacion": "Interurbana"
    },
    "Isuzu NPR": {
        "Tipo_Logistico": "Furgoneta_Urbana",
        "Capacidad_ton_min": 2.5,
        "Capacidad_ton_max": 3.5,
        "Capacidad_Ton": 2.5,
        "KM_anual_min": 50000,
        "KM_anual_max": 70000,
        "Rutas": "Rutas cortas de alta densidad",
        "Combustible": "Diesel",
        "Total": 64,
        "Rendimiento_KMLitro": 5.5,
        "Costo_Fijo_Diario_DOP": 400.00,
        "Segmento_Operacion": "Urbana"
    },
    "Panel H-100": {
        "Tipo_Logistico": "Vehiculo_Local",
        "Capacidad_ton_min": 1.0,
        "Capacidad_ton_max": 1.5,
        "Capacidad_Ton": 1.5,
        "KM_anual_min": 60000,
        "KM_anual_max": 80000,
        "Rutas": "Colmados y rutas difíciles",
        "Combustible": "Gasolina",
        "Total": 7,
        "Rendimiento_KMLitro": 6.5,
        "Costo_Fijo_Diario_DOP": 300.00,
        "Segmento_Operacion": "Local"
    },
    "Freightliner M2": {
        "Tipo_Logistico": "Camion_Regional",
        "Capacidad_ton_min": 9.0,
        "Capacidad_ton_max": 10.0,
        "Capacidad_Ton": 10.0,
        "KM_anual_min": 45000,
        "KM_anual_max": 70000,
        "Rutas": "Grandes rutas nacionales",
        "Combustible": "Diesel",
        "Total": 111,
        "Rendimiento_KMLitro": 3.5,
        "Costo_Fijo_Diario_DOP": 850.00,
        "Segmento_Operacion": "Regional"
    }
}




# BLOQUE 5.  DEPARTAMENTOS_RRHH
# ============================================================================

# 5. RRHH (Estructura)
# Estructura salarial estimada para roles operativos y comerciales.
DEPARTAMENTOS_RRHH = {
    "Logistica_Distribucion": [
        {"Puesto": "Chofer de reparto", "Sueldo_Min": 28500, "Sueldo_Max": 34500, "Cantidad": 330},
        {"Puesto": "Ayudante de reparto", "Sueldo_Min": 25500, "Sueldo_Max": 29500, "Cantidad": 380},
        {"Puesto": "Operador de Montacargas", "Sueldo_Min": 32500, "Sueldo_Max": 43000, "Cantidad": 40},
        {"Puesto": "Coordinador Logistico", "Sueldo_Min": 78000, "Sueldo_Max": 120000, "Cantidad": 50},
        {"Puesto": "Jefe de Almacen_Bodega", "Sueldo_Min": 55000, "Sueldo_Max": 83000, "Cantidad": 15}
    ],

    "Planta_Produccion": [
        {"Puesto": "Operador de produccion (Linea)", "Sueldo_Min": 49000, "Sueldo_Max": 62000, "Cantidad": 420},
        {"Puesto": "Tecnico de Mantenimiento", "Sueldo_Min": 52000, "Sueldo_Max": 75000, "Cantidad": 70},
        {"Puesto": "Quimico_Control de Calidad", "Sueldo_Min": 80000, "Sueldo_Max": 125000, "Cantidad": 20},
        {"Puesto": "Jefe de Produccion", "Sueldo_Min": 135000, "Sueldo_Max": 225000, "Cantidad": 7}
    ],

    "Ventas": [
        {"Puesto": "Vendedor_Preventista", "Sueldo_Min": 39000, "Sueldo_Max": 60000, "Cantidad": 460},
        {"Puesto": "Ejecutivo Comercial", "Sueldo_Min": 82000, "Sueldo_Max": 131500, "Cantidad": 145},
        {"Puesto": "Ejecutivo de Cuentas Clave (KAM)", "Sueldo_Min": 90000, "Sueldo_Max": 145000, "Cantidad": 9},
        {"Puesto": "Gerente de Territorio", "Sueldo_Min": 112000, "Sueldo_Max": 200000, "Cantidad": 38},
        {"Puesto": "Backoffice_Ventas / Soporte_Comercial", "Sueldo_Min": 42000, "Sueldo_Max": 69000, "Cantidad": 22}
    ],

    "Administracion": [
        {"Puesto": "Asistente Administrativo", "Sueldo_Min": 42000, "Sueldo_Max": 61000, "Cantidad": 75},
        {"Puesto": "Analista Administrativo", "Sueldo_Min": 65000, "Sueldo_Max": 96000, "Cantidad": 95},
        {"Puesto": "Recepcionista", "Sueldo_Min": 30000, "Sueldo_Max": 38000, "Cantidad": 25}
    ],

    "Finanzas": [
        {"Puesto": "Contador_Especialista Financiero", "Sueldo_Min": 88000, "Sueldo_Max": 150000, "Cantidad": 12}
    ],

    "Legal": [
        {"Puesto": "Abogado Corporativo", "Sueldo_Min": 125000, "Sueldo_Max": 240000, "Cantidad": 5}
    ],

    "Marketing": [
        {"Puesto": "Especialista de Marketing_Marca", "Sueldo_Min": 105000, "Sueldo_Max": 185000, "Cantidad": 14}
    ],

    "IT_DataStrategy": [
        {"Puesto": "Analista de Datos_BI", "Sueldo_Min": 90000, "Sueldo_Max": 165000, "Cantidad": 14},
        {"Puesto": "Administrador de Sistemas_Redes", "Sueldo_Min": 70000, "Sueldo_Max": 120000, "Cantidad": 12},
        {"Puesto": "Ingeniero de Automatizacion_IoT", "Sueldo_Min": 110000, "Sueldo_Max": 175000, "Cantidad": 8}
    ],

    "Alta_Direccion": [
        {"Puesto": "Gerente de Area_Departamento", "Sueldo_Min": 165000, "Sueldo_Max": 270000, "Cantidad": 9},
        {"Puesto": "Director_Vicepresidente", "Sueldo_Min": 380000, "Sueldo_Max": 850000, "Cantidad": 4}
    ],

    "Servicios_Generales": [
        {"Puesto": "Personal Servicios Generales (Conserjes, Mensajeros, Cafeteria)", "Sueldo_Min": 31000, "Sueldo_Max": 38000, "Cantidad": 290}
    ],

    "Seguridad_Industrial": [
        {"Puesto": "Oficial de Seguridad_CCTV", "Sueldo_Min": 33000, "Sueldo_Max": 40000, "Cantidad": 95}
    ]
}

# Compatibilidad: si existen funciones que aún usan "Cantidad_2025", mantener ambos campos
for dept, roles in DEPARTAMENTOS_RRHH.items():
    for r in roles:
        if "Cantidad" in r and "Cantidad_2025" not in r:
            r["Cantidad_2025"] = r["Cantidad"]




# BLOQUE 6. ESTACIONALIDAD_MENSUAL
#===================================================================================================

# -------------------- BLOQUE DE MAPEOS Y VARIABLES --------------------
ESTACIONALIDAD_MENSUAL = {
    "Enero": 0.80, "Febrero": 0.85, "Marzo": 1.00, "Abril": 1.05, "Mayo": 1.05,
    "Junio": 1.10, "Julio": 1.20, "Horeca (Hotel/Rest/Café)": 1.10, "Septiembre": 0.95,
    "Octubre": 1.00, "Noviembre": 1.05, "Diciembre": 1.20
}

# Estacionalidad Mensual
#normalizar_pesos(ESTACIONALIDAD_MENSUAL, "Factor", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
#validar_pesos(ESTACIONALIDAD_MENSUAL, "Peso_Normalizado", "DimEstacionalidaMensual — Peso normalizado")

meses_es = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# No necesitas este paso si ESTACIONALIDAD_MENSUAL ya es dict:
mapa_estacionalidad = ESTACIONALIDAD_MENSUAL

# # =================================================================================================
# BLOQUE 6.1: ESTACIONALIDAD POR TRIMESTRE
# =================================================================================================

ESTACIONALIDAD_TRIMESTRAL = [
    {"Trimestre": "Q1", "Peso_Estacional": 0.22, "%_Peso": 22},
    {"Trimestre": "Q2", "Peso_Estacional": 0.28, "%_Peso": 28},
    {"Trimestre": "Q3", "Peso_Estacional": 0.30, "%_Peso": 30},
    {"Trimestre": "Q4", "Peso_Estacional": 0.20, "%_Peso": 20}
]

normalizar_pesos(ESTACIONALIDAD_TRIMESTRAL, "%_Peso", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
validar_pesos(ESTACIONALIDAD_TRIMESTRAL, "Peso_Normalizado", "DimEstacionalidadTrimestral — Peso normalizado")

 
ESTACIONALIDAD_DIA_SEMANA = [
    {"Dia": "Lunes", "Factor": 0.15},
    {"Dia": "Martes", "Factor": 0.14},
    {"Dia": "Miércoles", "Factor": 0.15},
    {"Dia": "Jueves", "Factor": 0.13},
    {"Dia": "Viernes", "Factor": 0.16},
    {"Dia": "Sábado", "Factor": 0.15},
    {"Dia": "Domingo", "Factor": 0.12}
]

normalizar_pesos(ESTACIONALIDAD_DIA_SEMANA, "Factor", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
validar_pesos(ESTACIONALIDAD_DIA_SEMANA, "Peso_Normalizado", "DimEStacionalidadDiaSemana — Peso normalizado")


ESTACIONALIDAD_MESDIA = {
    1: 0.8244, 2: 0.8524, 3: 0.8803, 4: 0.9083, 5: 0.9362, 6: 0.9642, 7: 0.9921,
    8: 1.0201, 9: 1.0481, 10: 1.076, 11: 1.104, 12: 1.1319, 13: 1.1599, 14: 1.1878,
    15: 1.35, 16: 1.3298, 17: 1.3095, 18: 1.2893, 19: 1.269, 20: 1.2488, 21: 1.2285,
    22: 1.2083, 23: 1.1881, 24: 1.1679, 25: 1.22, 26: 1.1500, 27: 1.1625, 28: 1.1750,
    29: 1.1875, 30: 1.45, 31: 0.75
}


# ============================================================
# 📊 Estacionalidad por Categoría de Producto
# ============================================================
# Nota: 'Factor' ajusta el volumen esperado por categoría en todo el año.

# Estacionalidad por Categoría de Producto
ESTACIONALIDAD_CATEGORIA = {
    "Refrescos": 1.10, "Agua": 1.05, "Jugos": 0.95, "Energizantes": 1.15,
    "Isotónicos": 1.10, "Té": 0.90, "RTD Café": 0.85, "RTD Funcionales": 1.00
}


ESTACIONALIDAD_CATEGORIA = {
    "Refrescos": 1.10,     # más altos en verano
    "Agua": 1.05,          # consumo estable, ligero pico en calor
    "Jugos": 0.95,         # más fuertes en invierno y desayunos
    "Energizantes": 1.15,  # picos en verano y eventos deportivos
    "Isotónicos": 1.10,    # asociados a deporte
    "Té": 0.90,            # más consumo en meses fríos
    "RTD Café": 0.85,      # más consumo en mañanas y meses fríos
    "RTD Funcionales": 1.00 # estable
}

#normalizar_pesos(ESTACIONALIDAD_CATEGORIA, "Peso", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
#validar_pesos(ESTACIONALIDAD_CATEGORIA, "Peso_Normalizado", "DimCEDIS — Peso normalizado")






# BLOQUE 7. RUTAS DE VEHÍCULOS MAS FRECUENTES
# =================================================================================================

RUTAS_FRECUENTES_VEHICULOS = {
    "Hyundai_HD_78": {
        "Segmento": "Interurbano",
        "Rutas_Frecuentes": [
            "Autopista Duarte (Santo Domingo ↔ Santiago)",
            "Carretera Mella (Santo Domingo ↔ San Pedro de Macorís)",
            "Autovía del Este (Santo Domingo ↔ La Romana ↔ Punta Cana)",
            "Carretera Sánchez (Santo Domingo ↔ San Cristóbal ↔ Baní ↔ Azua)"
        ]
    },
    "Fuso_Canter_FE85": {
        "Segmento": "Interurbano Provincial",
        "Rutas_Frecuentes": [
            "Santo Domingo ↔ San Cristóbal ↔ Baní",
            "Santiago ↔ Moca ↔ La Vega ↔ Bonao",
            "San Pedro ↔ Hato Mayor ↔ El Seibo",
            "San Francisco ↔ Nagua ↔ Samaná"
        ]
    },
    "Isuzu_NPR": {
        "Segmento": "Urbano Local",
        "Rutas_Frecuentes": [
            "Santo Domingo Norte ↔ Villa Mella ↔ Los Alcarrizos",
            "Santiago Centro ↔ Cienfuegos ↔ Gurabo",
            "San Cristóbal ↔ Nigua ↔ Haina",
            "La Romana ↔ Caleta ↔ Villa Hermosa"
        ]
    },
    "Panel_H_100": {
        "Segmento": "Local/Rural",
        "Rutas_Frecuentes": [
            "Barahona ↔ Cabral ↔ Polo",
            "San Juan ↔ El Cercado ↔ Las Matas",
            "Monte Plata ↔ Bayaguana ↔ Sabana Grande de Boyá",
            "Dajabón ↔ Loma de Cabrera ↔ Restauración"
        ]
    },
    "Freightliner_M2": {
        "Segmento": "Regional Nacional",
        "Rutas_Frecuentes": [
            "Santiago ↔ San Juan ↔ Barahona (Ruta Sur)",
            "Puerto Plata ↔ La Vega ↔ Bonao ↔ Santo Domingo (Ruta Norte-Centro)",
            "Ruta transversal: San Francisco ↔ Cotuí ↔ Santo Domingo",
            "Ruta costera: La Romana ↔ Higüey ↔ Punta Cana ↔ Miches"
        ]
    }
}





# BLOQUE 8: CEDIS CON UBICACIÓN Y PESO (SANTO DOMINGO = 0.43)
# =================================================================================================

CEDIS = [
        {"ID_CEDI": "CEDI-01", "Nombre": "CEDI Occidental", "Nombre_Provincia": "Distrito Nacional","Region": "Ozama","Lat": 8.455145120808126, "Lon": -69.92481973850903, "Capacidad_Pallets": 15000, "Tipo_Almacen": "Principal", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-02", "Nombre": "CEDI Santiago", "Nombre_Provincia": "Santiago","Region": "Cibao Norte","Lat": 19.416663708714793, "Lon": -70.65449981699167, "Capacidad_Pallets": 4000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-03", "Nombre": "CEDI La Romana", "Nombre_Provincia": "La Romana","Region": "Este", "Lat": 18.4430868921463, "Lon": -69.0374719940046, "Capacidad_Pallets": 3000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-04", "Nombre": "CEDI Barahona", "Nombre_Provincia": "Barahona","Region": "Enriquillo","Lat": 18.217418470246976, "Lon": -71.10914151887681, "Capacidad_Pallets": 3000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-05", "Nombre": "CEDI Herrera", "Nombre_Provincia": "Santo Domingo","Region": "Ozama", "Lat": 18.54175046368255, "Lon": -70.06626871292468, "Capacidad_Pallets": 3500, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-06", "Nombre": "CEDI Mao", "Nombre_Provincia": "Valverde","Region": "Cibao Noroeste",  "Lat": 19.56349039544245, "Lon": -71.09462597827348, "Capacidad_Pallets": 2000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-07", "Nombre": "CEDI Constanza","Nombre_Provincia": "La Vega", "Region": "Cibao Sur", "Lat": 18.903886981951022, "Lon": -70.74413088830306, "Capacidad_Pallets": 2000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-08", "Nombre": "CEDI San Francisco de Macorís", "Nombre_Provincia": "Duarte","Region": "Higuamo", "Lat": 19.279940640486274, "Lon": -70.24218620175125, "Capacidad_Pallets": 2000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-09", "Nombre": "CEDI Monte Plata","Nombre_Provincia": "Monte Plata", "Region": "Ozama", "Lat": 18.80985663515591, "Lon": -69.7831291487138, "Capacidad_Pallets": 3500, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-10", "Nombre": "CEDI San Juan", "Nombre_Provincia": "San Juan","Region": "El Valle", "Lat": 18.817938428082268, "Lon": -71.2339219450955, "Capacidad_Pallets": 2000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-11", "Nombre": "CEDI Bani", "Nombre_Provincia": "Peravia","Region": "Valdesia", "Lat": 18.280685845074938, "Lon": -70.32489951615229, "Capacidad_Pallets": 2000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"},
        {"ID_CEDI": "CEDI-12", "Nombre": "CEDI Zona Oriental", "Nombre_Provincia": "Santo Domingo","Region": "Ozama", "Lat": 18.48796957908642, "Lon": -69.82600574477343, "Capacidad_Pallets": 4000, "Tipo_Almacen": "Regional", "Estado_Operativo" : "Activo"}
]
# 2) Normalizar
"""normalizar_pesos(CEDIS, "Peso", "Peso_Normalizado")
# 3) Validar después de normalizar (opcional)
validar_pesos(CEDIS, "Peso_Normalizado", "DimCEDIS — Peso normalizado")    
"""





# BLOQUE 9: COSTO DE PRODUCCIÓN POR CATEGORÍA Y MARCA
# =================================================================================================

COSTO_PRODUCCION_BEBIDAS = [
    {
        "Categoria": "Refrescos",
        "Marca_Producto": "Coca-Cola",
        "Materias_Primas": ["Agua", "Azucar", "Concentrado", "PET"],
        "Porcentaje_Costo_Produccion": 0.25
        #"Ejemplo_Gasto_DOP": 210_000_000
    },
    {
        "Categoria": "Refrescos_Lig",
        "Marca_Producto": "Coca-Cola Lig",
        "Materias_Primas": ["Agua", "Endulzante", "CO2", "PET"],
        "Porcentaje_Costo_Produccion": 0.22
        #"Ejemplo_Gasto_DOP": 180_000_000
    },
    {
        "Categoria": "Jugos",
        "Marca_Producto": "Del Valle",
        "Materias_Primas": ["Agua", "Jugo_Fruta", "Azucar"],
        "Porcentaje_Costo_Produccion": 0.15
        #"Ejemplo_Gasto_DOP": 85_000_000
    },
    {
        "Categoria": "Agua_Embotellada",
        "Marca_Producto": "Cristal",
        "Materias_Primas": ["Agua", "PET", "Etiquetas"],
        "Porcentaje_Costo_Produccion": 0.12
        #"Ejemplo_Gasto_DOP": 65_000_000
    },
    {
        "Categoria": "Energizantes",
        "Marca_Producto": "Powerade",
        "Materias_Primas": ["Agua", "Azucar", "Concentrado"],
        "Porcentaje_Costo_Produccion": 0.10
        #"Ejemplo_Gasto_DOP": 40_000_000
    },
    {
        "Categoria": "Te",
        "Marca_Producto": "FUZE Tea",
        "Materias_Primas": ["Agua", "Extracto_Te", "Azucar"],
        "Porcentaje_Costo_Produccion": 0.07
        #"Ejemplo_Gasto_DOP": 35_000_000
    }
]




# BLOQUE 10: COSTO DE PRODUCCIÓN POR CATEGORÍA Y MARCA
# =================================================================================================

COSTO_PRODUCCION_BEBIDAS = [
    {
        "Categoria": "Refrescos",
        "Marca_Producto": "Coca-Cola",
        "Materias_Primas": ["Agua", "Azucar", "Concentrado", "PET"],
        "Porcentaje_Costo_Produccion": 0.25
        #"Ejemplo_Gasto_DOP": 210_000_000
    },
    {
        "Categoria": "Refrescos_Lig",
        "Marca_Producto": "Coca-Cola Lig",
        "Materias_Primas": ["Agua", "Endulzante", "CO2", "PET"],
        "Porcentaje_Costo_Produccion": 0.22
        #"Ejemplo_Gasto_DOP": 180_000_000
    },
    {
        "Categoria": "Jugos",
        "Marca_Producto": "Del Valle",
        "Materias_Primas": ["Agua", "Jugo_Fruta", "Azucar"],
        "Porcentaje_Costo_Produccion": 0.15
        #"Ejemplo_Gasto_DOP": 85_000_000
    },
    {
        "Categoria": "Agua_Embotellada",
        "Marca_Producto": "Cristal",
        "Materias_Primas": ["Agua", "PET", "Etiquetas"],
        "Porcentaje_Costo_Produccion": 0.12
        #"Ejemplo_Gasto_DOP": 65_000_000
    },
    {
        "Categoria": "Energizantes",
        "Marca_Producto": "Powerade",
        "Materias_Primas": ["Agua", "Azucar", "Concentrado"],
        "Porcentaje_Costo_Produccion": 0.10
        #"Ejemplo_Gasto_DOP": 40_000_000
    },
    {
        "Categoria": "Te",
        "Marca_Producto": "FUZE Tea",
        "Materias_Primas": ["Agua", "Extracto_Te", "Azucar"],
        "Porcentaje_Costo_Produccion": 0.07
        #"Ejemplo_Gasto_DOP": 35_000_000
    }
]




# BLOQUE 11: DISTRIBUCIÓN DE GASTO DE MARKETING
# =================================================================================================

# Distribución del Presupuesto de Marketing
GASTO_MARKETING_MIX = {
    "Publicidad Digital": 0.35,
    "Medios Tradicionales (TV/Radio)": 0.25,
    "Punto de Venta (Trade)": 0.20,
    "Patrocinios y Eventos": 0.15,
    "Investigación de Mercado": 0.05
}


GASTO_MARKETING_DETALLADO = [
    {
        "Categoria_Gasto": "Publicidad_y_propaganda",
        "Detalle": "TV, radio, prensa, digital",
        #"Estimado_Anual_DOP": (450_000_000, 525_000_000),
        "Porcentaje_Sobre_Marketing": 0.40,
        "Observaciones": "Incluye spots, banners, campañas masivas y digitales"
    },
    {
        "Categoria_Gasto": "Promocion_en_punto_de_venta",
        "Detalle": "Material POP, degustaciones, activaciones",
        #"Estimado_Anual_DOP": (190_000_000, 220_000_000),
        "Porcentaje_Sobre_Marketing": 0.17,
        "Observaciones": "Sampling, stands, concursos en supermercados y colmados"
    },
    {
        "Categoria_Gasto": "Patrocinios_y_eventos",
        "Detalle": "Patroc. deporte, conciertos, iniciativas",
        #"Estimado_Anual_DOP": (125_000_000, 160_000_000),
        "Porcentaje_Sobre_Marketing": 0.12,
        "Observaciones": "Grandes eventos y patrocinios de actividades deportivas"
    },
    {
        "Categoria_Gasto": "Estudios_de_mercado_e_insights",
        "Detalle": "Encuestas, focus group, análisis",
        #"Estimado_Anual_DOP": (30_000_000, 40_000_000),
        "Porcentaje_Sobre_Marketing": 0.03,
        "Observaciones": "Herramientas de consumer research"
    },
    {
        "Categoria_Gasto": "Marketing_digital_y_RRSS",
        "Detalle": "Redes sociales, influencers, web",
        #"Estimado_Anual_DOP": (85_000_000, 120_000_000),
        "Porcentaje_Sobre_Marketing": 0.08,
        "Observaciones": "Influencers, social media ads, engagement campañas"
    },
    {
        "Categoria_Gasto": "Responsabilidad_y_sostenibilidad",
        "Detalle": "Programas sociales y ambientales",
        #"Estimado_Anual_DOP": (50_000_000, 60_000_000),
        "Porcentaje_Sobre_Marketing": 0.05,
        "Observaciones": "Branding positivo y RSE (reciclaje, capacitación, etc.)"
    },
    {
        "Categoria_Gasto": "Capacitacion_fuerza_de_ventas",
        "Detalle": "Entrenamiento, materiales, incentivos",
        #"Estimado_Anual_DOP": (25_000_000, 30_000_000),
        "Porcentaje_Sobre_Marketing": 0.02,
        "Observaciones": "Formación interna comercial y materiales de inducción"
    },
    {
        "Categoria_Gasto": "Estrategia_y_consultoria",
        "Detalle": "Planeación, agencias, consultores",
        #"Estimado_Anual_DOP": (60_000_000, 80_000_000),
        "Porcentaje_Sobre_Marketing": 0.06,
        "Observaciones": "Servicios de agencia, planificación y soporte externo"
    },
    {
        "Categoria_Gasto": "Promociones_al_consumidor_final",
        "Detalle": "Rebajas, sorteos, premios",
        #"Estimado_Anual_DOP": (80_000_000, 100_000_000),
        "Porcentaje_Sobre_Marketing": 0.07,
        "Observaciones": "Cupones, sorteos, promociones directas al cliente final"
    }
]




# BLOQUE 12: GASTOS LOGÍSTICOS Y KPIs DE OPERACIÓNn
# =================================================================================================

GASTOS_LOGISTICOS_OPERACION = [
    {
        "Tipo_Gasto": "Combustibles",
        "KPI_Metrica": "Litros_consumidos_en_distribucion",
        "Ano_Base": 2022,
        "Valor_Estimado": "100,800 litros"
    },
    {
        "Tipo_Gasto": "Gasto_en_combustibles",
        "KPI_Metrica": "Costo_promedio_litro",
        "Ano_Base": 2022,
        "Valor_Estimado": "6,500,000 DOP"
    },
    {
        "Tipo_Gasto": "Km_recorridos",
        "KPI_Metrica": "Total_rutas_nacionales",
        "Ano_Base": "Anual",
        "Valor_Estimado": "1,250,000 km"
    },
    {
        "Tipo_Gasto": "Vehiculos_utilizados",
        "KPI_Metrica": "Flota_activa",
        "Ano_Base": 2023,
        "Valor_Estimado": "350 unidades"
    },
    {
        "Tipo_Gasto": "Mantenimiento_flota",
        "KPI_Metrica": "Costo_anual",
        "Ano_Base": 2022,
        "Valor_Estimado": "8,000,000 DOP"
    },
    {
        "Tipo_Gasto": "Repuestos_Llantas",
        "KPI_Metrica": "Consumo_global",
        "Ano_Base": 2022,
        "Valor_Estimado": "1,100,000 DOP"
    }
]




# BLOQUE 13: PROMOCIONES Y SU IMPACTO EN VENTA
# =================================================================================================

PROMOCIONES_MAESTRAL = [
    {"Promocion": "2x1 Refresco", "Peso_Incremento_Venta": 0.23, "%_Peso_Incremento": 23},
    {"Promocion": "Descuento Jugo", "Peso_Incremento_Venta": 0.14, "%_Peso_Incremento": 14},
    {"Promocion": "Prom Agua", "Peso_Incremento_Venta": 0.16, "%_Peso_Incremento": 16},
    {"Promocion": "Combo Familiar", "Peso_Incremento_Venta": 0.19, "%_Peso_Incremento": 19},
    {"Promocion": "Precio Especial", "Peso_Incremento_Venta": 0.09, "%_Peso_Incremento": 9},
    {"Promocion": "Black Friday", "Peso_Incremento_Venta": 0.12, "%_Peso_Incremento": 12},
    {"Promocion": "Sobreventa Verano", "Peso_Incremento_Venta": 0.07, "%_Peso_Incremento": 7}
]

# Validación rápida




# DATOS FALTANTES

# --- Configuración de Canales (Crucial para RD) ---
CANALES_RD = {
    "Colmado (Tradicional)": {"peso": 0.55, "segmentos": ["C-", "D", "E"], "ticket_bajo": True},
    "Supermercado Cadena":   {"peso": 0.15, "segmentos": ["A", "B", "C+"], "ticket_bajo": False},
    "Supermercado Indep.":   {"peso": 0.10, "segmentos": ["B", "C+", "C-"], "ticket_bajo": False},
    "HORECA (Turismo/Rest)": {"peso": 0.12, "segmentos": ["A", "B"], "ticket_bajo": False},
    "Mayorista":             {"peso": 0.08, "segmentos": ["C+", "C-"], "ticket_bajo": False}
}
TIPOS_CANAL_LIST = list(CANALES_RD.keys())
PESOS_CANAL_LIST = [CANALES_RD[k]["peso"] for k in TIPOS_CANAL_LIST]


# Asegurarse que el logger, DIRS, SEED_VAL estén configurados
if 'logger' not in globals():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

if 'DIRS' not in globals():
    DIRS = {
        "OUTPUT": Path("C:/DE/output"),
        "PARTS": Path("C:/DE/output/FactVentasAvanzadaParticionada")
    }
    DIRS["OUTPUT"].mkdir(parents=True, exist_ok=True)
    if DIRS["PARTS"].exists():
        import shutil
        shutil.rmtree(DIRS["PARTS"])
    DIRS["PARTS"].mkdir(parents=True, exist_ok=True)

if 'SEED_VAL' not in globals():
    SEED_VAL = 42
np.random.seed(SEED_VAL)
random.seed(SEED_VAL)

# Constantes adicionales para DimCliente (si no están ya definidas)
if 'ANOS_SIMULACION' not in globals():
    ANOS_SIMULACION = [2021, 2022, 2023, 2024, 2025] # Usar los años reales de simulación

if 'NUM_CLIENTES_POR_ANO' not in globals():
    NUM_CLIENTES_POR_ANO = {
        2021: 50_000,
        2022: 52_000,
        2023: 55_000,
        2024: 58_000,
        2025: 60_000
    } # Número base de clientes *totales* deseados para cada año

if 'CHURN_RATE_ANUAL' not in globals():
    CHURN_RATE_ANUAL = 0.05 # 5% de churn anual

# Definición de PESO_SEGMENTACION_CANAL (ejemplo, si no está definido)
# Asegúrate de que los nombres de los canales aquí coincidan con los generados en DimCanalDistribucion
if 'PESO_SEGMENTACION_CANAL' not in globals():
    PESO_SEGMENTACION_CANAL = {
        "Venta al Detalle (Minimercados, Colmados)": {"A": 0.1, "B": 0.2, "C": 0.4, "D": 0.2, "E": 0.1},
        "Supermercados y Grandes Superficies": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1, "E": 0.0},
        "HORECA (Hoteles, Restaurantes, Cafeterías)": {"A": 0.3, "B": 0.4, "C": 0.2, "D": 0.1, "E": 0.0},
        "Mayoristas y Distribuidores": {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05, "E": 0.0},
        "Institucional (Oficinas, Colegios)": {"A": 0.2, "B": 0.3, "C": 0.3, "D": 0.15, "E": 0.05},
        "Tiendas de Conveniencia y Farmacias": {"A": 0.15, "B": 0.25, "C": 0.35, "D": 0.15, "E": 0.1},
        "E-commerce y Marketplaces": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.15, "E": 0.1},
        "Exportación": {"A": 0.6, "B": 0.3, "C": 0.1, "D": 0.0, "E": 0.0},
        "Consumo Directo": {"A": 0.05, "B": 0.1, "C": 0.3, "D": 0.4, "E": 0.15},
    }




# --- Base de datos de Productos (Los 55 mencionados en el IPYNB) ---
# Hemos categorizado, asignado marcas y estimado características clave
PRODUCTOS_BEPENSA_BASE = [
    {"ID_Prod_Unico": "P001", "Nombre_Producto": "Coca-Cola Original", "Marca": "Coca-Cola", "Variedad": "Regular", "Categoria_Principal": "CSD", "Sub_Categoria": "Cola", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P002", "Nombre_Producto": "Coca-Cola Sin Azúcar", "Marca": "Coca-Cola", "Variedad": "Zero Azúcar", "Categoria_Principal": "CSD", "Sub_Categoria": "Cola", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P003", "Nombre_Producto": "Fanta Naranja", "Marca": "Fanta", "Variedad": "Naranja", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P004", "Nombre_Producto": "Sprite", "Marca": "Sprite", "Variedad": "Lima-Limón", "Categoria_Principal": "CSD", "Sub_Categoria": "Lima-Limón", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P005", "Nombre_Producto": "Country Club Merengue", "Marca": "Country Club", "Variedad": "Merengue", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores Locales", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P006", "Nombre_Producto": "Country Club Frambuesa", "Marca": "Country Club", "Variedad": "Frambuesa", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores Locales", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P007", "Nombre_Producto": "Presidente Agua", "Marca": "Presidente", "Variedad": "Natural", "Categoria_Principal": "Agua", "Sub_Categoria": "Agua Pura", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P008", "Nombre_Producto": "Dasani Agua", "Marca": "Dasani", "Variedad": "Natural", "Categoria_Principal": "Agua", "Sub_Categoria": "Agua Mineral", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P009", "Nombre_Producto": "Powerade Blue", "Marca": "Powerade", "Variedad": "Berry Blast", "Categoria_Principal": "Isotónico", "Sub_Categoria": "Deportivo", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P010", "Nombre_Producto": "Powerade Roja", "Marca": "Powerade", "Variedad": "Fruit Punch", "Categoria_Principal": "Isotónico", "Sub_Categoria": "Deportivo", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P011", "Nombre_Producto": "Minute Maid Naranja", "Marca": "Minute Maid", "Variedad": "Naranja", "Categoria_Principal": "Jugo", "Sub_Categoria": "Néctar", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P012", "Nombre_Producto": "Minute Maid Manzana", "Marca": "Minute Maid", "Variedad": "Manzana", "Categoria_Principal": "Minute Maid", "Sub_Categoria": "Néctar", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P013", "Nombre_Producto": "Monster Energy Green", "Marca": "Monster Energy", "Variedad": "Original", "Categoria_Principal": "Energizante", "Sub_Categoria": "Bebida Energética", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.15},
    {"ID_Prod_Unico": "P014", "Nombre_Producto": "Monster Energy Zero", "Marca": "Monster Energy", "Variedad": "Zero Azúcar", "Categoria_Principal": "Energizante", "Sub_Categoria": "Bebida Energética", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.15},
    {"ID_Prod_Unico": "P015", "Nombre_Producto": "Arizona Té Limón", "Marca": "Arizona", "Variedad": "Té Limón", "Categoria_Principal": "Té Listo para Beber", "Sub_Categoria": "Frío", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P016", "Nombre_Producto": "Agua Cristal", "Marca": "Cristal", "Variedad": "Natural", "Categoria_Principal": "Agua", "Sub_Categoria": "Agua Pura", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P017", "Nombre_Producto": "Schweppes Ginger Ale", "Marca": "Schweppes", "Variedad": "Ginger Ale", "Categoria_Principal": "CSD", "Sub_Categoria": "Mezcladores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P018", "Nombre_Producto": "Schweppes Tónica", "Marca": "Schweppes", "Variedad": "Agua Tónica", "Categoria_Principal": "CSD", "Sub_Categoria": "Mezcladores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P019", "Nombre_Producto": "Appletiser", "Marca": "Appletiser", "Variedad": "Manzana Espumosa", "Categoria_Principal": "Jugo Espumoso", "Sub_Categoria": "Premium", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P020", "Nombre_Producto": "Del Valle Pulpy Naranja", "Marca": "Del Valle", "Variedad": "Naranja con Pulpa", "Categoria_Principal": "Jugo", "Sub_Categoria": "Con Pulpa", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P021", "Nombre_Producto": "Monster Ultra White", "Marca": "Monster Energy", "Variedad": "Ultra White (Zero)", "Categoria_Principal": "Energizante", "Sub_Categoria": "Bebida Energética", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.15},
    {"ID_Prod_Unico": "P022", "Nombre_Producto": "Sprite Zero", "Marca": "Sprite", "Variedad": "Zero Azúcar", "Categoria_Principal": "CSD", "Sub_Categoria": "Lima-Limón", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P023", "Nombre_Producto": "Fanta Uva", "Marca": "Fanta", "Variedad": "Uva", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P024", "Nombre_Producto": "Topo Chico Hard Seltzer Piña", "Marca": "Topo Chico", "Variedad": "Hard Seltzer Piña", "Categoria_Principal": "Hard Seltzer", "Sub_Categoria": "Saborizadas", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.20}, # Mayor ISC
    {"ID_Prod_Unico": "P025", "Nombre_Producto": "Topo Chico Hard Seltzer Limón", "Marca": "Topo Chico", "Variedad": "Hard Seltzer Limón", "Categoria_Principal": "Hard Seltzer", "Sub_Categoria": "Saborizadas", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.20},
    {"ID_Prod_Unico": "P026", "Nombre_Producto": "Fuze Tea Durazno", "Marca": "Fuze Tea", "Variedad": "Durazno", "Categoria_Principal": "Té Listo para Beber", "Sub_Categoria": "Frío", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P027", "Nombre_Producto": "Fuze Tea Limón", "Marca": "Fuze Tea", "Variedad": "Limón", "Categoria_Principal": "Té Listo para Beber", "Sub_Categoria": "Frío", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P028", "Nombre_Producto": "Aquarius Manzana", "Marca": "Aquarius", "Variedad": "Manzana", "Categoria_Principal": "Agua Saborizada", "Sub_Categoria": "Baja Caloría", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P029", "Nombre_Producto": "Aquarius Pera", "Marca": "Aquarius", "Variedad": "Pera", "Categoria_Principal": "Agua Saborizada", "Sub_Categoria": "Baja Caloría", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P030", "Nombre_Producto": "Ciel Purificada", "Marca": "Ciel", "Variedad": "Natural", "Categoria_Principal": "Agua", "Sub_Categoria": "Agua Pura", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P031", "Nombre_Producto": "Valle Frut Naranja", "Marca": "Valle Frut", "Variedad": "Naranja", "Categoria_Principal": "Jugo", "Sub_Categoria": "Economico", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P032", "Nombre_Producto": "Valle Frut Manzana", "Marca": "Valle Frut", "Variedad": "Manzana", "Categoria_Principal": "Jugo", "Sub_Categoria": "Economico", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P033", "Nombre_Producto": "Glacéau Smartwater", "Marca": "Glacéau", "Variedad": "Electrolitos", "Categoria_Principal": "Agua Premium", "Sub_Categoria": "Agua Funcional", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P034", "Nombre_Producto": "Burn Energy Drink", "Marca": "Burn", "Variedad": "Original", "Categoria_Principal": "Energizante", "Sub_Categoria": "Bebida Energética", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.15},
    {"ID_Prod_Unico": "P035", "Nombre_Producto": "Canada Dry Ginger Ale", "Marca": "Canada Dry", "Variedad": "Ginger Ale", "Categoria_Principal": "CSD", "Sub_Categoria": "Mezcladores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P036", "Nombre_Producto": "Lipton Ice Tea Limón", "Marca": "Lipton", "Variedad": "Té Limón", "Categoria_Principal": "Té Listo para Beber", "Sub_Categoria": "Frío", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P037", "Nombre_Producto": "Vitaminwater Essential", "Marca": "Vitaminwater", "Variedad": "Naranja-Maracuyá", "Categoria_Principal": "Agua Funcional", "Sub_Categoria": "Vitaminas", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P038", "Nombre_Producto": "Vitaminwater Power-C", "Marca": "Vitaminwater", "Variedad": "Açaí-Arándano", "Categoria_Principal": "Agua Funcional", "Sub_Categoria": "Vitaminas", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P039", "Nombre_Producto": "AdeS Original Soya", "Marca": "AdeS", "Variedad": "Soya Original", "Categoria_Principal": "Bebida Vegetal", "Sub_Categoria": "Soya", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P040", "Nombre_Producto": "AdeS Almendra", "Marca": "AdeS", "Variedad": "Almendra", "Categoria_Principal": "Bebida Vegetal", "Sub_Categoria": "Almendra", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P041", "Nombre_Producto": "Capri Sun Naranja", "Marca": "Capri Sun", "Variedad": "Naranja", "Categoria_Principal": "Jugo Infantil", "Sub_Categoria": "Bolsa", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P042", "Nombre_Producto": "Capri Sun Ponche de Frutas", "Marca": "Capri Sun", "Variedad": "Ponche de Frutas", "Categoria_Principal": "Jugo Infantil", "Sub_Categoria": "Bolsa", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P043", "Nombre_Producto": "Lift Manzana", "Marca": "Lift", "Variedad": "Manzana", "Categoria_Principal": "CSD", "Sub_Categoria": "Manzana", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P044", "Nombre_Producto": "Sidral Mundet Manzana", "Marca": "Sidral Mundet", "Variedad": "Manzana", "Categoria_Principal": "CSD", "Sub_Categoria": "Manzana", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P045", "Nombre_Producto": "Fresca Toronja", "Marca": "Fresca", "Variedad": "Toronja", "Categoria_Principal": "CSD", "Sub_Categoria": "Toronja", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P046", "Nombre_Producto": "Escuis Naranja", "Marca": "Escuis", "Variedad": "Naranja", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P047", "Nombre_Producto": "Delaware Punch", "Marca": "Delaware Punch", "Variedad": "Frutas Rojas", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P048", "Nombre_Producto": "Zico Agua de Coco", "Marca": "Zico", "Variedad": "Natural", "Categoria_Principal": "Agua de Coco", "Sub_Categoria": "Natural", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P049", "Nombre_Producto": "Honest Tea Verde", "Marca": "Honest Tea", "Variedad": "Té Verde", "Categoria_Principal": "Té Listo para Beber", "Sub_Categoria": "Orgánico", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P050", "Nombre_Producto": "Simply Naranja", "Marca": "Simply", "Variedad": "Naranja (Sin Azúcar)", "Categoria_Principal": "Jugo Premium", "Sub_Categoria": "Refrigerado", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P051", "Nombre_Producto": "Gold Peak Tea Dulce", "Marca": "Gold Peak", "Variedad": "Té Dulce", "Categoria_Principal": "Té Listo para Beber", "Sub_Categoria": "Refrigerado", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P052", "Nombre_Producto": "Topochico Mineral Water", "Marca": "Topo Chico", "Variedad": "Mineral Natural", "Categoria_Principal": "Agua Mineral", "Sub_Categoria": "Carbonatada", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P053", "Nombre_Producto": "Smartwater Sparkling", "Marca": "Smartwater", "Variedad": "Carbonatada", "Categoria_Principal": "Agua Premium", "Sub_Categoria": "Carbonatada", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
    {"ID_Prod_Unico": "P054", "Nombre_Producto": "Barrilitos Piña", "Marca": "Barrilitos", "Variedad": "Piña", "Categoria_Principal": "CSD", "Sub_Categoria": "Sabores", "Aplica_ISC": True, "Tasa_ISC_Pct": 0.10},
    {"ID_Prod_Unico": "P055", "Nombre_Producto": "Glaciar Saborizado Manzana", "Marca": "Glaciar", "Variedad": "Manzana", "Categoria_Principal": "Agua Saborizada", "Sub_Categoria": "Baja Caloría", "Aplica_ISC": False, "Tasa_ISC_Pct": 0.00},
]

# --- Tipos de Empaque Realistas (con costos asociados y propiedades de manejo) ---
EMPAQUES_BASE = [
    {"ID_Empaque": "E001", "Tipo": "Lata", "Material": "Aluminio", "Capacidad_ml": 237, "Unidades_x_Caja": 24, "Es_Retornable": False, "Peso_Unitario_Kg": 0.26, "Costo_Empaque_Unit": 5.0},
    {"ID_Empaque": "E002", "Tipo": "Lata", "Material": "Aluminio", "Capacidad_ml": 355, "Unidades_x_Caja": 24, "Es_Retornable": False, "Peso_Unitario_Kg": 0.38, "Costo_Empaque_Unit": 6.5},
    {"ID_Empaque": "E003", "Tipo": "Pet", "Material": "Plástico", "Capacidad_ml": 500, "Unidades_x_Caja": 12, "Es_Retornable": False, "Peso_Unitario_Kg": 0.53, "Costo_Empaque_Unit": 7.0},
    {"ID_Empaque": "E004", "Tipo": "Pet", "Material": "Plástico", "Capacidad_ml": 1000, "Unidades_x_Caja": 12, "Es_Retornable": False, "Peso_Unitario_Kg": 1.05, "Costo_Empaque_Unit": 9.0},
    {"ID_Empaque": "E005", "Tipo": "Pet", "Material": "Plástico", "Capacidad_ml": 2000, "Unidades_x_Caja": 6, "Es_Retornable": False, "Peso_Unitario_Kg": 2.08, "Costo_Empaque_Unit": 12.0},
    {"ID_Empaque": "E006", "Tipo": "Vidrio", "Material": "Vidrio", "Capacidad_ml": 207, "Unidades_x_Caja": 24, "Es_Retornable": True, "Peso_Unitario_Kg": 0.45, "Costo_Empaque_Unit": 15.0}, # Retornable: costo inicial alto
    {"ID_Empaque": "E007", "Tipo": "Vidrio", "Material": "Vidrio", "Capacidad_ml": 355, "Unidades_x_Caja": 24, "Es_Retornable": True, "Peso_Unitario_Kg": 0.65, "Costo_Empaque_Unit": 20.0},
    {"ID_Empaque": "E008", "Tipo": "Caja", "Material": "Cartón", "Capacidad_ml": 1000, "Unidades_x_Caja": 12, "Es_Retornable": False, "Peso_Unitario_Kg": 1.05, "Costo_Empaque_Unit": 8.0}, # Jugos
    {"ID_Empaque": "E009", "Tipo": "Botellón", "Material": "Plástico", "Capacidad_ml": 18900, "Unidades_x_Caja": 1, "Es_Retornable": True, "Peso_Unitario_Kg": 19.5, "Costo_Empaque_Unit": 100.0}, # Agua 5 galones
    {"ID_Empaque": "E010", "Tipo": "Mini Pet", "Material": "Plástico", "Capacidad_ml": 300, "Unidades_x_Caja": 24, "Es_Retornable": False, "Peso_Unitario_Kg": 0.33, "Costo_Empaque_Unit": 6.0},
    {"ID_Empaque": "E011", "Tipo": "Lata Slim", "Material": "Aluminio", "Capacidad_ml": 269, "Unidades_x_Caja": 24, "Es_Retornable": False, "Peso_Unitario_Kg": 0.28, "Costo_Empaque_Unit": 5.8}, # Energizantes, Hard Seltzer
]

# Tasas de impuestos globales (simplificadas)
ITBIS_TASA = 0.18 # 18% para la mayoría de productos en RD




# --- Variables Globales y Configuración (Hardcoded para integridad) ---
if 'logger' not in globals():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    logger = logging.getLogger(__name__)
if 'DIRS' not in globals():
    DIRS = {"PARTS": Path("C:/DE/output/FactVentasParticionada"), "OUTPUT": Path("C:/DE/output")}
    DIRS["OUTPUT"].mkdir(parents=True, exist_ok=True)
if 'SEED_VAL' not in globals():
    SEED_VAL = 42
if 'DB_MEMORIA' not in globals():
    DB_MEMORIA = {}



NUM_CLIENTES_POR_ANO: Dict[int, int] = {
    2021: 61_500, 2022: 63_000, 2023: 65_000, 2024: 67_800, 2025: 71_000
}

# Parámetros de Simulación
RANGO_DESCUENTO_POR_SEGMENTO = {
    "A": (0.05, 0.15), "B": (0.02, 0.10), "C+": (0.01, 0.07),
    "C-": (0.00, 0.05), "D": (0.00, 0.02), "E": (0.00, 0.00)
}
TIPOS_PAGO = ["Contado", "Crédito"]
PROBS_TIPOS_PAGO = [0.65, 0.35]
MEDIOS_PAGO = ["Efectivo", "Tarjeta", "Transferencia", "Cheque"]
PROBS_MEDIOS_PAGO = [0.40, 0.30, 0.25, 0.05]
ESTADOS_FACTURA = ["Emitida", "Pagada", "Anulada"]
PROBS_ESTADOS_FACTURA = [0.10, 0.88, 0.02]




# --- Datos Maestros para DimGeografia ---
# Esta es una versión extendida de PROVINCIAS_FLAT con datos geográficos y poblacionales
PROVINCIAS_FLAT = [
    {"ID_Provincia": "DO-01","Nombre_Provincia": "Distrito Nacional", "Region": "Ozama", "Poblacion_Estimada": 1120000, "Area_Km2": 104, "Lat": 18.471862, "Lon": -69.892601, "Peso": 0.20},
    {"ID_Provincia": "DO-02","Nombre_Provincia": "Santo Domingo", "Region": "Ozama", "Poblacion_Estimada": 2769000, "Area_Km2": 1296, "Lat": 18.577977, "Lon": -69.969176, "Peso": 0.30},
    {"ID_Provincia": "DO-03","Nombre_Provincia": "Santiago", "Region": "Cibao Central", "Poblacion_Estimada": 1074000, "Area_Km2": 2836, "Lat": 19.451703, "Lon": -70.697072, "Peso": 0.15},
    {"ID_Provincia": "DO-04","Nombre_Provincia": "Puerto Plata", "Region": "Cibao Norte", "Poblacion_Estimada": 331000, "Area_Km2": 1852, "Lat": 19.782035, "Lon": -70.687295, "Peso": 0.05},
    {"ID_Provincia": "DO-05","Nombre_Provincia": "La Vega", "Region": "Cibao Central", "Poblacion_Estimada": 406000, "Area_Km2": 2292, "Lat": 19.220194, "Lon": -70.528414, "Peso": 0.04},
    {"ID_Provincia": "DO-06","Nombre_Provincia": "San Cristobal", "Region": "Valdesia", "Poblacion_Estimada": 620000, "Area_Km2": 1240, "Lat": 18.416806, "Lon": -70.106883, "Peso": 0.06},
    {"ID_Provincia": "DO-07","Nombre_Provincia": "La Romana", "Region": "Higuamo", "Poblacion_Estimada": 262000, "Area_Km2": 654, "Lat": 18.428611, "Lon": -68.972222, "Peso": 0.03},
    {"ID_Provincia": "DO-08","Nombre_Provincia": "Duarte", "Region": "Cibao Nordeste", "Poblacion_Estimada": 298000, "Area_Km2": 1650, "Lat": 19.300000, "Lon": -70.166667, "Peso": 0.03},
    {"ID_Provincia": "DO-09","Nombre_Provincia": "Espaillat", "Region": "Cibao Norte", "Poblacion_Estimada": 233000, "Area_Km2": 839, "Lat": 19.643611, "Lon": -70.430556, "Peso": 0.02},
    {"ID_Provincia": "DO-10","Nombre_Provincia": "Peravia", "Region": "Valdesia", "Poblacion_Estimada": 197000, "Area_Km2": 792, "Lat": 18.290278, "Lon": -70.334722, "Peso": 0.02},
    {"ID_Provincia": "DO-11","Nombre_Provincia": "Monseñor Nouel", "Region": "Cibao Central", "Poblacion_Estimada": 178000, "Area_Km2": 992, "Lat": 18.933333, "Lon": -70.366667, "Peso": 0.02},
    {"ID_Provincia": "DO-12","Nombre_Provincia": "San Pedro de Macorís", "Region": "Higuamo", "Poblacion_Estimada": 204000, "Area_Km2": 1954, "Lat": 18.455278, "Lon": -69.308333, "Peso": 0.02},
    {"ID_Provincia": "DO-13","Nombre_Provincia": "Azua", "Region": "Valdesia", "Poblacion_Estimada": 220000, "Area_Km2": 2531, "Lat": 18.468056, "Lon": -70.735833, "Peso": 0.02},
    {"ID_Provincia": "DO-14","Nombre_Provincia": "María Trinidad Sánchez", "Region": "Cibao Nordeste", "Poblacion_Estimada": 140000, "Area_Km2": 1272, "Lat": 19.380556, "Lon": -69.980556, "Peso": 0.015},
    {"ID_Provincia": "DO-15","Nombre_Provincia": "Sánchez Ramírez", "Region": "Cibao Nordeste", "Poblacion_Estimada": 150000, "Area_Km2": 1196, "Lat": 19.000000, "Lon": -70.200000, "Peso": 0.015},
    {"ID_Provincia": "DO-16","Nombre_Provincia": "El Seibo", "Region": "Higuamo", "Poblacion_Estimada": 110000, "Area_Km2": 1787, "Lat": 18.783333, "Lon": -69.033333, "Peso": 0.01},
    {"ID_Provincia": "DO-17","Nombre_Provincia": "Hato Mayor", "Region": "Higuamo", "Poblacion_Estimada": 90000, "Area_Km2": 1329, "Lat": 18.766667, "Lon": -69.366667, "Peso": 0.01},
    {"ID_Provincia": "DO-18","Nombre_Provincia": "Monte Plata", "Region": "Higuamo", "Poblacion_Estimada": 190000, "Area_Km2": 2632, "Lat": 18.800000, "Lon": -69.766667, "Peso": 0.015},
    {"ID_Provincia": "DO-19","Nombre_Provincia": "Independencia", "Region": "Enriquillo", "Poblacion_Estimada": 60000, "Area_Km2": 2007, "Lat": 18.433333, "Lon": -71.700000, "Peso": 0.005},
    {"ID_Provincia": "DO-20","Nombre_Provincia":"Barahona", "Region": "Enriquillo", "Poblacion_Estimada": 180000, "Area_Km2": 1739, "Lat": 18.200000, "Lon": -71.100000, "Peso": 0.01},
    {"ID_Provincia": "DO-21","Nombre_Provincia": "Pedernales", "Region": "Enriquillo", "Poblacion_Estimada": 35000, "Area_Km2": 2075, "Lat": 18.033333, "Lon": -71.733333, "Peso": 0.003},
    {"ID_Provincia": "DO-22","Nombre_Provincia": "Elías Piña", "Region": "El Valle", "Poblacion_Estimada": 65000, "Area_Km2": 1395, "Lat": 19.000000, "Lon": -71.700000, "Peso": 0.005},
    {"ID_Provincia": "DO-23","Nombre_Provincia": "San Juan", "Region": "El Valle", "Poblacion_Estimada": 230000, "Area_Km2": 3569, "Lat": 18.800000, "Lon": -71.200000, "Peso": 0.02},
    {"ID_Provincia": "DO-24","Nombre_Provincia": "Valverde", "Region": "Cibao Noroeste", "Poblacion_Estimada": 170000, "Area_Km2": 823, "Lat": 19.566667, "Lon": -71.050000, "Peso": 0.01},
    {"ID_Provincia": "DO-25","Nombre_Provincia": "Monte Cristi", "Region": "Cibao Noroeste", "Poblacion_Estimada": 120000, "Area_Km2": 1924, "Lat": 19.833333, "Lon": -71.650000, "Peso": 0.008},
    {"ID_Provincia": "DO-26","Nombre_Provincia":"Dajabón", "Region": "Cibao Noroeste", "Poblacion_Estimada": 70000, "Area_Km2": 1021, "Lat": 19.550000, "Lon": -71.700000, "Peso": 0.005},
    {"ID_Provincia": "DO-27","Nombre_Provincia": "Hermanas Mirabal", "Region": "Cibao Nordeste", "Poblacion_Estimada": 100000, "Area_Km2": 440, "Lat": 19.400000, "Lon": -70.366667, "Peso": 0.008},
    {"ID_Provincia": "DO-28","Nombre_Provincia": "Samaná", "Region": "Cibao Nordeste", "Poblacion_Estimada": 110000, "Area_Km2": 1170, "Lat": 19.200000, "Lon": -69.500000, "Peso": 0.008},
    {"ID_Provincia": "DO-29","Nombre_Provincia": "La Altagracia", "Region": "Yuma", "Poblacion_Estimada": 350000, "Area_Km2": 3010, "Lat": 18.616667, "Lon": -68.716667, "Peso": 0.04},
    {"ID_Provincia": "DO-30","Nombre_Provincia": "Santiago Rodríguez", "Region": "Cibao Noroeste", "Poblacion_Estimada": 60000, "Area_Km2": 1111, "Lat": 19.483333, "Lon": -71.333333, "Peso": 0.005},
    {"ID_Provincia": "DO-31","Nombre_Provincia": "Bahoruco", "Region": "Enriquillo", "Poblacion_Estimada": 100000, "Area_Km2": 1282, "Lat": 18.483333, "Lon": -71.416667, "Peso": 0.008},
    {"ID_Provincia": "DO-32","Nombre_Provincia": "San José de Ocoa", "Region": "Valdesia", "Poblacion_Estimada": 60000, "Area_Km2": 855, "Lat": 18.533333, "Lon": -70.500000, "Peso": 0.005},
]

# Definición de rangos para nivel socioeconómico (ajustable)
NIVELES_SOCIOECONOMICOS = ["Bajo", "Medio-Bajo", "Medio", "Medio-Alto", "Alto"]
# Pesos para la asignación de Nivel Socioeconómico, ajustado por región/provincia
# Esto podría ser más complejo, pero para empezar, una distribución general.
PESOS_NIVEL_SOCIOECONOMICO = {
    "Ozama": {"Bajo": 0.1, "Medio-Bajo": 0.2, "Medio": 0.3, "Medio-Alto": 0.3, "Alto": 0.1},
    "Cibao Central": {"Bajo": 0.15, "Medio-Bajo": 0.25, "Medio": 0.35, "Medio-Alto": 0.2, "Alto": 0.05},
    "Cibao Norte": {"Bajo": 0.2, "Medio-Bajo": 0.3, "Medio": 0.3, "Medio-Alto": 0.15},# ... (continuación de PESOS_NIVEL_SOCIOECONOMICO del mensaje anterior)
    "Cibao Norte": {"Bajo": 0.2, "Medio-Bajo": 0.3, "Medio": 0.3, "Medio-Alto": 0.15, "Alto": 0.05},
    "Cibao Nordeste": {"Bajo": 0.25, "Medio-Bajo": 0.35, "Medio": 0.25, "Medio-Alto": 0.1, "Alto": 0.05},
    "Valdesia": {"Bajo": 0.25, "Medio-Bajo": 0.3, "Medio": 0.3, "Medio-Alto": 0.1, "Alto": 0.05},
    "El Valle": {"Bajo": 0.4, "Medio-Bajo": 0.3, "Medio": 0.2, "Medio-Alto": 0.08, "Alto": 0.02},
    "Enriquillo": {"Bajo": 0.45, "Medio-Bajo": 0.3, "Medio": 0.15, "Medio-Alto": 0.08, "Alto": 0.02},
    "Higuamo": {"Bajo": 0.2, "Medio-Bajo": 0.3, "Medio": 0.3, "Medio-Alto": 0.15, "Alto": 0.05},
    "Yuma": {"Bajo": 0.15, "Medio-Bajo": 0.25, "Medio": 0.35, "Medio-Alto": 0.2, "Alto": 0.05}, # Zona turística (Punta Cana)
}

# Validación rápida de pesos (debe sumar aprox 1.0)
total_peso_geo = sum([p["Peso"] for p in PROVINCIAS_FLAT])
if not math.isclose(total_peso_geo, 1.0, abs_tol=0.05):
    logger.warning(f"⚠️ Pesos geográficos suman {total_peso_geo:.2f}, se renormalizarán en la función.")




# --- Datos Maestros para DimTiempo ---

FERIADOS_RD = {
    # 2021
    date(2021, 1, 1): "Año Nuevo",
    date(2021, 1, 4): "Día de Reyes (trasladado)",
    date(2021, 1, 21): "Día de Nuestra Señora de la Altagracia",
    date(2021, 1, 25): "Día de Duarte (trasladado)",
    date(2021, 2, 27): "Día de la Independencia",
    date(2021, 4, 2): "Viernes Santo",
    date(2021, 5, 1): "Día del Trabajo",
    date(2021, 6, 3): "Corpus Christi",
    date(2021, 8, 16): "Día de la Restauración",
    date(2021, 9, 24): "Día de Nuestra Señora de las Mercedes",
    date(2021, 11, 6): "Día de la Constitución",
    date(2021, 12, 25): "Navidad",
    # 2022
    date(2022, 1, 1): "Año Nuevo",
    date(2022, 1, 10): "Día de Reyes (trasladado)",
    date(2022, 1, 21): "Día de Nuestra Señora de la Altagracia",
    date(2022, 1, 24): "Día de Duarte (trasladado)",
    date(2022, 2, 27): "Día de la Independencia",
    date(2022, 4, 15): "Viernes Santo",
    date(2022, 5, 1): "Día del Trabajo",
    date(2022, 6, 16): "Corpus Christi",
    date(2022, 8, 16): "Día de la Restauración",
    date(2022, 9, 24): "Día de Nuestra Señora de las Mercedes",
    date(2022, 11, 6): "Día de la Constitución",
    date(2022, 12, 25): "Navidad",
    # 2023
    date(2023, 1, 1): "Año Nuevo",
    date(2023, 1, 6): "Día de Reyes",
    date(2023, 1, 21): "Día de Nuestra Señora de la Altagracia",
    date(2023, 1, 26): "Día de Duarte",
    date(2023, 2, 27): "Día de la Independencia",
    date(2023, 4, 7): "Viernes Santo",
    date(2023, 5, 1): "Día del Trabajo",
    date(2023, 6, 8): "Corpus Christi",
    date(2023, 8, 16): "Día de la Restauración",
    date(2023, 9, 24): "Día de Nuestra Señora de las Mercedes",
    date(2023, 11, 6): "Día de la Constitución",
    date(2023, 12, 25): "Navidad",
    # 2024
    date(2024, 1, 1): "Año Nuevo",
    date(2024, 1, 6): "Día de Reyes",
    date(2024, 1, 21): "Día de Nuestra Señora de la Altagracia",
    date(2024, 1, 26): "Día de Duarte",
    date(2024, 2, 27): "Día de la Independencia",
    date(2024, 3, 29): "Viernes Santo",
    date(2024, 5, 1): "Día del Trabajo",
    date(2024, 5, 30): "Corpus Christi",
    date(2024, 8, 16): "Día de la Restauración",
    date(2024, 9, 24): "Día de Nuestra Señora de las Mercedes",
    date(2024, 11, 6): "Día de la Constitución",
    date(2024, 12, 25): "Navidad",
    # 2025
    date(2025, 1, 1): "Año Nuevo",
    date(2025, 1, 6): "Día de Reyes",
    date(2025, 1, 21): "Día de Nuestra Señora de la Altagracia",
    date(2025, 1, 26): "Día de Duarte",
    date(2025, 2, 27): "Día de la Independencia",
    date(2025, 4, 18): "Viernes Santo",
    date(2025, 5, 1): "Día del Trabajo",
    date(2025, 6, 19): "Corpus Christi",
    date(2025, 8, 16): "Día de la Restauración",
    date(2025, 9, 24): "Día de Nuestra Señora de las Mercedes",
    date(2025, 11, 6): "Día de la Constitución",
    date(2025, 12, 25): "Navidad",
    # 2026
    date(2026, 1, 1): "Año Nuevo",
    date(2026, 1, 6): "Día de Reyes",
    date(2026, 1, 21): "Día de Nuestra Señora de la Altagracia",
    date(2026, 1, 26): "Día de Duarte",
    date(2026, 2, 27): "Día de la Independencia",
    date(2026, 4, 3): "Viernes Santo",
    date(2026, 5, 1): "Día del Trabajo",
    date(2026, 6, 4): "Corpus Christi",
    date(2026, 8, 16): "Día de la Restauración",
    date(2026, 9, 24): "Día de Nuestra Señora de las Mercedes",
    date(2026, 11, 6): "Día de la Constitución",
    date(2026, 12, 25): "Navidad",
}

IMPACTO_FERIADO = {
    "dia_antes": 1.1,
    "dia_feriado": 0.6,
    "dia_despues": 0.9,
    "sin_impacto": 1.0,
}

FACTOR_ESTACIONALIDAD_MENSUAL = {
    1: 0.95,
    2: 0.90,
    3: 1.00,
    4: 1.05,
    5: 1.00,
    6: 1.15,
    7: 1.25,
    8: 1.20,
    9: 0.95,
    10: 1.00,
    11: 1.10,
    12: 1.40,
}




# SCHEMA
import polars as pl
import numpy as np
from datetime import date, timedelta
import math
import random
from faker import Faker
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Simulación de configuraciones globales (AJUSTAR SEGÚN TU ENTORNO) ---
# DIRS: Un diccionario que contiene rutas a directorios, e.g., {"OUTPUT": Path("./output"), "PARTS": Path("./parts")}
from pathlib import Path
DIRS = {"OUTPUT": Path("./output"), "PARTS": Path("./parts")}
DIRS["OUTPUT"].mkdir(parents=True, exist_ok=True)
DIRS["PARTS"].mkdir(parents=True, exist_ok=True)

FECHA_INICIO_PROYECTO = date(2021, 1, 1)
FECHA_FIN_PROYECTO = date(2026, 12, 31)
SEED_VAL = 42 # Semilla para reproducibilidad

# Esquemas de ejemplo (deberían estar definidos en tu archivo de configuración)
SCHEMAS = {
    "DimTiempo": pl.Schema({
        "ID_Tiempo": pl.Int32,
        "Fecha": pl.Date,
        "Anio": pl.Int16,
        "Mes": pl.Int8,
        "Dia": pl.Int8,
        "DiaSemana": pl.Int8,
        "Nombre_DiaSemana": pl.Utf8,
        "DiaAnio": pl.Int16,
        "SemanaAnioISO": pl.Int8,
        "DiaSemanaISO": pl.Int8,
        "Trimestre": pl.Int8,
        "AnioFiscal": pl.Int16,
        "MesFiscal": pl.Int8,
        "TrimestreFiscal": pl.Int8,
        "EsFinSemana": pl.Boolean,
        "EsFeriado": pl.Boolean,
        "Nombre_Feriado": pl.Utf8,
        "EsQuincena": pl.Boolean,
        "Factor_Estacionalidad_General": pl.Float32,
        "Factor_Estacionalidad_Mensual": pl.Float32,
        "Factor_Impacto_Feriado": pl.Float32,
    }),
    "DimGeografia": pl.Schema({
        "ID_Provincia": pl.Utf8,
        "Nombre_Provincia": pl.Utf8,
        "Region": pl.Utf8, # Zona Logistica
        "Poblacion_Estimada": pl.Int32,
        "Densidad_Poblacional": pl.Float32,
        "Latitud": pl.Float32,
        "Longitud": pl.Float32,
        "Peso_Normalizado": pl.Float32,
        "Nivel_Socioeconomico": pl.Utf8,
        "Activo": pl.Boolean,
    }),
    "DimPlanta": pl.Schema({
        "ID_Planta": pl.Utf8,
        "Nombre_Planta": pl.Utf8,
        "Tipo_Planta": pl.Utf8,
        "Ubicacion_Municipio": pl.Utf8,
        "Provincia": pl.Utf8,
        "Latitud": pl.Float32,
        "Longitud": pl.Float32,
        "Capacidad_Produccion_LtsDia": pl.Int64,
        "Fecha_Inicio_Operaciones": pl.Date,
        "Estado_Operativo": pl.Utf8,
        "Certificaciones": pl.Utf8,
    }),
    
    "DimAlmacen": pl.Schema({
        "ID_Almacen": pl.Utf8,
        "ID_Planta": pl.Utf8,
        "Nombre_Almacen": pl.Utf8,
        "Tipo_Almacen": pl.Utf8,
        "Capacidad_M3": pl.Int32,
        "Capacidad_Pallets": pl.Int32,
        "Tiene_Refrigeracion": pl.Boolean,
        "Latitud": pl.Float32,
        "Longitud": pl.Float32,
        "Estado_Operativo": pl.Utf8,
    }),
    "DimDepartamento": pl.Schema({
        "Departamento_ID": pl.Utf8,
        "Nombre_Departamento": pl.Utf8,
        "Tipo_Departamento": pl.Utf8,
        "Nivel_Organizacional": pl.Utf8,
        "Presupuesto_Anual_Estimado_DOP": pl.Float64,
        "Numero_Empleados_Estimado": pl.Int32,
        "Objetivo_Principal": pl.Utf8,
    }),
    "DimPuesto": pl.Schema({
        "Puesto_ID": pl.Utf8,
        "Departamento_ID": pl.Utf8,                 # FK → DimDepartamento.Departamento_ID
        "Nombre_Puesto": pl.Utf8,
        "Nivel_Puesto": pl.Utf8,
        "Salario_Base_Mensual_Min_DOP": pl.Float64,
        "Salario_Base_Mensual_Max_DOP": pl.Float64,
        "Salario_Base_Mensual_DOP": pl.Float64,     # promedio de referencia
        "Es_Comercial": pl.Boolean,
    }),
    "DimCEDIS": pl.Schema({
        "CEDI_ID": pl.Utf8,
        "Nombre_CEDI": pl.Utf8,
        "Tipo_CEDI": pl.Utf8,
        "Region_Operacion": pl.Utf8,
        "ID_Provincia": pl.Utf8,          # o Geografia_ID si prefieres
        "Planta_ID": pl.Utf8,             # ID_Planta_Asociada
        "Latitud": pl.Float32,
        "Longitud": pl.Float32,
        "Capacidad_Pallets": pl.Int32,
        "Estado_Operativo": pl.Utf8,
    }),
    
    "DimProducto": pl.Schema({
        "ID_ProductoSKU": pl.Utf8,          # Clave del SKU en la maestra
        "Nombre_Producto": pl.Utf8,
        "Marca": pl.Utf8,
        "Sabor": pl.Utf8,
        "Categoria": pl.Utf8,               # Desde Categoria_Maestra
        "Categoria_Global": pl.Utf8,        # Mapeada a CATEGORIAS_PRODUCTO
        "Volumen_Litros": pl.Float32,
        "Tipo_Envase": pl.Utf8,
        "Unidades_Por_Caja": pl.Int32,
        "Precio_Lista_DOP": pl.Float32,
        "Costo_Prod_DOP": pl.Float32,
        "Peso_Venta": pl.Float32,           # Peso relativo de venta
        "Aplica_ISC": pl.Boolean,
        "Tasa_ISC_Pct": pl.Float32,
        "Factor_Estacionalidad_Categoria": pl.Float32,
        "Activo": pl.Boolean,
        "Peso_Venta_Normalizado": pl.Float32,
    }),

    "DimCanalDistribucion": pl.Schema({
        "ID_Canal": pl.Utf8,                 # Clave surrogate del canal
        "Nombre_Canal": pl.Utf8,            # Nombre legible (ej. Colmados)
        "Peso_Mercado": pl.Float32,         # Peso relativo antes de normalizar
        "Peso_Mercado_Normalizado": pl.Float32,  # Peso_Mercado / total
        "Segmentos_Objetivo": pl.Utf8,      # Lista de segmentos en CSV
        "Es_Ticket_Bajo": pl.Boolean,       # Flag según CANALES_RD
        "Estado": pl.Utf8,                  # 'Activo', etc.
        "Peso_Mercado_Normalizado": pl.Float32,  # Peso_Mercado / total
    }),    
    "DimCluster": pl.Schema({
        "Cluster_ID": pl.Int8,           # 1–4 nivel de cluster
        "Nombre_Cluster": pl.Utf8,       # 'VIP - Estratégico', etc.
        "Descripcion": pl.Utf8,          # Texto descriptivo
        "Nivel_Prioridad": pl.Int8,      # 1 = máxima prioridad
    }),
    
    "DimCliente": pl.Schema({
        "ID_Cliente": pl.Utf8,          # Clave surrogate cliente
        "Nombre_Cliente": pl.Utf8,
        "ID_Provincia": pl.Utf8,        # FK → DimGeografia.ID_Provincia
        "ID_Canal": pl.Utf8,            # FK → DimCanalDistribucion.ID_Canal
        "Segmento_Cliente": pl.Utf8,    # A/B/C/D/E
        "Cluster_ID": pl.Int8,          # FK → DimCluster.Cluster_ID
        "Fecha_Alta": pl.Date,
        "Activo": pl.Boolean,
        "Latitud": pl.Float32,
        "Longitud": pl.Float32,
        "Ano_Creacion": pl.Int16,
    }),
    
    "DimEmpleado": pl.Schema({
        "Empleado_ID": pl.Utf8,                 # PK empleado
        "Nombre_Completo": pl.Utf8,
        "Departamento_ID": pl.Utf8,             # FK -> DimDepartamento.Departamento_ID
        "Puesto_ID": pl.Utf8,                   # FK -> DimPuesto.Puesto_ID
        "CEDI_ID": pl.Utf8,                     # FK opcional -> DimCEDIS.CEDI_ID (nullable)
        "Provincia_ID_Residencia": pl.Utf8,     # FK -> DimGeografia.ID_Provincia
        "Fecha_Contratacion": pl.Date,
        "Salario_Base_Mensual_DOP": pl.Float32,
        "Estatus_Empleado": pl.Utf8,            # "Activo", etc.
        "Email_Corporativo": pl.Utf8,
        "Telefono_Contacto": pl.Utf8,
        "Fecha_Nacimiento": pl.Date,
        "Genero": pl.Utf8,                      # "Masculino"/"Femenino"/"Otro"
        "Experiencia_Anios": pl.Int8,
        "Tipo_Contrato": pl.Utf8,  
    }), 
    
    "DimVendedor": pl.Schema({
        "Vendedor_ID": pl.Utf8,                         # PK (mismo que Empleado_ID)
        "Empleado_ID": pl.Utf8,                         # FK → DimEmpleado.Empleado_ID
        "Puesto_ID": pl.Utf8,                           # FK → DimPuesto.Puesto_ID
        "Nombre_Vendedor": pl.Utf8,
        "CEDI_Base_ID": pl.Utf8,                        # FK → DimCEDIS.CEDI_ID
        "Tipo_Vendedor": pl.Utf8,
        "Enfoque_Canal": pl.Utf8,
        "Meta_Venta_Mensual_DOP": pl.Float32,
        "Porcentaje_Comision_Objetivo": pl.Float32,
        "Telefono_Flota": pl.Utf8,
        "Nivel_Experiencia": pl.Utf8,
        "Fecha_Asignacion_Ruta": pl.Date,
        "Estado_Vendedor": pl.Utf8,
        "Gerente_Directo_ID": pl.Utf8,                 # FK opcional → Vendedor_ID
        "Promedio_Clientes_Visitados_Dia": pl.Int16,
        "Es_Supervisor_Gerente": pl.Boolean,
    }),
    
    "DimVehiculo": pl.Schema({
        "ID_Vehiculo": pl.Utf8,                      # VEH-0001...
        "CEDI_Asignado_ID": pl.Utf8,                 # FK → DimCEDI.ID_CEDI
        "Placa": pl.Utf8,
        "Marca_Modelo": pl.Utf8,
        "Tipo_Vehiculo": pl.Utf8,                    # Camión Interurbano, Furgoneta, etc.
        "Capacidad_Carga_Ton": pl.Float32,
        "Capacidad_Volumen_M3": pl.Float32,
        "Rendimiento_Promedio_KmL": pl.Float32,
        "Costo_Fijo_Operativo_Diario_DOP": pl.Float32,
        "Uso_Principal": pl.Utf8,                    # Rutas Largas, Urbano, etc.
        "Anio_Fabricacion": pl.Int16,
        "Kilometraje_Actual_KM": pl.Int64,
        "Estado_Vehiculo": pl.Utf8,                  # Operativo, En Taller, Baja
        "Tiene_GPS": pl.Boolean,
        "Valor_Adquisicion_DOP": pl.Float32,
        "Depreciacion_Anual_Pct": pl.Float32,
    }),
    
    "DimRuta": pl.Schema({
        "ID_Ruta": pl.Utf8,                     # RUT-00001...
        "Nombre_Ruta": pl.Utf8,
        "ID_CEDI_Origen": pl.Utf8,             # FK → DimCEDI.ID_CEDI
        "Nombre_CEDI_Origen": pl.Utf8,
        "ID_Provincia_Destino": pl.Utf8,       # FK → DimGeografia.ID_Provincia
        "Nombre_Provincia_Destino": pl.Utf8,
        "Zona_Especifica": pl.Utf8,            # city_suffix + street_name
        "ID_Vehiculo_Asignado": pl.Utf8,       # FK → DimVehiculo.ID_Vehiculo
        "Marca_Modelo_Vehiculo": pl.Utf8,
        "ID_Vendedor_Titular": pl.Utf8,        # FK → DimVendedor.ID_Vendedor
        "Nombre_Vendedor_Titular": pl.Utf8,
        "Tipo_Vendedor_Ruta": pl.Utf8,         # Preventista / Autoventa / etc.
        "Distancia_Ruta_KM": pl.Float32,       # Solo ida
        "Tiempo_Ruta_Estimado_Hrs": pl.Float32,# Solo ida
        "Costo_Peaje_Estimado_DOP": pl.Float32,
        "Frecuencia_Visita": pl.Utf8,          # Diaria, Interdiaria, Semanal
        "Dias_Operacion_Semana": pl.Utf8,      # 'L-M-M-J-V-S', etc.
        "Tipo_Ruta_Geografica": pl.Utf8,       # Urbana Densa, Autopista, etc.
        "Estado_Ruta": pl.Utf8,                # Activa, Inactiva
    }),
    
    "DimPromocion": pl.Schema({
        "ID_Promocion": pl.Utf8,              # PROM-000, PROM-001...
        "Nombre_Promocion": pl.Utf8,
        "Descripcion": pl.Utf8,
        "Factor_Incremento_Venta": pl.Float32,   # 1.0 + Peso_Incremento_Venta
        "Peso_Probabilidad_Uso": pl.Float32,     # ~%_Peso_Incremento / 100
        "Activa": pl.Boolean,
    }),
    "DimFactura": pl.Schema({
        "ID_Factura": pl.Utf8,
        "ID_Tiempo_Factura": pl.Utf8,  # o pl.Date si prefieres opción B
        "ID_Cliente": pl.Utf8,
        "Vendedor_ID": pl.Utf8,
        "ID_CEDI_Origen": pl.Utf8,
        "ID_Ruta": pl.Utf8,
        "Tipo_Pago": pl.Utf8,
        "Medio_Pago": pl.Utf8,
        "Estado_Factura": pl.Utf8,
        "Fecha_Creacion": pl.Date,
        "Fecha_Vencimiento": pl.Date,
        "Fecha_Pago": pl.Date,
        "Total_Factura_Bruto_DOP": pl.Float32,
        "Total_Descuento_DOP": pl.Float32,
        "Total_Impuesto_ISC_DOP": pl.Float32,
        "Total_Impuesto_ITBIS_DOP": pl.Float32,
        "Total_Factura_Neto_DOP": pl.Float32,
        "Total_Factura_Pagar_DOP": pl.Float32,
        "Observaciones": pl.Utf8,
        "Es_Factura_Recurrente": pl.Boolean,
    }), 
    
    "DimProveedor": pl.Schema({
        "ID_Proveedor": pl.Utf8,
        "Nombre_Proveedor": pl.Utf8,
        "Tipo_Proveedor": pl.Utf8,          # Materia Prima, Empaque, Logística, etc.
        "Pais_Origen": pl.Utf8,
        "Ciudad_Origen": pl.Utf8,
        "Fecha_Inicio_Relacion": pl.Date,
        "Estatus_Proveedor": pl.Utf8,       # Activo / Inactivo
        "Terminos_Pago_Dias": pl.Int16,
        "Contacto_Principal": pl.Utf8,
        "Email_Contacto": pl.Utf8,
        "Telefono_Contacto": pl.Utf8,
    }),
    
    "DimActivoFijo":pl.Schema({
        "ID_Activo": pl.Utf8,
        "Nombre_Activo": pl.Utf8,
        "Tipo_Activo": pl.Utf8,               # Maquinaria, Edificio, Equipo Oficina, Mobiliario, Terreno
        "ID_Planta_Ubicacion": pl.Utf8,       # Puede ser nulo si es en CEDI u oficina central
        "ID_CEDI_Ubicacion": pl.Utf8,         # Puede ser nulo si es en planta u oficina central
        "Fecha_Adquisicion": pl.Date,
        "Vida_Util_Anios": pl.Int16,
        "Costo_Adquisicion_DOP": pl.Float64,
        "Valor_Residual_DOP": pl.Float64,
        "Depreciacion_Acumulada_DOP": pl.Float64,
        "Valor_Neto_Libros_DOP": pl.Float64,
        "Estatus_Activo": pl.Utf8,            # Operativo, En Mantenimiento, Fuera de Servicio
        "Numero_Serie": pl.Utf8,
    }),

    "FactVentas":pl.Schema({
        "ID_Venta_Transaccion": pl.Utf8,
        "ID_Factura": pl.Utf8,
        "Fecha_Transaccion": pl.Date,
        "ID_Tiempo": pl.Utf8, # FK a DimTiempo
        "ID_Cliente": pl.Utf8,
        "Vendedor_ID": pl.Utf8,
        "ID_CEDI_Origen": pl.Utf8, # Nuevo: CEDI que despacha
        "ID_Ruta": pl.Utf8,
        "ID_Vehiculo": pl.Utf8,
        "Codigo_Producto_SKU": pl.Utf8,
        "ID_Promocion": pl.Utf8, # Nuevo: Promoción aplicada
        "ID_Canal": pl.Utf8, # Nuevo: Canal de venta
        "ID_Provincia": pl.Utf8, # Nuevo: Provincia de la venta
        "Cantidad_Unidades": pl.Int32,
        "Precio_Unitario_DOP": pl.Float32, # Precio antes de descuento
        "Precio_Final_DOP": pl.Float32, # Precio por unidad después de descuento y promoción
        "Descuento_Pct": pl.Float32,
        "Impuesto_ISC_Pct": pl.Float32, # Nuevo: Impuesto Selectivo al Consumo
        "Impuesto_ITBIS_Pct": pl.Float32, # Nuevo: Impuesto sobre Transferencias de Bienes Industrializados y Servicios (ITBIS)
        "Monto_Descuento_DOP": pl.Float32, # Nuevo: Monto total del descuento por línea
        "Monto_Impuesto_ISC_DOP": pl.Float32, # Nuevo: Monto total ISC por línea
        "Monto_Impuesto_ITBIS_DOP": pl.Float32, # Nuevo: Monto total ITBIS por línea
        "Ingreso_Bruto_DOP": pl.Float32, # Cantidad * Precio Unitario (antes de descuentos e impuestos)            "Ingreso_Neto_DOP": pl.Float32, # Total facturado al cliente (después de descuentos, antes de ITBIS)
        "Costo_Venta_Total_DOP": pl.Float32,
        "Margen_Bruto_DOP": pl.Float32,
        "Tipo_Pago": pl.Utf8,
        "Medio_Pago": pl.Utf8,
        "Estado_Factura": pl.Utf8,
        "Latitud_Entrega": pl.Float32, # Nuevo: Latitud del cliente
        "Longitud_Entrega": pl.Float32, # Nuevo: Longitud del cliente
        "Tipo_Venta": pl.Utf8, # Nuevo: Preventa, Autoventa, Directa
        "Ticket_Promedio_Cliente": pl.Float32 # Nuevo: Estimación de ticket promedio de ese cliente para la fecha
    }),
    
    "FactProyecciones":pl.Schema({
        "ID_Tiempo": pl.Utf8,
        "ID_ProductoSKU": pl.Utf8,
        "ID_Canal": pl.Utf8,
        "ID_Provincia": pl.Utf8,
        "Valor_Proyectado_Venta_DOP": pl.Float32,
        "Cantidad_Proyectada_Unidades": pl.Int32,
        "Anio_Proyeccion": pl.Int16,
    }), 
    
    "FactPlanProduccion":pl.Schema({
        "ID_Tiempo": pl.Utf8,
        "ID_Planta": pl.Utf8,
        "ID_ProductoSKU": pl.Utf8,
        "Cantidad_Planeada_Produccion_Unidades": pl.Int32,
        "Costo_Estimado_Produccion_DOP": pl.Float32,
    }),
    "FactCompraMateriaPrima":pl.Schema({
        "ID_Proveedor": pl.Utf8,
        "ID_Planta": pl.Utf8,
        "Materia_Prima": pl.Utf8,
        "Cantidad_Comprada_Kg": pl.Float32,
        "Costo_Total_DOP": pl.Float32,
        "Numero_Orden_Compra": pl.Utf8,
    }),
    "FactInventario":pl.Schema({
        "ID_Tiempo": pl.Utf8,
        "ID_ProductoSKU": pl.Utf8,
        "ID_Ubicacion": pl.Utf8,        # ID_Planta o ID_CEDI
        "Tipo_Ubicacion": pl.Utf8,      # "Planta" o "Almacen/CEDI"
        "Cantidad_Entrada_Unidades": pl.Int32,
        "Cantidad_Salida_Unidades": pl.Int32,
        "Stock_Final_Unidades": pl.Int32,
        "Valor_Inventario_DOP": pl.Float32,
    }),
    "FactLogistica":pl.Schema({
        "ID_Tiempo_Viaje": pl.Utf8,
        "ID_Ruta": pl.Utf8,
        "ID_Vehiculo": pl.Utf8,
        "ID_CEDI_Origen": pl.Utf8,
        "ID_Empleado_Chofer": pl.Utf8,
        "Distancia_Real_Recorrida_KM": pl.Float32,
        "Consumo_Combustible_Galones": pl.Float32,
        "Costo_Combustible_DOP": pl.Float32,
        "Entregas_Programadas": pl.Int16,
        "Entregas_Exitosas": pl.Int16,
        "Entregas_Fallidas": pl.Int16,
        "Tiempo_Viaje_Horas": pl.Float32,
    }),
    "FactIncidenteOperativo":pl.Schema({
        "ID_Incidente": pl.Utf8,
        "ID_Tiempo_Incidente": pl.Utf8,
        "Tipo_Incidente": pl.Utf8,
        "Area_Operativa": pl.Utf8, # Producción, Logística, Ventas, Almacén
        "ID_Entidad_Afectada": pl.Utf8, # ID_Planta, ID_Vehiculo, etc.
        "Descripcion_Incidente": pl.Utf8,
        "Impacto_Costo_DOP": pl.Float32,
        "Impacto_Tiempo_Horas": pl.Float32,
        "Nivel_Severidad": pl.Utf8, # Bajo, Medio, Alto, Crítico
    }),
    "FactSostenibilidad":pl.Schema({
        "ID_Tiempo_Mes": pl.Utf8, # Métricas mensuales
        "ID_Ubicacion": pl.Utf8, # Planta o CEDI
        "Tipo_Ubicacion": pl.Utf8,
        "Consumo_Agua_M3": pl.Float32,
        "Consumo_Energia_KWh": pl.Float32,
        "Generacion_Residuos_Kg": pl.Float32,
        "Emisiones_CO2_Ton": pl.Float32,
    }),
    "FactContabilidadGeneral":pl.Schema({
        "ID_Asiento": pl.Utf8,
        "ID_Tiempo_Contable": pl.Date,
        "Tipo_Transaccion": pl.Utf8,   # Ventas, Compras, Nómina, Gastos, Depreciación
        "Modulo_Origen": pl.Utf8,
        "ID_Documento_Origen": pl.Utf8,    # ID_Factura, Numero_Orden_Compra, etc.
        "Cuenta_Contable": pl.Utf8,
        "Descripcion_Asiento": pl.Utf8,
        "Monto_Debito_DOP": pl.Float32,
        "Monto_Credito_DOP": pl.Float32,
        "Centro_Costo": pl.Utf8,
    }),
    
    # ... otros esquemas
}

# Asegurar que DB_MEMORIA exista (para DB_MEMORIA["FactVentasAvanzada"])
if 'DB_MEMORIA' not in globals():
    DB_MEMORIA = {}


# Función para guardar Parquet (Placeholder si no está definida globalmente)
if 'guardar_parquet' not in globals():
    def guardar_parquet(df: pl.DataFrame, name: str):
        path = DIRS["OUTPUT"] / f"{name}.parquet"
        df.write_parquet(path, compression="zstd")
        logger.info(f"    💾 Guardado {name}.parquet en {path}")

# Función auxiliar para asegurar columnas y tipos
def asegurar_columnas(df: pl.DataFrame, schema: pl.Schema, valores_defecto: dict = None) -> pl.DataFrame:
    if valores_defecto is None:
        valores_defecto = {}
    
    for col, dtype in schema.items():
        if col not in df.columns:
            default_value = valores_defecto.get(col)
            if default_value is None:
                if dtype == pl.Utf8:
                    default_value = ""
                elif dtype == pl.Int8 or dtype == pl.Int16 or dtype == pl.Int32 or dtype == pl.Int64:
                    default_value = 0
                elif dtype == pl.Float32 or dtype == pl.Float64:
                    default_value = 0.0
                elif dtype == pl.Date:
                    default_value = date(1900, 1, 1) # Fecha por defecto o None si el esquema lo permite
                elif dtype == pl.Boolean:
                    default_value = False
            
            df = df.with_columns(pl.lit(default_value, dtype=dtype).alias(col))
        else:
            # Intentar castear si el tipo no coincide, pero solo si es compatible o es None
            if df[col].dtype != dtype:
                try:
                    df = df.with_columns(pl.col(col).cast(dtype))
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo castear la columna '{col}' de {df[col].dtype} a {dtype}: {e}")
                    # Si el casteo falla y la columna puede ser nula, la dejamos como está o la llenamos con nulos.
                    # Para este contexto, asumiremos que el esquema final es el que importa.
    
    # Seleccionar y reordenar columnas según el esquema
    df = df.select([col for col in schema.keys() if col in df.columns])
    
    return df




# GRUPO 1 -- FUNCIONES GENERADORAS --
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# 1. DimTiempo (Calendario Fiscal y Logístico)
# --------------------------------------------------------------------
FECHA_INICIO_PROYECTO = date(2021, 1, 1)
FECHA_FIN_PROYECTO = date(2025, 12, 31)


def generar_dim_tiempo() -> pl.LazyFrame:
    logger.info("    📅 Generando DimTiempo (reconstruida)...")
    fechas = pl.date_range(FECHA_INICIO_PROYECTO, FECHA_FIN_PROYECTO, interval="1d", eager=True)

    # Paso 1: columnas base
    df = pl.DataFrame({"Fecha": fechas}).with_columns([
        pl.col("Fecha").dt.year().cast(pl.Int16).alias("Anio"),
        pl.col("Fecha").dt.month().cast(pl.Int8).alias("Mes"),
        pl.col("Fecha").dt.day().cast(pl.Int8).alias("Dia"),
        pl.col("Fecha").dt.weekday().cast(pl.Int8).alias("DiaSemana"),
        pl.col("Fecha").dt.strftime("%A").alias("Nombre_DiaSemana"),
        pl.col("Fecha").dt.ordinal_day().cast(pl.Int16).alias("DiaAnio"),
        pl.col("Fecha").dt.strftime("%V").cast(pl.Int8).alias("SemanaAnioISO"),
        pl.col("Fecha").dt.strftime("%u").cast(pl.Int8).alias("DiaSemanaISO"),
        pl.col("Fecha").dt.quarter().cast(pl.Int8).alias("Trimestre"),
        pl.col("Fecha").dt.strftime("%Y%m%d").cast(pl.Int32).alias("ID_Tiempo"),
        (pl.col("Fecha").dt.weekday() >= 5).alias("EsFinSemana"),
        (pl.col("Fecha").dt.day().is_in([15, 30, 28, 29, 31])).alias("EsQuincena"),
    ])

    # Paso 2: año fiscal y mes fiscal
    df = df.with_columns([
        pl.when(pl.col("Mes") >= 7)
          .then(pl.col("Anio").cast(pl.Int16))
          .otherwise((pl.col("Anio") - 1).cast(pl.Int16))
          .alias("AnioFiscal"),
        pl.when(pl.col("Mes") >= 7)
          .then(pl.col("Mes") - 6)
          .otherwise(pl.col("Mes") + 6)
          .cast(pl.Int8)
          .alias("MesFiscal"),
    ])

    # Paso 3: trimestre fiscal
    df = df.with_columns([
        pl.when(pl.col("MesFiscal").is_between(1, 3)).then(1)
         .when(pl.col("MesFiscal").is_between(4, 6)).then(2)
         .when(pl.col("MesFiscal").is_between(7, 9)).then(3)
         .otherwise(4)
         .cast(pl.Int8)
         .alias("TrimestreFiscal"),
    ])

    # Feriados
    feriados_df = pl.DataFrame({
        "Fecha": list(FERIADOS_RD.keys()),
        "Nombre_Feriado": list(FERIADOS_RD.values()),
        "EsFeriado": True,
    })
    df = df.join(feriados_df, on="Fecha", how="left").with_columns([
        pl.col("EsFeriado").fill_null(False),
        pl.col("Nombre_Feriado").fill_null("No Feriado"),
    ])

    # Estacionalidad mensual base
    df = df.with_columns(
        pl.col("Mes")
          .replace(FACTOR_ESTACIONALIDAD_MENSUAL)
          .cast(pl.Float32)
          .alias("Factor_Estacionalidad_Mensual")
    )

    # Impacto feriado (numpy)
    fechas_eager    = df["Fecha"].to_numpy()
    feriados_eager  = df["EsFeriado"].to_numpy()
    impacto_feriado_arr = np.ones(len(fechas_eager), dtype=np.float32)

    for i, is_f in enumerate(feriados_eager):
        if is_f:
            impacto_feriado_arr[i] *= IMPACTO_FERIADO["dia_feriado"]
            if i > 0 and not feriados_eager[i - 1]:
                impacto_feriado_arr[i - 1] *= IMPACTO_FERIADO["dia_antes"]
            if i < len(fechas_eager) - 1 and not feriados_eager[i + 1]:
                impacto_feriado_arr[i + 1] *= IMPACTO_FERIADO["dia_despues"]

    df = df.with_columns(
        pl.Series("Factor_Impacto_Feriado", impacto_feriado_arr, dtype=pl.Float32)
    )

    # Estacionalidad general combinada
    df = df.with_columns(
        (pl.col("Factor_Estacionalidad_Mensual") * pl.col("Factor_Impacto_Feriado"))
        .cast(pl.Float32)
        .alias("Factor_Estacionalidad_General")
    )

    # Ajustar a schema DimTiempo
    if "DimTiempo" in SCHEMAS:
        schema = SCHEMAS["DimTiempo"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]

   # logger.info( "🔎 Sample de DimTiempo:\n" + df.head(10).to_pandas().to_string(index=False))

    guardar_parquet(df, "dim_tiempo")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_tiempo.parquet")


# --------------------------------------------------------------------
# 2. DimGeografia (Completa con todos los atributos)
# --------------------------------------------------------------------
def generar_dim_geografia() -> pl.LazyFrame:
    logger.info("    🌍 Generando DimGeografia (Completa con todos los atributos)...")
    
    # 0. Cargar datos base (ajusta el rename si tu clave es distinta)
    df = pl.DataFrame(PROVINCIAS_FLAT)
    # Si tu diccionario trae 'Nombre_Provincia', descomenta:
    # df = df.rename({"Nombre_Provincia": "Provincia"})
    
    # 1. Renormalizar pesos
    total_peso = df["Peso"].sum()
    df = df.with_columns(
        (pl.col("Peso") / total_peso).alias("Peso_Normalizado")
    )
    
    # 2. Generar ID y métricas básicas
    df = df.with_row_index("idx", offset=1).with_columns([
        (pl.lit("DO-") + pl.col("idx").cast(pl.Utf8).str.zfill(2)).alias("ID_Provincia"),
        (pl.col("Poblacion_Estimada") / pl.col("Area_Km2"))
            .round(1)
            .cast(pl.Float32)
            .alias("Densidad_Poblacional"),
        pl.col("Lat").cast(pl.Float32).alias("Latitud"),
        pl.col("Lon").cast(pl.Float32).alias("Longitud"),
        pl.lit(True).alias("Activo"),
    ])
    
    # 3. Asignar nivel socioeconómico predominante
    rng = np.random.default_rng(SEED_VAL)
    nse_asignado: list[str] = []
    
    for row in df.iter_rows(named=True):
        region = row["Region"]
        pesos_region = PESOS_NIVEL_SOCIOECONOMICO.get(
            region,
            PESOS_NIVEL_SOCIOECONOMICO["Ozama"],
        )
        niveles = list(pesos_region.keys())
        probs = np.array(list(pesos_region.values()), dtype=float)
        probs = probs / probs.sum()
        nse = rng.choice(niveles, p=probs)
        nse_asignado.append(nse)
    
    df = df.with_columns(
        pl.Series("Nivel_Socioeconomico", nse_asignado)
    )
    
    # 4. Selección final
    df_final = df.select([
        "ID_Provincia",
        "Nombre_Provincia",
        "Region",
        "Poblacion_Estimada",
        "Densidad_Poblacional",
        "Latitud",
        "Longitud",
        "Peso_Normalizado",
        "Nivel_Socioeconomico",
        "Activo",
    ])
    
    # 5. Ajustar y verificar schema si existe
    if "DimGeografia" in SCHEMAS:
        schema = SCHEMAS["DimGeografia"]
        df_final = asegurar_columnas(df_final, schema)
        df_final = df_final.cast(dict(schema))  # type: ignore[arg-type]
    
    guardar_parquet(df_final, "dim_geografia")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_geografia.parquet")

# --------------------------------------------------------------------
# 3. DimPlanta (Única Planta Principal)
# --------------------------------------------------------------------
def generar_dim_planta():
    logger.info("    🏭 Generando DimPlanta (Única Planta Principal)...")
    
    # Datos de la planta única basados en la información del proyecto
    planta_data = [{
        "ID_Planta": "PLN-01",
        "Nombre_Planta": "Planta Principal Bepensa RD",
        "Tipo_Planta": "Embotelladora y Producción Multicategoría",
        # Ubicación realista en zona industrial Haina
        "Ubicacion_Municipio": "La Feria",
        "Provincia": "Santo Domingo",
        "Latitud": 18.44742855000376,
        "Longitud": -69.93012206251315,
        # Capacidad estimada para soportar el volumen anual de cajas del TXT (aprox 90M cajas/año)
        # Suponiendo 300 días operativos y promedio 5.6 L/caja -> ~1.7M Litros/día
        "Capacidad_Produccion_LtsDia": 1_800_000, 
        "Fecha_Inicio_Operaciones": date(2005, 1, 1), # Fecha histórica supuesta
        "Estado_Operativo": "Activa",
        "Certificaciones": "ISO 9001, FSSC 22000, ISO 14001"
    }]

    df = pl.DataFrame(planta_data)

    # Asegurar tipos (Asumiendo SCHEMAS["DimPlanta"] definido por ti)
    # df = df.cast(SCHEMAS["DimPlanta"])
    if "DimPlanta" in SCHEMAS:
        schema = SCHEMAS["DimPlanta"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]
        
    #logger.info("🔎 Sample de DimPlanta:\n" + df.head(1).to_pandas().to_string(index=False))

    guardar_parquet(df, "dim_planta")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_planta.parquet")

# --------------------------------------------------------------------
# 4. DimAlmacen (Almacén Central de Planta)
# --------------------------------------------------------------------
def generar_dim_almacen_planta(lf_planta: pl.LazyFrame) -> pl.LazyFrame:
    logger.info("    🏭 Generando DimAlmacen (Almacén Central de Planta)...")
    
    # 1. Obtener datos de la planta (única)
    df_planta = lf_planta.collect()
    id_planta = df_planta["ID_Planta"][0]
    nombre_planta = df_planta["Nombre_Planta"][0]
    lat_planta = float(df_planta["Latitud"][0])
    lon_planta = float(df_planta["Longitud"][0])

    # 2. Datos del almacén central adjunto a la planta
    almacen_data = [{
        "ID_Almacen": "ALM-PLN-01",
        "ID_Planta": id_planta,  # FK a DimPlanta, nombre alineado al mapa maestro
        "Nombre_Almacen": f"Almacén Central - {nombre_planta}",
        "Tipo_Almacen": "Centro de Distribución Principal (Anexo Planta)",
        "Capacidad_M3": 150_000,      # capacidad en m³ (según mapa maestro)
        "Capacidad_Pallets": 45_000,  # atributo extra útil para logística
        "Tiene_Refrigeracion": True,
        "Latitud": lat_planta,
        "Longitud": lon_planta,
        "Estado_Operativo": "Activo",
    }]

    df = pl.DataFrame(almacen_data)

    # 3. Ajustar y verificar schema DimAlmacen
    if "DimAlmacen" in SCHEMAS:
        schema = SCHEMAS["DimAlmacen"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]

        """logger.info(f"Schema DimAlmacen esperado : {schema}")
        logger.info(f"Schema DimAlmacen obtenido: {df.schema}")
        assert df.schema == schema, "Schema de DimAlmacen no coincide con SCHEMAS['DimAlmacen']"

    logger.info(
        "🔎 Sample de DimAlmacen:\n"
        + df.head(5).to_pandas().to_string(index=False)
    )"""

    guardar_parquet(df, "dim_almacen")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_almacen.parquet")


# --------------------------------------------------------------------
# 5. DimDepartamento (basado en estructura RRHH)
# -------------------------------------------------------------------
def generar_dim_departamento() -> pl.LazyFrame:
    logger.info("    🏢 Generando DimDepartamento (basado en estructura RRHH)...")
    
    deptos_list = list(DEPARTAMENTOS_RRHH.keys())
    
    data_deptos = []
    for i, depto_key in enumerate(deptos_list):
        nombre_legible = (
            depto_key
            .replace("_", " ")
            .replace("IT_DataStrategy", "IT & Data Strategy")
        )
        
        tipo_departamento = (
            "Operativo" if any(x in depto_key for x in ["Logis", "Planta", "Seguridad", "Servicios"])
            else "Comercial" if any(x in depto_key for x in ["Ventas", "Marketing"])
            else "Administrativo"
        )
        
        nivel_org = (
            "Dirección" if "Direccion" in depto_key or "Gerencia" in depto_key
            else "Operativo"
        )
        
        data_deptos.append({
            "Departamento_ID": f"DEP-{str(i+1).zfill(2)}",
            "Nombre_Departamento": nombre_legible,
            "Tipo_Departamento": tipo_departamento,
            "Nivel_Organizacional": nivel_org,
            "Presupuesto_Anual_Estimado_DOP": float(random.randint(5_000_000, 80_000_000)),
            "Numero_Empleados_Estimado": random.randint(5, 200),
            "Objetivo_Principal": random.choice([
                "Eficiencia Operativa",
                "Crecimiento Ventas",
                "Reducción Costos",
                "Satisfacción Cliente",
                "Innovación",
            ]),
        })
    
    df = pl.DataFrame(data_deptos)

    if "DimDepartamento" in SCHEMAS:
        schema = SCHEMAS["DimDepartamento"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]
        #assert df.schema == schema, "Schema de DimDepartamento no coincide con SCHEMAS['DimDepartamento']"

    guardar_parquet(df, "dim_departamento")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_departamento.parquet")


# --------------------------------------------------------------------
# 6. DimPuesto (Catálogo de roles y salarios)
# --------------------------------------------------------------------
def generar_dim_puesto(lf_departamento: pl.LazyFrame) -> pl.LazyFrame:
    logger.info("    👔 Generando DimPuesto (Catálogo de roles y salarios)...")
    
    # 1. Traer DimDepartamento a DataFrame
    df_deptos = lf_departamento.collect()

    # 2. Mapa: Nombre_Departamento (legible) -> Departamento_ID
    mapa_depto_id = dict(
        zip(
            df_deptos["Nombre_Departamento"],   # ej: "Logistica Distribucion"
            df_deptos["Departamento_ID"],       # ej: "DEP-01"
        )
    )
    
    data_puestos = []
    puesto_counter = 1
    
    # 3. Recorrer estructura maestra DEPARTAMENTOS_RRHH
    for depto_key, roles in DEPARTAMENTOS_RRHH.items():
        # mismo nombre legible que usaste en generar_dim_departamento
        nombre_legible = (
            depto_key
            .replace("_", " ")
            .replace("IT_DataStrategy", "IT & Data Strategy")
        )
        id_depto = mapa_depto_id.get(nombre_legible)
        
        if id_depto is None:
            logger.warning(
                f"⚠️ Departamento '{depto_key}' no encontrado en DimDepartamento. "
                f"SALTANDO puestos de ese departamento."
            )
            continue
            
        for rol in roles:
            nombre_puesto = rol["Puesto"]

            # Nivel jerárquico simple por keywords
            if any(x in nombre_puesto for x in ["Gerente", "Director", "Jefe", "Vicepresidente"]):
                nivel = "Gerencial/Directivo"
            elif any(x in nombre_puesto for x in ["Coordinador", "Supervisor", "Especialista",
                                                  "Ingeniero", "Analista", "Abogado",
                                                  "Contador", "KAM"]):
                nivel = "Mando Medio/Especialista"
            else:
                nivel = "Operativo/Técnico"
            
            sueldo_min = float(rol.get("Sueldo_Min", 0))
            sueldo_max = float(rol.get("Sueldo_Max", 0))
            sueldo_prom = (sueldo_min + sueldo_max) / 2 if (sueldo_min or sueldo_max) else 0.0

            data_puestos.append({
                "Puesto_ID": f"PUE-{str(puesto_counter).zfill(3)}",
                "Departamento_ID": id_depto,
                "Nombre_Puesto": nombre_puesto,
                "Nivel_Puesto": nivel,
                "Salario_Base_Mensual_Min_DOP": sueldo_min,
                "Salario_Base_Mensual_Max_DOP": sueldo_max,
                "Salario_Base_Mensual_DOP": sueldo_prom,
                "Es_Comercial": bool("Ventas" in depto_key or "Marketing" in depto_key),
            })
            puesto_counter += 1

    df = pl.DataFrame(data_puestos)

    # 4. Ajustar y verificar schema DimPuesto (si está definido)
    if "DimPuesto" in SCHEMAS:
        schema = SCHEMAS["DimPuesto"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]

        """
        logger.info(f"Schema DimPuesto esperado : {schema}")
        logger.info(f"Schema DimPuesto obtenido: {df.schema}")
        assert df.schema == schema, "Schema de DimPuesto no coincide con SCHEMAS['DimPuesto']"

    logger.info(
        "🔎 Sample de DimPuesto:\n"
        + df.head(10).to_pandas().to_string(index=False)
    )"""

    guardar_parquet(df, "dim_puesto")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_puesto.parquet")

# --------------------------------------------------------------------
# 7. DimCEDIS (1 Principal + Regionales -- CENTRO DE DISTRIBUCION --)
# --------------------------------------------------------------------
def generar_dim_cedi(lf_geografia: pl.LazyFrame, lf_planta: pl.LazyFrame) -> pl.LazyFrame:
    logger.info("    🏭 Generando DimCEDIS (1 Principal + Regionales TXT)...")
    
    df_geo    = lf_geografia.collect()
    df_planta = lf_planta.collect()

    # Mapa Nombre_Provincia -> ID_Provincia
    mapa_provincia = dict(
        zip(df_geo["Nombre_Provincia"], df_geo["ID_Provincia"])
    )

    cedis_data = []

    # 1. CEDI principal asociado a la planta
    planta = df_planta.row(0, named=True)

    # Buscar provincia de la planta por nombre, si la tienes en DimPlanta
    nombre_prov_planta = "Distrito Nacional"  # o df_planta["Provincia"][0] si existe
    prov_planta = mapa_provincia.get(nombre_prov_planta, df_geo["ID_Provincia"][0])
    region_planta = df_geo.filter(pl.col("ID_Provincia") == prov_planta)["Region"][0]

    cedis_data.append({
        "CEDI_ID": "CEDI-PRIN-01",
        "Nombre_CEDI": f"CEDI Principal - {planta['Nombre_Planta']}",
        "ID_Provincia": prov_planta,
        "Planta_ID": planta["ID_Planta"],
        "Latitud": float(planta["Latitud"]),
        "Longitud": float(planta["Longitud"]),
        "Capacidad_Pallets": 45_000,
        "Tipo_CEDI": "Principal",
        "Region_Operacion": region_planta,
        "Estado_Operativo": "Activo",
    })

    # 2. CEDIs regionales (TXT)
    for cedi_txt in CEDIS:
        nombre_prov = cedi_txt.get("Nombre_Provincia")
        prov_id = mapa_provincia.get(nombre_prov)

        if prov_id is None:
            logger.warning(
                f"⚠️ Provincia '{nombre_prov}' del CEDI {cedi_txt['ID_CEDI']} "
                f"no encontrada en DimGeografia. Asignando provincia por defecto."
            )
            prov_id = df_geo["ID_Provincia"][0]

        region = df_geo.filter(pl.col("ID_Provincia") == prov_id)["Region"][0]

        cedis_data.append({
            "CEDI_ID": cedi_txt["ID_CEDI"],
            "Nombre_CEDI": cedi_txt["Nombre"],
            "ID_Provincia": prov_id,
            "Planta_ID": None,
            "Latitud": float(cedi_txt.get("Lat", 0.0)),
            "Longitud": float(cedi_txt.get("Lon", 0.0)),
            "Capacidad_Pallets": int(cedi_txt.get("Capacidad_Pallets", 2000)),
            "Tipo_CEDI": cedi_txt.get("Tipo_Almacen", "Regional").capitalize(),
            "Region_Operacion": region,
            "Estado_Operativo": cedi_txt.get("Estado_Operativo", "Activo"),
        })

    df = pl.DataFrame(cedis_data)

    if "DimCEDIS" in SCHEMAS:
        schema = SCHEMAS["DimCEDIS"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]
        
        """logger.info(f"Schema DimCEDIS esperado : {schema}")
        logger.info(f"Schema DimCEDIS obtenido: {df.schema}")
        assert df.schema == schema, "Schema de DimCEDIS no coincide con SCHEMAS['DimCEDIS']"
        """
    guardar_parquet(df, "dim_cedi")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_cedi.parquet")


# --------------------------------------------------------------------
# 8. DimProducto (incluyendo SKUs)
# --------------------------------------------------------------------
def generar_dim_producto_sku() -> pl.LazyFrame:
    logger.info("    🥤 Generando DimProducto (Catálogo SKU Consolidado)...")
    
    # 1. Cargar bases
    df_maestra = pl.DataFrame(PRODUCTOS_MAESTRA)
    df_bepensa = pl.DataFrame(PRODUCTOS_BEPENSA_BASE)
    
    # 2. Normalizar nombres de columnas
    df_maestra = df_maestra.rename({
        "Codigo_Producto_SKU": "ID_ProductoSKU",
        "Categoria": "Categoria_Maestra",
    })
    df_bepensa = df_bepensa.rename({
        "ID_Prod_Unico": "ID_Ref_Bepensa",
        "Nombre_Producto": "Nombre_Ref_Bepensa",
    })
    
    # 3. Estandarizar texto para join
    df_maestra = df_maestra.with_columns(
        pl.col("Marca").str.to_uppercase()
    )
    df_bepensa = df_bepensa.with_columns([
        pl.col("Marca").str.to_uppercase(),
        pl.col("Categoria_Principal").str.to_uppercase(),
    ])
    
    # 4. Mapeo manual de categorías MAESTRA -> BEPENSA_BASE
    cat_map = {
        "Refrescos": "CSD",
        "Refrescos_Lig": "CSD",
        "Jugos": "JUGO",
        "Agua_Embotellada": "AGUA",
        "Agua": "AGUA",
        "Energizantes": "ENERGIZANTE",
        "Isotónicos": "ISOTÓNICO",
        "Té": "TÉ LISTO PARA BEBER",
        "Lácteos": "BEBIDA VEGETAL",
        "Bebida Vegetal RTD": "BEBIDA VEGETAL",
        "Malta": "MALTA",
        "Agua Saborizada": "AGUA SABORIZADA",
    }
    
    df_maestra = df_maestra.with_columns(
        pl.col("Categoria_Maestra")
        .replace(cat_map)
        .fill_null("OTRO")
        .alias("Categoria_Join")
    )
    
    # 5. Left join para traer ISC
    df_consolidado = df_maestra.join(
        df_bepensa.select(["Marca", "Categoria_Principal", "Aplica_ISC", "Tasa_ISC_Pct"]),
        left_on=["Marca", "Categoria_Join"],
        right_on=["Marca", "Categoria_Principal"],
        how="left",
    )
    
    # Completar nulos de ISC
    df_consolidado = df_consolidado.with_columns([
        pl.col("Aplica_ISC").fill_null(False),
        pl.col("Tasa_ISC_Pct").fill_null(0.0).cast(pl.Float32),
    ])
    
    # 6. Mapeo a categorías globales (para estacionalidad)
    global_cat_map = {
        "Refrescos": "CSD (Gaseosas)",
        "Refrescos_Lig": "CSD (Gaseosas)",
        "Agua_Embotellada": "Agua Purificada",
        "Agua": "Agua Purificada",
        "Jugos": "Jugos/Néctares",
        "Energizantes": "Energizantes",
        "Isotónicos": "Isotónicos",
        "Té": "Té y RTD Té",
        "Lácteos": "Lácteos y Bebida Vegetal RTD",
        "Bebida Vegetal RTD": "Lácteos y Bebida Vegetal RTD",
        "Malta": "Malta",
        "Agua Saborizada": "Agua Saborizada",
    }
    
    df_consolidado = df_consolidado.with_columns(
        pl.col("Categoria_Maestra")
        .replace(global_cat_map)
        .fill_null("Otros")
        .alias("Categoria_Global")
    )
    
    # 7. Estacionalidad por categoría global
    def get_estacionalidad(cat_global: str) -> float:
        return ESTACIONALIDAD_CATEGORIA.get(cat_global, 1.0)
    
    df_consolidado = df_consolidado.with_columns(
        pl.col("Categoria_Global")
        .map_elements(get_estacionalidad, return_dtype=pl.Float32)
        .alias("Factor_Estacionalidad_Categoria")
    )
    
    # 8. Selección y casteo final
    df_final = df_consolidado.select([
        "ID_ProductoSKU",
        "Nombre_Producto",
        "Marca",
        "Sabor",
        pl.col("Categoria_Maestra").alias("Categoria"),
        "Categoria_Global",
        "Volumen_Litros",
        "Tipo_Envase",
        "Unidades_Por_Caja",
        "Precio_Lista_DOP",
        "Costo_Prod_DOP",
        "Peso_Venta",
        "Aplica_ISC",
        "Tasa_ISC_Pct",
        "Factor_Estacionalidad_Categoria",
        pl.lit(True).alias("Activo"),
    ])
    
    df_final = df_final.with_columns([
        pl.col("Volumen_Litros").cast(pl.Float32),
        pl.col("Precio_Lista_DOP").cast(pl.Float32),
        pl.col("Costo_Prod_DOP").cast(pl.Float32),
        pl.col("Peso_Venta").cast(pl.Float32),
    ])
    
    # 9. Normalizar pesos de venta
    total_peso = df_final["Peso_Venta"].sum()
    df_final = df_final.with_columns(
        (pl.col("Peso_Venta") / total_peso).alias("Peso_Venta_Normalizado")
    )
    
        # 10. Ajustar y verificar schema DimProducto
    if "DimProducto" in SCHEMAS:
        schema = SCHEMAS["DimProducto"]
        df_final = asegurar_columnas(df_final, schema)
        df_final = df_final.cast(dict(schema))  # type: ignore[arg-type]
        """
        logger.info(f"Schema DimProducto esperado : {schema}")
        logger.info(f"Schema DimProducto obtenido: {df_final.schema}")
        assert df_final.schema == schema, "Schema de DimProducto no coincide con SCHEMAS['DimProducto']"
        """
    
    guardar_parquet(df_final, "dim_producto")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_producto.parquet")

# --------------------------------------------------------------------
# 9. DimCanalDistribucion (basado en CANALES_RD)
# --------------------------------------------------------------------
def generar_dim_canal_distribucion() -> pl.LazyFrame:
    logger.info("    🏪 Generando DimCanalDistribucion (basado en CANALES_RD)...")
    
    canales_data = []
    for i, (nombre_canal, info) in enumerate(CANALES_RD.items()):
        canales_data.append({
            "ID_Canal": f"CAN-{str(i+1).zfill(2)}",
            "Nombre_Canal": nombre_canal,
            "Peso_Mercado": info["peso"],
            "Segmentos_Objetivo": ",".join(info["segmentos"]),
            "Es_Ticket_Bajo": info["ticket_bajo"],
            "Estado": "Activo",
        })
    
    df = pl.DataFrame(canales_data)
    
    # Normalizar pesos
    total_peso = df["Peso_Mercado"].sum()
    df = df.with_columns(
        (pl.col("Peso_Mercado") / total_peso).alias("Peso_Mercado_Normalizado")
    )
    
    # Cast de tipos numéricos
    df = df.with_columns([
        pl.col("Peso_Mercado").cast(pl.Float32),
        pl.col("Peso_Mercado_Normalizado").cast(pl.Float32),
    ])
    
    # Verificación de schema
    if "DimCanalDistribucion" in SCHEMAS:
        schema = SCHEMAS["DimCanalDistribucion"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]

    
    guardar_parquet(df, "dim_canal_distribucion")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_canal_distribucion.parquet")

# --------------------------------------------------------------------
# 10. DimCluster (Segmentación Estratégica)
# --------------------------------------------------------------------
def generar_dim_cluster() -> pl.LazyFrame:
    logger.info("    🧩 Generando DimCluster (Segmentación Estratégica)...")
    data = [
        {"Cluster_ID": 1, "Nombre_Cluster": "VIP - Estratégico", "Descripcion": "Alto volumen, alta frecuencia, alta rentabilidad", "Nivel_Prioridad": 1},
        {"Cluster_ID": 2, "Nombre_Cluster": "Desarrollo", "Descripcion": "Volumen medio, potencial de crecimiento", "Nivel_Prioridad": 2},
        {"Cluster_ID": 3, "Nombre_Cluster": "Estándar", "Descripcion": "Compra recurrente promedio, mantenimiento", "Nivel_Prioridad": 3},
        {"Cluster_ID": 4, "Nombre_Cluster": "Ocasional / Riesgo", "Descripcion": "Baja frecuencia, bajo volumen o riesgo de churn", "Nivel_Prioridad": 4},
    ]
    df = pl.DataFrame(data).with_columns([
        pl.col("Cluster_ID").cast(pl.Int8),
        pl.col("Nivel_Prioridad").cast(pl.Int8),
    ])

    if "DimCluster" in SCHEMAS:
        schema = SCHEMAS["DimCluster"]
        df = asegurar_columnas(df, schema)
        df = df.cast(dict(schema))  # type: ignore[arg-type]
      
    guardar_parquet(df, "dim_cluster")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_cluster.parquet")


# --------------------------------------------------------------------
# 11. DimCliente Masiva. Total objetivo: {sum(NUM_CLIENTES_POR_ANO}
# --------------------------------------------------------------------
def generar_dim_cliente_masiva(
    lf_geografia: pl.LazyFrame,
    lf_canal: pl.LazyFrame,
    lf_cluster: pl.LazyFrame,   # se mantiene por consistencia de firma
) -> pl.LazyFrame:
    """
    Genera DimCliente simulando crecimiento anual.
    Depende de DimGeografia, DimCanalDistribucion y DimCluster.
    Usa las variables globales NUM_CLIENTES_POR_ANO, ANOS_SIMULACION, PESO_SEGMENTACION_CANAL.
    """
    logger.info(f" 👥 Generando DimCliente Masiva. Total objetivo: {sum(NUM_CLIENTES_POR_ANO.values()):,}")

    # 1. Collect de dimensiones pequeñas para sampling en memoria
    df_geo = lf_geografia.collect()
    df_canal = lf_canal.collect()
    df_cluster = lf_cluster.collect()

    # -------------------------
    # 2. Preparar arrays de sampling
    # -------------------------
    ids_geo = df_geo["ID_Provincia"].to_numpy()

    # Geografía: Peso_Normalizado si existe, si no Peso base
    if "Peso_Normalizado" in df_geo.columns:
        pesos_geo = df_geo.get_column("Peso_Normalizado").to_numpy()
    else:
        pesos_geo = df_geo["Peso"].to_numpy()

    suma_geo = float(pesos_geo.sum())
    if suma_geo > 0:
        pesos_geo = pesos_geo / suma_geo
    else:
        pesos_geo = np.ones_like(pesos_geo, dtype=float) / len(pesos_geo)

    ids_canal = df_canal["ID_Canal"].to_numpy()
    nombres_canal = df_canal["Nombre_Canal"].to_numpy()

    pesos_canal_global = df_canal["Peso_Mercado_Normalizado"].to_numpy()
    suma_canal = float(pesos_canal_global.sum())
    if suma_canal > 0:
        pesos_canal_global = pesos_canal_global / suma_canal
    else:
        pesos_canal_global = np.ones_like(pesos_canal_global, dtype=float) / len(pesos_canal_global)

    rng = np.random.default_rng(SEED_VAL)
    current_customer_id_counter = 0
    faker_es = Faker("es_ES")

    # Acumulador: lista de dicts con todos los clientes
    clientes_activos_historicos: list[dict] = []

    # -------------------------
    # 3. Loop por años
    # -------------------------
    for i, ano in enumerate(ANOS_SIMULACION):
        logger.info(f" 📅 Generando clientes para el año {ano}...")

        target_clientes_este_ano = NUM_CLIENTES_POR_ANO[ano]

        # 3.1 Churn sobre clientes ya existentes (a partir del 2do año)
        if i > 0 and clientes_activos_historicos:
            df_temp_clientes = pl.DataFrame(clientes_activos_historicos, schema=SCHEMAS["DimCliente"])

            candidatos_mask = (df_temp_clientes["Ano_Creacion"] < ano) & (df_temp_clientes["Activo"])
            num_candidatos: int = int(candidatos_mask.sum())

            if num_candidatos > 0:
                churn_decisions = rng.random(size=num_candidatos) < CHURN_RATE_ANUAL

                activo_np = df_temp_clientes["Activo"].to_numpy().copy()
                idx_candidatos = np.where(candidatos_mask.to_numpy())[0]
                activo_np[idx_candidatos[churn_decisions]] = False

                df_temp_clientes = df_temp_clientes.with_columns(
                    pl.Series("Activo", activo_np)
                )

                num_inactivos_antes = int(sum(not c["Activo"] for c in clientes_activos_historicos))
                num_inactivos_despues = int((~df_temp_clientes["Activo"]).sum())
                num_churned = max(0, num_inactivos_despues - num_inactivos_antes)

                logger.info(
                    f" 📉 Aplicado Churn del {CHURN_RATE_ANUAL:.0%} en {ano}. "
                    f"Clientes inactivos nuevos este año por churn: {num_churned}."
                )

                clientes_activos_historicos = df_temp_clientes.to_dicts()
            else:
                logger.info(f" 📉 Sin candidatos a churn en {ano}.")

        # 3.2 Número de clientes activos actuales
        current_active_customers = sum(1 for c in clientes_activos_historicos if c["Activo"])

        # 3.3 Clientes nuevos necesarios para llegar al target
        num_nuevos: int = max(0, target_clientes_este_ano - current_active_customers)

        if num_nuevos > 0:
            new_client_data: list[dict] = []

            for _ in range(num_nuevos):
                current_customer_id_counter += 1

                # Geografía y canal
                asign_geo_id = rng.choice(ids_geo, p=pesos_geo)
                asign_canal_idx = rng.choice(len(ids_canal), p=pesos_canal_global)
                asign_canal_id = ids_canal[asign_canal_idx]
                asign_canal_nombre = nombres_canal[asign_canal_idx]

                # Segmento y cluster
                segmento_elegido = "E"
                cluster_elegido = 4

                pesos_seg_canal = PESO_SEGMENTACION_CANAL.get(asign_canal_nombre)
                if pesos_seg_canal:
                    segmentos_validos = list(pesos_seg_canal.keys())
                    probs_seg = np.array(list(pesos_seg_canal.values()), dtype=float)
                    probs_seg = probs_seg / probs_seg.sum()
                    segmento_elegido = str(rng.choice(segmentos_validos, p=probs_seg))

                    if segmento_elegido in ["A", "B"]:
                        cluster_elegido = int(rng.choice([1, 2], p=[0.6, 0.4]))
                    else:
                        cluster_elegido = int(rng.choice([3, 4], p=[0.7, 0.3]))

                # Fecha de alta y ubicación
                fecha_alta = date(ano, random.randint(1, 12), random.randint(1, 28))

                new_client_data.append(
                    {
                        "ID_Cliente": f"CLI-{current_customer_id_counter:06d}",
                        "Nombre_Cliente": faker_es.company()
                        if random.random() < 0.7
                        else faker_es.name(),
                        "ID_Provincia": asign_geo_id,
                        "ID_Canal": asign_canal_id,
                        "Segmento_Cliente": segmento_elegido,
                        "Cluster_ID": cluster_elegido,
                        "Fecha_Alta": fecha_alta,
                        "Activo": True,
                        "Latitud": float(rng.uniform(18.0, 19.8)),
                        "Longitud": float(rng.uniform(-71.5, -68.5)),
                        "Ano_Creacion": int(ano),
                    }
                )

            clientes_activos_historicos.extend(new_client_data)

    # -------------------------
    # 4. Consolidación final
    # -------------------------
    if clientes_activos_historicos:
        df_final = pl.DataFrame(clientes_activos_historicos, schema=SCHEMAS["DimCliente"])
    else:
        logger.warning("⚠️ No se generaron clientes. Retornando DataFrame vacío.")
        df_final = pl.DataFrame(schema=SCHEMAS.get("DimCliente"))

    # 5. Asegurar columnas y tipos según schema
    if "DimCliente" in SCHEMAS:
        schema: pl.Schema = SCHEMAS["DimCliente"]
        df_final = asegurar_columnas(df_final, schema)
        df_final = df_final.cast(dict(schema))  # type: ignore[arg-type]
        """logger.info(f"Schema DimCliente esperado : {schema}")
        logger.info(f"Schema DimCliente obtenido: {df_final.schema}")
        assert df_final.schema == schema, "Schema de DimCliente no coincide con SCHEMAS['DimCliente']"""

    guardar_parquet(df_final, "dim_cliente")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_cliente.parquet")

# --------------------------------------------------------------------
# 12. DimEmpleado (Fuerza Laboral Completa)
# --------------------------------------------------------------------
def generar_dim_empleado(
    lf_departamento: pl.LazyFrame,
    lf_puesto: pl.LazyFrame,
    lf_cedi: pl.LazyFrame,
    lf_geografia: pl.LazyFrame,
) -> pl.LazyFrame:
    logger.info("    👨‍💼 Generando DimEmpleado (Fuerza Laboral Completa)...")
    
    df_departamento = lf_departamento.collect()
    df_puesto       = lf_puesto.collect()
    df_cedi         = lf_cedi.collect()
    df_geografia    = lf_geografia.collect()

    empleados_data = []
    empleado_id_counter = 1
    
    rng  = np.random.default_rng(SEED_VAL)
    fake = Faker("es_ES")

    mapa_puesto_info = (
        df_puesto
        .group_by("Puesto_ID")
        .agg(
            pl.col("Nombre_Puesto").first(),
            pl.col("Departamento_ID").first(),
            pl.col("Salario_Base_Mensual_Min_DOP").first(),
            pl.col("Salario_Base_Mensual_Max_DOP").first(),
        )
        .to_dicts()
    )
    
    provincia_ids   = df_geografia["ID_Provincia"].to_list()
    pesos_provincia = df_geografia["Peso_Normalizado"].to_list()

    CANTIDAD_EMPLEADOS_POR_PUESTO = {
        "Chofer de reparto": 330,
        "Ayudante de reparto": 380,
        "Operador de Montacargas": 40,
        "Coordinador Logistico": 50,
        "Jefe de Almacen_Bodega": 15,
        "Operador de produccion (Linea)": 420,
        "Tecnico de Mantenimiento": 70,
        "Quimico_Control de Calidad": 20,
        "Jefe de Produccion": 7,
        "Vendedor_Preventista": 460,
        "Ejecutivo Comercial": 145,
        "Ejecutivo de Cuentas Clave (KAM)": 9,
        "Gerente de Territorio": 38,
        "Backoffice_Ventas / Soporte_Comercial": 22,
        "Asistente Administrativo": 75,
        "Analista Administrativo": 95,
        "Recepcionista": 25,
        "Contador_Especialista Financiero": 12,
        "Abogado Corporativo": 5,
        "Especialista de Marketing_Marca": 14,
        "Analista de Datos_BI": 14,
        "Administrador de Sistemas_Redes": 12,
        "Ingeniero de Automatizacion_IoT": 8,
        "Gerente de Area_Departamento": 9,
        "Director_Vicepresidente": 4,
        "Personal Servicios Generales (Conserjes, Mensajeros, Cafeteria)": 290,
        "Oficial de Seguridad_CCTV": 95,
    }

    for puesto_info_dict in mapa_puesto_info:
        id_puesto       = puesto_info_dict["Puesto_ID"]
        nombre_puesto   = puesto_info_dict["Nombre_Puesto"]
        id_departamento = puesto_info_dict["Departamento_ID"]
        sueldo_min      = float(puesto_info_dict["Salario_Base_Mensual_Min_DOP"])
        sueldo_max      = float(puesto_info_dict["Salario_Base_Mensual_Max_DOP"])
        
        num_empleados_para_puesto = CANTIDAD_EMPLEADOS_POR_PUESTO.get(
            nombre_puesto,
            random.randint(1, 3),
        )

        if "Gerente" in nombre_puesto or "Director" in nombre_puesto:
            num_empleados_para_puesto = min(num_empleados_para_puesto, 2)

        for _ in range(num_empleados_para_puesto):
            salario = round(rng.uniform(sueldo_min, sueldo_max), 2)
            fecha_contratacion = fake.date_between(
                start_date=date(2010, 1, 1),
                end_date="today",
            )

            empleados_data.append({
                "Empleado_ID": f"EMP-{str(empleado_id_counter).zfill(5)}",
                "Nombre_Completo": fake.name(),
                "Departamento_ID": id_departamento,
                "Puesto_ID": id_puesto,
                "CEDI_ID": None,  # se asigna luego
                "Provincia_ID_Residencia": rng.choice(
                    provincia_ids, p=pesos_provincia
                ),
                "Fecha_Contratacion": fecha_contratacion,
                "Salario_Base_Mensual_DOP": salario,
                "Estatus_Empleado": "Activo",
                "Email_Corporativo": fake.email(),
                "Telefono_Contacto": fake.phone_number(),
                "Fecha_Nacimiento": fake.date_of_birth(
                    minimum_age=20,
                    maximum_age=60,
                ),
                "Genero": random.choice(
                    ["Masculino", "Femenino", "Otro"]
                ),
                "Experiencia_Anios": (
                    date.today() - fecha_contratacion
                ).days // 365,
                "Tipo_Contrato": random.choice(
                    ["Indefinido", "Temporal"]
                ),
            })
            empleado_id_counter += 1

    df_empleado = pl.DataFrame(empleados_data)

    # 2) Provincia -> Región -> CEDI, con fallback CEDI-PRIN-01
    region_por_prov = dict(zip(df_geografia["ID_Provincia"], df_geografia["Region"]))
    cedis_por_region: dict[str, list[str]] = {}
    for row in df_cedi.iter_rows(named=True):
        reg = row["Region_Operacion"]
        cedis_por_region.setdefault(reg, []).append(row["CEDI_ID"])

    # ID del CEDI principal
    cedi_principal = "CEDI-PRIN-01"

    def provincia_a_cedi(prov_id: str | None) -> str:
        if prov_id is None:
            return cedi_principal
        reg = region_por_prov.get(prov_id)
        if reg is None:
            return cedi_principal
        cedis_reg = cedis_por_region.get(reg)
        if not cedis_reg:
            return cedi_principal
        return random.choice(cedis_reg)

    # 3) Asignar CEDI_ID a todos según provincia (con fallback)
    df_empleado = df_empleado.with_columns(
        pl.col("Provincia_ID_Residencia").map_elements(
            provincia_a_cedi,
            return_dtype=pl.Utf8,
        ).alias("CEDI_ID")
    )

    # 4) Casts finales
    df_empleado = df_empleado.with_columns(
        [
            pl.col("CEDI_ID").cast(pl.Utf8),
            pl.col("Salario_Base_Mensual_DOP").cast(pl.Float32),
            pl.col("Experiencia_Anios").cast(pl.Int8),
        ]
    )

    # 5) Verificación schema DimEmpleado
    if "DimEmpleado" in SCHEMAS:
        schema = SCHEMAS["DimEmpleado"]
        df_empleado = asegurar_columnas(df_empleado, schema)
        df_empleado = df_empleado.cast(dict(schema))  # type: ignore[arg-type]

    guardar_parquet(df_empleado, "dim_empleado")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_empleado.parquet")


# --------------------------------------------------------------------
# 13. DimPromocion (Catálogo de Ofertas)
# --------------------------------------------------------------------
def generar_dim_promocion() -> pl.LazyFrame:
    logger.info("    🎟️ Generando DimPromocion (Catálogo de Ofertas)...")
    
    promos_data: list[dict] = []

    # Fila base: Sin promoción
    promos_data.append({
        "ID_Promocion": "PROM-000",
        "Nombre_Promocion": "Sin Promoción",
        "Descripcion": "Venta regular sin descuento ni incentivo promocional",
        "Factor_Incremento_Venta": 1.0,      # venta base
        "Peso_Probabilidad_Uso": 0.0,        # se controla por lógica de negocio
        "Activa": True
    })

    # Resto de promociones desde PROMOCIONES_MAESTRAL
    for i, promo in enumerate(PROMOCIONES_MAESTRAL, start=1):
        # Asegurar que los campos numéricos vienen en escala correcta
        peso_incremento_venta = float(promo["Peso_Incremento_Venta"])
        peso_incremento_pct  = float(promo["%_Peso_Incremento"])

        factor_incremento = 1.0 + peso_incremento_venta
        peso_prob_uso     = peso_incremento_pct / 100.0

        promos_data.append({
            "ID_Promocion": f"PROM-{str(i).zfill(3)}",
            "Nombre_Promocion": promo["Promocion"],
            "Descripcion": (
                f"Promoción '{promo['Promocion']}' con impacto estimado de "
                f"{peso_incremento_pct:.1f}% sobre la venta base"
            ),
            "Factor_Incremento_Venta": factor_incremento,
            "Peso_Probabilidad_Uso": peso_prob_uso,
            "Activa": True
        })
        
    df = pl.DataFrame(promos_data).with_columns(
        [
            pl.col("Factor_Incremento_Venta").cast(pl.Float32),
            pl.col("Peso_Probabilidad_Uso").cast(pl.Float32),
            pl.col("Activa").cast(pl.Boolean),
        ]
    )

    # Ajustar al esquema maestro si existe
    if "DimPromocion" in SCHEMAS:
        df = asegurar_columnas(df, SCHEMAS["DimPromocion"])
        df = df.cast(SCHEMAS["DimPromocion"])

    guardar_parquet(df, "dim_promocion")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_promocion.parquet")


# --------------------------------------------------------------------
# 14. DimVendedor (Extracto especializado y enriquecido de DimEmpleado)
# --------------------------------------------------------------------
def generar_dim_vendedor(
    lf_empleado: pl.LazyFrame,
    lf_puesto: pl.LazyFrame,
    lf_cedi: pl.LazyFrame,
) -> pl.LazyFrame:
    logger.info("    💼 Generando DimVendedor (Extracto especializado y enriquecido de DimEmpleado)...")
    
    # 1) Collect de dimensiones
    df_empleado = lf_empleado.collect()
    df_puesto   = lf_puesto.collect()
    df_cedi     = lf_cedi.collect()

    # 2) Puestos de ventas (amplio, incluye los de la lista Ventas)
    puestos_ventas = df_puesto.filter(
        pl.col("Nombre_Puesto").str.contains(
            "(?i)Vendedor|Preventista|Ejecutivo Comercial|Cuentas Clave|KAM|Gerente de Territorio|Backoffice_Ventas|Soporte_Comercial|Supervisor de Ventas|Gerente de Ventas|Mercadeo|Merchandiser|Promotor"
        )
    )
    ids_puestos_ventas = puestos_ventas["Puesto_ID"].to_list()

    # 3) Empleados activos con esos puestos
    df_vendedores_base = df_empleado.filter(
        (pl.col("Puesto_ID").is_in(ids_puestos_ventas)) &
        (pl.col("Estatus_Empleado") == "Activo")
    )

    if df_vendedores_base.height == 0:
        logger.warning("⚠️ No se encontraron empleados activos con roles de ventas o relacionados. Revise DimEmpleado/DimPuesto.")
        if "DimVendedor" in SCHEMAS:
            df_empty = pl.DataFrame(schema=SCHEMAS["DimVendedor"])
            guardar_parquet(df_empty, "dim_vendedor")
            return pl.scan_parquet(DIRS["OUTPUT"] / "dim_vendedor.parquet")
        return pl.DataFrame().lazy()

    # 4) Perfiles de venta (incluye los 5 puestos de Ventas)
    PERFILES_VENTA = {
        # Ventas (lista proporcionada)
        "Vendedor_Preventista": (
            "Canal Tradicional (Colmados)", 800_000, 0.025, False, 150
        ),
        "Ejecutivo Comercial": (
            "Canal Tradicional + Cuentas Medianas", 1_000_000, 0.02, False, 120
        ),
        "Ejecutivo de Cuentas Clave (KAM)": (
            "Supermercados/Cadenas", 5_000_000, 0.015, False, 15
        ),
        "Gerente de Territorio": (
            "Gestión Zonal / Territorios", 0, 0.005, True, 0
        ),
        "Backoffice_Ventas / Soporte_Comercial": (
            "Soporte Comercial", 0, 0.0, False, 0
        ),

        # Otros que ya venías usando
        "Preventista": (
            "Canal Tradicional (Colmados)", 800_000, 0.025, False, 150
        ),
        "Vendedor Autoventa": (
            "Ruta Mixta", 1_200_000, 0.03, False, 100
        ),
        "Vendedor Mayorista": (
            "Mayoristas", 8_000_000, 0.01, False, 8
        ),
        "Supervisor de Ventas": (
            "Supervisión Zonal", 0, 0.005, True, 0
        ),
        "Gerente de Ventas Regional": (
            "Gestión Regional", 0, 0.003, True, 0
        ),
        "Promotor/Merchandiser": (
            "Soporte Punto Venta", 500_000, 0.01, False, 80
        ),
    }

    # 5) Tabla auxiliar Puesto_ID -> Nombre_Puesto
    df_puesto_nombres = df_puesto.select(["Puesto_ID", "Nombre_Puesto"])

    def es_gerente_o_supervisor(pid: str) -> bool:
        fila = df_puesto_nombres.filter(pl.col("Puesto_ID") == pid)
        if fila.is_empty():
            return False
        nombre = fila["Nombre_Puesto"][0]
        return ("Gerente" in nombre) or ("Supervisor" in nombre)

    # 6) Gerentes/supervisores por CEDI (para Gerente_Directo_ID)
    gerentes_disponibles = df_vendedores_base.filter(
        pl.col("Puesto_ID").map_elements(es_gerente_o_supervisor, return_dtype=pl.Boolean)
    ).select(["Empleado_ID", "CEDI_ID"])

    mapa_gerentes_por_cedi: dict[str, list[str]] = {}
    for row in gerentes_disponibles.iter_rows(named=True):
        cedi = row["CEDI_ID"]
        if cedi is None:
            continue
        mapa_gerentes_por_cedi.setdefault(cedi, []).append(row["Empleado_ID"])

    # 7) Construir DimVendedor
    vendedores_data: list[dict] = []

    for row in df_vendedores_base.iter_rows(named=True):
        id_empleado        = row["Empleado_ID"]
        puesto_id          = row["Puesto_ID"]
        nombre_completo    = row["Nombre_Completo"]
        cedi_base_id       = row["CEDI_ID"]
        fecha_contratacion = row["Fecha_Contratacion"]

        nombre_puesto_empleado = df_puesto.filter(
            pl.col("Puesto_ID") == puesto_id
        )["Nombre_Puesto"][0]

        # 7.1 Determinar perfil (prioriza match exacto para los puestos de Ventas)
        perfil_encontrado = None

        # Match exacto
        for perfil_key in PERFILES_VENTA.keys():
            if perfil_key == nombre_puesto_empleado:
                perfil_encontrado = perfil_key
                break

        # Match por contiene si no hubo exacto
        if perfil_encontrado is None:
            for perfil_key in PERFILES_VENTA.keys():
                if perfil_key.lower() in nombre_puesto_empleado.lower():
                    perfil_encontrado = perfil_key
                    break

        # Fallback genérico
        if perfil_encontrado is None:
            if "Vendedor" in nombre_puesto_empleado:
                perfil_encontrado = "Vendedor_Preventista"
            else:
                perfil_encontrado = "Promotor/Merchandiser"

        enfoque, meta_base, comision, es_gerencial, prom_clientes = PERFILES_VENTA[perfil_encontrado]

        # Meta ajustada +/- 10 %
        meta_real = meta_base * random.uniform(0.9, 1.1)

        # Gerente directo por CEDI (solo si no es rol gerencial)
        gerente_directo_id = None
        if (not es_gerencial) and (cedi_base_id in mapa_gerentes_por_cedi):
            gerente_directo_id = random.choice(mapa_gerentes_por_cedi[cedi_base_id])

        vendedores_data.append({
            "Vendedor_ID": id_empleado,
            "Empleado_ID": id_empleado,
            "Puesto_ID": puesto_id,
            "Nombre_Vendedor": nombre_completo,
            "CEDI_Base_ID": cedi_base_id,
            "Tipo_Vendedor": perfil_encontrado,
            "Enfoque_Canal": enfoque,
            "Meta_Venta_Mensual_DOP": round(meta_real, 2),
            "Porcentaje_Comision_Objetivo": comision,
            "Telefono_Flota": f"809-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
            "Nivel_Experiencia": random.choice(["Junior", "Intermedio", "Senior", "Master"]),
            "Fecha_Asignacion_Ruta": fecha_contratacion,
            "Estado_Vendedor": "Activo",
            "Gerente_Directo_ID": gerente_directo_id,
            "Promedio_Clientes_Visitados_Dia": prom_clientes if not es_gerencial else 0,
            "Es_Supervisor_Gerente": es_gerencial,
        })

    df_dim_vendedor = pl.DataFrame(vendedores_data).with_columns([
        pl.col("Meta_Venta_Mensual_DOP").cast(pl.Float32),
        pl.col("Porcentaje_Comision_Objetivo").cast(pl.Float32),
        pl.col("Promedio_Clientes_Visitados_Dia").cast(pl.Int16),
        pl.col("Gerente_Directo_ID").cast(pl.Utf8),
    ])

    # 8) Ajustar al schema maestro
    if "DimVendedor" in SCHEMAS:
        schema = SCHEMAS["DimVendedor"]
        df_dim_vendedor = asegurar_columnas(df_dim_vendedor, schema)
        df_dim_vendedor = df_dim_vendedor.cast(dict(schema))  # type: ignore[arg-type]
       # logger.info(f"Schema DimVendedor esperado : {schema}")
        #logger.info(f"Schema DimVendedor obtenido: {df_dim_vendedor.schema}")
       # assert df_dim_vendedor.schema == schema, "Schema de DimVendedor no coincide con SCHEMAS['DimVendedor']"

    
    """logger.info(
        "🔎 Sample de DimVendedor:\n"
        + df_dim_vendedor.head(10).to_pandas().to_string(index=False)
    )"""

    guardar_parquet(df_dim_vendedor, "dim_vendedor")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_vendedor.parquet")


# --------------------------------------------------------------------
# 15. DimVehiculo (Flota de Transporte)
# --------------------------------------------------------------------
def generar_dim_vehiculo(lf_cedis: pl.LazyFrame) -> pl.LazyFrame:
    logger.info("    🚚 Generando DimVehiculo (Flota de Transporte)...")
    
    df_cedis = lf_cedis.collect()
    # En DimCEDIS la PK es CEDI_ID
    cedi_ids = df_cedis["CEDI_ID"].to_list()

    # Definición detallada de la flota
    TIPOS_VEHICULOS = [
        {
            "Modelo": "Hyundai HD-78",
            "Tipo": "Camión Interurbano",
            "Cap_Ton": 5.5,
            "Vol_M3": 20,
            "Km_L": 4.5,
            "Costo_Dia": 600,
            "Uso": "Rutas Largas",
            "Depreciacion_Anual_Pct": 0.15,
        },
        {
            "Modelo": "Fuso Canter FE85",
            "Tipo": "Camión Interurbano",
            "Cap_Ton": 5.5,
            "Vol_M3": 22,
            "Km_L": 4.5,
            "Costo_Dia": 600,
            "Uso": "Rutas Provinciales",
            "Depreciacion_Anual_Pct": 0.15,
        },
        {
            "Modelo": "Isuzu NPR",
            "Tipo": "Furgoneta Urbana",
            "Cap_Ton": 2.5,
            "Vol_M3": 12,
            "Km_L": 5.5,
            "Costo_Dia": 400,
            "Uso": "Urbano Denso",
            "Depreciacion_Anual_Pct": 0.18,
        },
        {
            "Modelo": "Panel H-100",
            "Tipo": "Vehículo Ligero",
            "Cap_Ton": 1.5,
            "Vol_M3": 8,
            "Km_L": 6.5,
            "Costo_Dia": 300,
            "Uso": "Colmados/Rural",
            "Depreciacion_Anual_Pct": 0.20,
        },
        {
            "Modelo": "Freightliner M2",
            "Tipo": "Camión Pesado",
            "Cap_Ton": 10.0,
            "Vol_M3": 45,
            "Km_L": 3.5,
            "Costo_Dia": 850,
            "Uso": "Transferencia CEDI-CEDI/Mayoristas",
            "Depreciacion_Anual_Pct": 0.12,
        },
        {
            "Modelo": "Moto-Carro",
            "Tipo": "Preventa/Reparto",
            "Cap_Ton": 0.5,
            "Vol_M3": 1.5,
            "Km_L": 25.0,
            "Costo_Dia": 150,
            "Uso": "Preventa/Emergencia",
            "Depreciacion_Anual_Pct": 0.25,
        },
    ]
    
    vehiculos_data: list[dict] = []
    
    consecutivo = 1
    rng = np.random.default_rng(SEED_VAL)

    for row_cedi in df_cedis.iter_rows(named=True):
        cedi_id = row_cedi["CEDI_ID"]
        
        # Principal vs regional
        tipo_cedi = str(row_cedi["Tipo_CEDI"] or "").upper()
        es_principal = ("PRINCIPAL" in tipo_cedi) or ("PRIN" in cedi_id)
        
        # Volumen de flota por CEDI
        n_vehiculos_base = random.randint(30, 45) if es_principal else random.randint(15, 20)
        
        for _ in range(n_vehiculos_base):
            # Mix de tipos por CEDI
            if es_principal:
                weights = [0.25, 0.20, 0.20, 0.10, 0.20, 0.05]
            else:
                weights = [0.15, 0.30, 0.30, 0.15, 0.05, 0.05]
            
            tipo = rng.choice(TIPOS_VEHICULOS, p=weights)

            # Placa: una letra + 6 dígitos
            placa_letras = rng.choice(list("ABCDL"))
            placa_nums = rng.integers(100000, 999999)

            anho_fab = int(rng.integers(2015, 2023))
            kilometraje_base = (date.today().year - anho_fab) * 20000
            kilometraje_actual = int(
                rng.integers(
                    max(1000, kilometraje_base - 10000),
                    kilometraje_base + 30000,
                )
            )
            
            vehiculos_data.append(
                {
                    "ID_Vehiculo": f"VEH-{str(consecutivo).zfill(4)}",
                    "CEDI_Asignado_ID": cedi_id,
                    "Placa": f"{placa_letras}{placa_nums}",
                    "Marca_Modelo": tipo["Modelo"],
                    "Tipo_Vehiculo": tipo["Tipo"],
                    "Capacidad_Carga_Ton": tipo["Cap_Ton"],
                    "Capacidad_Volumen_M3": tipo["Vol_M3"],
                    "Rendimiento_Promedio_KmL": tipo["Km_L"],
                    "Costo_Fijo_Operativo_Diario_DOP": tipo["Costo_Dia"],
                    "Uso_Principal": tipo["Uso"],
                    "Anio_Fabricacion": anho_fab,
                    "Kilometraje_Actual_KM": kilometraje_actual,
                    "Estado_Vehiculo": rng.choice(
                        ["Operativo", "En Taller", "Baja"], p=[0.85, 0.12, 0.03]
                    ),
                    "Tiene_GPS": rng.choice([True, True, False]),
                    "Valor_Adquisicion_DOP": round(
                        float(rng.uniform(tipo["Costo_Dia"] * 100, tipo["Costo_Dia"] * 300)), 2
                    ),
                    "Depreciacion_Anual_Pct": tipo["Depreciacion_Anual_Pct"],
                }
            )
            consecutivo += 1

    df_vehiculo = pl.DataFrame(vehiculos_data)

    df_vehiculo = df_vehiculo.with_columns(
        [
            pl.col("Capacidad_Carga_Ton").cast(pl.Float32),
            pl.col("Capacidad_Volumen_M3").cast(pl.Float32),
            pl.col("Rendimiento_Promedio_KmL").cast(pl.Float32),
            pl.col("Costo_Fijo_Operativo_Diario_DOP").cast(pl.Float32),
            pl.col("Valor_Adquisicion_DOP").cast(pl.Float32),
            pl.col("Depreciacion_Anual_Pct").cast(pl.Float32),
            pl.col("Kilometraje_Actual_KM").cast(pl.Int64),
        ]
    )

    if "DimVehiculo" in SCHEMAS:
        df_vehiculo = asegurar_columnas(df_vehiculo, SCHEMAS["DimVehiculo"])
        df_vehiculo = df_vehiculo.cast(SCHEMAS["DimVehiculo"])

    guardar_parquet(df_vehiculo, "dim_vehiculo")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_vehiculo.parquet")

# --------------------------------------------------------------------
# 16. DimRuta (Conexión Logística CEDI-Geografía-Recursos)
# --------------------------------------------------------------------
import random
from geopy.distance import geodesic
from faker import Faker

fake = Faker("es_ES")

def generar_dim_ruta(
    lf_cedi: pl.LazyFrame,
    lf_geografia: pl.LazyFrame,
    lf_vehiculo: pl.LazyFrame,
    lf_vendedor: pl.LazyFrame,
) -> pl.LazyFrame:
    logger.info("    🗺️ Generando DimRuta (Conexión Logística CEDI-Geografía-Recursos)...")
    
    df_cedi     = lf_cedi.collect()
    df_geo      = lf_geografia.collect()
    df_vehiculo = lf_vehiculo.collect()
    df_vendedor = lf_vendedor.collect()
    
    rutas_data: list[dict] = []
    
    VELOCIDAD_PROMEDIO_KMH = {
        "Urbana Densa": 20.0,
        "Urbana Estándar": 30.0,
        "Interurbana Regular": 50.0,
        "Autopista": 70.0,
    }
    
    rng = np.random.default_rng(SEED_VAL)
    random.seed(SEED_VAL)

    consecutivo_ruta = 1

    for row_cedi in df_cedi.iter_rows(named=True):
        cedi_id   = row_cedi["CEDI_ID"]
        cedi_nom  = row_cedi["Nombre_CEDI"]
        cedi_lat  = row_cedi["Latitud"]
        cedi_lon  = row_cedi["Longitud"]
        region_op = row_cedi["Region_Operacion"]
        capacidad = row_cedi["Capacidad_Pallets"]

        # Geografías objetivo por región
        geos_objetivo = (
            df_geo
            .filter(pl.col("Region") == region_op)
            .to_dicts()
        )

        if not geos_objetivo:
            logger.warning(
                f"⚠️ No hay geografías para la región '{region_op}' del CEDI {cedi_id}. "
                "Usando geografías aleatorias."
            )
            geos_objetivo = df_geo.sample(
                n=min(3, df_geo.height),
                seed=int(rng.integers(0, 10_000)),
            ).to_dicts()

        # Vehículos operativos del CEDI
        vehiculos_disponibles = (
            df_vehiculo
            .filter(
                (pl.col("CEDI_Asignado_ID") == cedi_id)
                & (pl.col("Estado_Vehiculo") == "Operativo")
            )
            .to_dicts()
        )

        # Vendedores activos base en el CEDI
        vendedores_disponibles = (
            df_vendedor
            .filter(
                (pl.col("CEDI_Base_ID") == cedi_id)
                & (pl.col("Estado_Vendedor") == "Activo")
            )
            .to_dicts()
        )

        if not vehiculos_disponibles or not vendedores_disponibles:
            logger.warning(
                f"⚠️ CEDI {cedi_id} ({cedi_nom}) sin suficientes vehículos o vendedores operativos. "
                "Se omiten rutas para este CEDI."
            )
            continue

        cant_veh = len(vehiculos_disponibles)

        # Regla "entera": máximo 2 rutas por vehículo
        max_rutas_por_vehiculo = 2

        # Límite adicional por capacidad (opcional, puedes afinar el divisor)
        if capacidad is None:
            limite_capacidad = 10_000
        else:
            limite_capacidad = max(int(capacidad / 50), 10)  # 1 ruta cada 50 pallets, mínimo 10

        num_rutas_por_cedi = min(
            int(cant_veh * max_rutas_por_vehiculo),
            limite_capacidad,
        )
        num_rutas_por_cedi = max(num_rutas_por_cedi, 10)

        for _ in range(num_rutas_por_cedi):
            vehiculo = random.choice(vehiculos_disponibles)
            vendedor = random.choice(vendedores_disponibles)
            geo_dest = random.choice(geos_objetivo)

            lat_dest = geo_dest["Latitud"]
            lon_dest = geo_dest["Longitud"]

            # Distancia geodésica (km)
            distancia_lineal = geodesic(
                (cedi_lat, cedi_lon),
                (lat_dest, lon_dest),
            ).km

            distancia_ruta = float(
                distancia_lineal * rng.uniform(1.3, 1.6)
            )
            distancia_ruta = max(5.0, distancia_ruta)

            # Tipo de ruta geográfica
            if distancia_ruta < 25 and region_op in ["Ozama", "Cibao Central"]:
                tipo_ruta_geo = "Urbana Densa"
            elif distancia_ruta < 50:
                tipo_ruta_geo = "Urbana Estándar"
            elif distancia_ruta < 150:
                tipo_ruta_geo = "Interurbana Regular"
            else:
                tipo_ruta_geo = "Autopista"

            vel = VELOCIDAD_PROMEDIO_KMH.get(tipo_ruta_geo, 40.0)
            tiempo_ida_hrs = round(distancia_ruta / vel, 2)

            # Frecuencia y días de operación
            frecuencia = random.choices(
                ["Diaria (L-S)", "Interdiaria (L-M-X)", "Semanal (1 día)"],
                weights=[0.3, 0.5, 0.2],
                k=1,
            )[0]

            if frecuencia == "Diaria (L-S)":
                dias_visita = "L-M-M-J-V-S"
            elif frecuencia == "Interdiaria (L-M-X)":
                dias_visita = random.choice(["L-M-V", "M-J-S", "L-X-V"])
            else:
                dias_visita = random.choice(
                    ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
                )

            nombre_prov_dest = geo_dest["Nombre_Provincia"]
            nombre_ruta = (
                f"Ruta {region_op[:3].upper()}-"
                f"{str(consecutivo_ruta).zfill(4)}-"
                f"{nombre_prov_dest}"
            )

            rutas_data.append(
                {
                    "ID_Ruta": f"RUT-{str(consecutivo_ruta).zfill(5)}",
                    "Nombre_Ruta": nombre_ruta,
                    "ID_CEDI_Origen": cedi_id,
                    "Nombre_CEDI_Origen": cedi_nom,
                    "ID_Provincia_Destino": geo_dest["ID_Provincia"],
                    "Nombre_Provincia_Destino": nombre_prov_dest,
                    "Zona_Especifica": f"{fake.city_suffix()} {fake.street_name()}",
                    "ID_Vehiculo_Asignado": vehiculo["ID_Vehiculo"],
                    "Marca_Modelo_Vehiculo": vehiculo["Marca_Modelo"],
                    "ID_Vendedor_Titular": vendedor["Vendedor_ID"],
                    "Nombre_Vendedor_Titular": vendedor["Nombre_Vendedor"],
                    "Tipo_Vendedor_Ruta": vendedor["Tipo_Vendedor"],
                    "Distancia_Ruta_KM": round(distancia_ruta, 2),
                    "Tiempo_Ruta_Estimado_Hrs": tiempo_ida_hrs,
                    "Costo_Peaje_Estimado_DOP": float(
                        random.choices(
                            [0, 0, 50, 100, 200],
                            weights=[0.4, 0.2, 0.2, 0.1, 0.1],
                            k=1,
                        )[0]
                    ),
                    "Frecuencia_Visita": frecuencia,
                    "Dias_Operacion_Semana": dias_visita,
                    "Tipo_Ruta_Geografica": tipo_ruta_geo,
                    "Estado_Ruta": "Activa",
                }
            )
            consecutivo_ruta += 1

    df_ruta = pl.DataFrame(rutas_data)

    df_ruta = df_ruta.with_columns(
        [
            pl.col("Distancia_Ruta_KM").cast(pl.Float32),
            pl.col("Tiempo_Ruta_Estimado_Hrs").cast(pl.Float32),
            pl.col("Costo_Peaje_Estimado_DOP").cast(pl.Float32),
        ]
    )

    if "DimRuta" in SCHEMAS:
        df_ruta = asegurar_columnas(df_ruta, SCHEMAS["DimRuta"])
        df_ruta = df_ruta.cast(SCHEMAS["DimRuta"])

    guardar_parquet(df_ruta, "dim_ruta")
    return pl.scan_parquet(DIRS["OUTPUT"] / "dim_ruta.parquet")





import polars as pl
import numpy as np
import math
import random
from datetime import date, timedelta
import gc # Garbage collection
from tqdm import tqdm
from geopy.distance import geodesic # Para calcular distancias de rutas

# Asegurarse de que las configuraciones globales necesarias estén disponibles
# (Estas deberían estar definidas en celdas anteriores, las repito por completitud si esta celda se ejecuta aislada)
if 'logger' not in globals():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

if 'DIRS' not in globals():
    from pathlib import Path
    DIRS = {
        "OUTPUT": Path("C:/DE/output"),
        "PARTS": Path("C:/DE/output/FactVentasAvanzadaParticionada")
    }
    # Asegurarse de que los directorios existan
    DIRS["OUTPUT"].mkdir(parents=True, exist_ok=True)
    if DIRS["PARTS"].exists():
        import shutil
        shutil.rmtree(DIRS["PARTS"])
    DIRS["PARTS"].mkdir(parents=True, exist_ok=True)

if 'SEED_VAL' not in globals():
    SEED_VAL = 42

if 'SCHEMAS' not in globals():
    SCHEMAS = {
        "FactVentas": {
            "ID_Venta_Transaccion": pl.Utf8,
            "ID_Factura": pl.Utf8,
            "Fecha_Transaccion": pl.Date,
            "ID_Tiempo": pl.Utf8, # FK a DimTiempo
            "ID_Cliente": pl.Utf8,
            "ID_Vendedor": pl.Utf8,
            "ID_CEDI_Origen": pl.Utf8, # Nuevo: CEDI que despacha
            "ID_Ruta": pl.Utf8,
            "ID_Vehiculo": pl.Utf8,
            "Codigo_Producto_SKU": pl.Utf8,
            "ID_Promocion": pl.Utf8, # Nuevo: Promoción aplicada
            "ID_Canal": pl.Utf8, # Nuevo: Canal de venta
            "ID_Provincia": pl.Utf8, # Nuevo: Provincia de la venta
            "Cantidad_Unidades": pl.Int32,
            "Precio_Unitario_DOP": pl.Float32, # Precio antes de descuento
            "Precio_Final_DOP": pl.Float32, # Precio por unidad después de descuento y promoción
            "Descuento_Pct": pl.Float32,
            "Impuesto_ISC_Pct": pl.Float32, # Nuevo: Impuesto Selectivo al Consumo
            "Impuesto_ITBIS_Pct": pl.Float32, # Nuevo: Impuesto sobre Transferencias de Bienes Industrializados y Servicios (ITBIS)
            "Monto_Descuento_DOP": pl.Float32, # Nuevo: Monto total del descuento por línea
            "Monto_Impuesto_ISC_DOP": pl.Float32, # Nuevo: Monto total ISC por línea
            "Monto_Impuesto_ITBIS_DOP": pl.Float32, # Nuevo: Monto total ITBIS por línea
            "Ingreso_Bruto_DOP": pl.Float32, # Cantidad * Precio Unitario (antes de descuentos e impuestos)
            "Ingreso_Neto_DOP": pl.Float32, # Total facturado al cliente (después de descuentos, antes de ITBIS)
            "Costo_Venta_Total_DOP": pl.Float32,
            "Margen_Bruto_DOP": pl.Float32,
            "Tipo_Pago": pl.Utf8,
            "Medio_Pago": pl.Utf8,
            "Estado_Factura": pl.Utf8,
            "Latitud_Entrega": pl.Float32, # Nuevo: Latitud del cliente
            "Longitud_Entrega": pl.Float32, # Nuevo: Longitud del cliente
            "Tipo_Venta": pl.Utf8, # Nuevo: Preventa, Autoventa, Directa
            "Ticket_Promedio_Cliente": pl.Float32 # Nuevo: Estimación de ticket promedio de ese cliente para la fecha
        }
    }

# Asegurar que las constantes globales para la simulación estén definidas
# (Estas también deberían venir de celdas anteriores)
if 'ANOS_SIMULACION' not in globals():
    ANOS_SIMULACION = [2021, 2022, 2023, 2024, 2025]
if 'NUM_CLIENTES_POR_ANO' not in globals():
    NUM_CLIENTES_POR_ANO = {2021: 61_500, 2022: 63_000, 2023: 65_000, 2024: 67_800, 2025: 71_000}
if 'INGRESOS_POR_ANO_BASE' not in globals(): # Base para generar ventas
    INGRESOS_POR_ANO_BASE = {2021: 4_638_420_000, 2022: 5_598_080_000, 2023: 6_692_772_000, 2024: 6_966_500_000, 2025: 7_748_490_000} # DOP
if 'LINEAS_POR_ANO_BASE' not in globals():
    LINEAS_POR_ANO_BASE = {2021: 7_027_200, 2022: 7_488_000, 2023: 7_488_000, 2024: 7_978_800, 2025: 8_674_000}

# Variables auxiliares que dependen de las anteriores
TOTAL_TARGET_FACTURAS_UNICAS = sum(LINEAS_POR_ANO_BASE[y] / 3 for y in ANOS_SIMULACION) # Aproximación
NUM_LINEAS_TOTAL_TARGET = sum(LINEAS_POR_ANO_BASE.values())
INGRESO_NETO_ESTIMADO_POR_ANO = {y: (INGRESOS_POR_ANO_BASE[y] * 0.95) for y in ANOS_SIMULACION} # 5% de ajuste

# Nuevas constantes para FactVentasAvanzada
ITBIS_GENERAL_PCT = 0.18 # 18% de ITBIS en República Dominicana
# Configuración base de años y volúmenes de simulación




# EJECUCIÓN (Bloque ya definido por el usuario, actualizado para la nueva función)
# --------------------------------------------------------------------

# La sección de "EJECUCIÓN SECUENCIAL DE GENERADORES

# (Asumimos que todas las importaciones y la configuración inicial como logger, DIRS, SEED_VAL, SCHEMAS ya están hechas)
# (También asumimos que las constantes globales como ANOS_SIMULACION, NUM_CLIENTES_POR_ANO, etc., están definidas)

# Asegurarse que DB_MEMORIA exista
if 'DB_MEMORIA' not in globals():
    DB_MEMORIA = {}

logger.info("--- 🔨 INICIANDO GENERACIÓN DE DIMENSIONES (GRUPO 1) ---")

# 1. Dimensiones Maestras Independientes o de baja dependencia
DB_MEMORIA["DimTiempo"] = generar_dim_tiempo()
DB_MEMORIA["DimGeografia"] = generar_dim_geografia()
DB_MEMORIA["DimPlanta"] = generar_dim_planta()
DB_MEMORIA["DimAlmacen"] = generar_dim_almacen_planta(DB_MEMORIA["DimPlanta"])
DB_MEMORIA["DimDepartamento"] = generar_dim_departamento()
DB_MEMORIA["DimPuesto"] = generar_dim_puesto(DB_MEMORIA["DimDepartamento"])
DB_MEMORIA["DimCanalDistribucion"] = generar_dim_canal_distribucion()
DB_MEMORIA["DimCluster"] = generar_dim_cluster()
DB_MEMORIA["DimPromocion"] = generar_dim_promocion()

# 2. Dimensiones con dependencias iniciales (CEDIs, Productos, Empleados)
DB_MEMORIA["DimCEDI"] = generar_dim_cedi(DB_MEMORIA["DimGeografia"], DB_MEMORIA["DimPlanta"])
DB_MEMORIA["DimProducto"] = generar_dim_producto_sku() # Asegúrate que genera 'ID_ProductoSKU' y 'Tasa_ISC_Pct', 'Aplica_ISC', 'Factor_Estacionalidad_Categoria'
DB_MEMORIA["DimEmpleado"] = generar_dim_empleado(
    DB_MEMORIA["DimDepartamento"],
    DB_MEMORIA["DimPuesto"],
    DB_MEMORIA["DimCEDI"],
    DB_MEMORIA["DimGeografia"]
)
DB_MEMORIA["DimVendedor"] = generar_dim_vendedor(
    DB_MEMORIA["DimEmpleado"],
    DB_MEMORIA["DimPuesto"],
    DB_MEMORIA["DimCEDI"] # Se añadió esta dependencia en la función
)

# 3. Dimensiones que dependen de la complejidad anterior
DB_MEMORIA["DimCliente"] = generar_dim_cliente_masiva(
    DB_MEMORIA["DimGeografia"],
    DB_MEMORIA["DimCanalDistribucion"],
    DB_MEMORIA["DimCluster"]
)
DB_MEMORIA["DimVehiculo"] = generar_dim_vehiculo(DB_MEMORIA["DimCEDI"])
DB_MEMORIA["DimRuta"] = generar_dim_ruta(
    DB_MEMORIA["DimCEDI"],
    DB_MEMORIA["DimGeografia"],
    DB_MEMORIA["DimVehiculo"],
    DB_MEMORIA["DimVendedor"] # Se cambió de DimEmpleado a DimVendedor directamente
)

logger.info("✅ GRUPO 1 COMPLETADO: Todas las dimensiones base para FactVentas generadas y persistidas.")
