from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "collector-manager-api"
    env: str = "dev"

    # MongoDB (puerto 27018: el 27017 ya lo usa finance-mongo-1)
    mongo_url: str = "mongodb://localhost:27018/?replicaSet=rs0"
    mongo_db: str = "collector_manager"

    # JWT
    jwt_secret: str = "dev-only-secret-change-me-32bytes-min!!"
    jwt_alg: str = "HS256"
    access_ttl_min: int = 60
    refresh_ttl_days: int = 30

    # Ids de usuario (coma) con permiso para editar el catálogo global.
    admin_user_ids: str = ""

    # Premium / cuota de escaneos. La cuota nace apagada hasta que la app con
    # el paywall esté en manos de los usuarios.
    scan_limit_enabled: bool = False
    premium_user_ids: str = ""  # ids (coma) con Premium forzado (pruebas/regalos)
    free_daily_scans: int = 20
    premium_grace_days: int = 3
    revenuecat_secret_key: str = ""
    revenuecat_webhook_auth: str = ""

    # Autocompletado de catálogo: crea piezas ausentes tras un escaneo manual.
    ai_catalog_enabled: bool = False
    ai_catalog_daily_cap: int = 10
    # Lecturas PAGADAS (no altas): por usuario y día, y global por día (corte
    # de emergencia del gasto). Se cuentan antes de cada lectura.
    ai_catalog_daily_ai_calls: int = 30
    ai_catalog_global_daily_calls: int = 500
    # Rango válido del número Mini GT (MGT00001..). Fuera de rango se ignora.
    ai_catalog_mgt_max: int = 2000
    ai_catalog_max_top_score: float = 0.75
    ai_catalog_receipt_ttl_s: int = 600
    # Barreras anti-duplicado (inclinadas a NO crear): cobertura del nombre de
    # un candidato textual por la lectura, y similitud del recorte del coche
    # contra el índice completo (vistas aisladas incluidas).
    ai_catalog_dup_text_overlap: float = 0.8
    ai_catalog_dup_embed_score: float = 0.90

    # Bootstrap admin (dev). Con la contraseña por defecto NO se crea y, si ya
    # existe con ella, se desactiva al arrancar (estaba publicada en el repo).
    seed_admin_email: str = "admin@collector.local"
    seed_admin_password: str = "AdminDev123!"
    seed_admin_username: str = "admin"

    # Seed del catálogo Mini GT scrapeado
    catalog_seed_path: str = "../data/minigt-catalog.json"
    # Seed del catálogo Hot Wheels / Mattel scrapeado
    hotwheels_catalog_seed_path: str = "../data/hotwheels-catalog.json"
    # Seed multi-marca diecast extra (Inno64, Tarmac, Kaido…) — cualquier escala
    diecast_extra_seed_path: str = "../data/diecast-extra-catalog.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
