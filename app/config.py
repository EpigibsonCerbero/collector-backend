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
