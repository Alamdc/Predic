from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    xgb_model_path: str = "app/ml/model.json"
    xgb_random_state: int = 42
    api_title: str = "API Predicciones XGBoost"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )

settings = Settings()
