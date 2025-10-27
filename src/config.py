import json
import os
from typing import Dict, Any, Optional

class Config:
    """Класс для загрузки и работы с конфигурацией"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Загружает конфигурацию из JSON файла
        
        Args:
            config_path: путь к файлу конфигурации
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
            
        self._validate_config()
    
    def _validate_config(self):
        """Проверяет корректность конфигурации"""
        required_sections = ['data', 'model', 'training', 'paths', 'system']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Отсутствует секция '{section}' в конфигурации")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Получает значение из конфигурации
        
        Args:
            section: секция конфигурации (data, model, training, etc.)
            key: ключ в секции
            default: значение по умолчанию если ключ не найден
        """
        return self._config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """
        Устанавливает значение в конфигурации
        
        Args:
            section: секция конфигурации
            key: ключ в секции  
            value: новое значение
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save(self, config_path: Optional[str] = None):
        """
        Сохраняет конфигурацию в файл
        
        Args:
            config_path: путь для сохранения (если None, использует исходный путь)
        """
        if config_path is None:
            config_path = "config.json"
            
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=4, ensure_ascii=False)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Возвращает секцию data"""
        return self._config['data']
    
    @property
    def model(self) -> Dict[str, Any]:
        """Возвращает секцию model"""
        return self._config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Возвращает секцию training"""
        return self._config['training']
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Возвращает секцию paths"""
        return self._config['paths']
    
    @property
    def generation(self) -> Dict[str, Any]:
        """Возвращает секцию generation"""
        return self._config['generation']
    
    @property
    def system(self) -> Dict[str, Any]:
        """Возвращает секцию system"""
        return self._config['system']
    
    def __str__(self) -> str:
        """Возвращает строковое представление конфигурации"""
        return json.dumps(self._config, indent=2, ensure_ascii=False)


def load_config(config_path: str = "config.json") -> Config:
    """
    Удобная функция для загрузки конфигурации
    
    Args:
        config_path: путь к файлу конфигурации
        
    Returns:
        объект Config
    """
    return Config(config_path)