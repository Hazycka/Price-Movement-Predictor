"""
Training services: linear probing, LoRA, и применение артефактов к моделям.

Модули:
  artifact_loader.py — загрузка адаптеров поверх foundation-модели по artifact_id
  dataset.py         — PyTorch Dataset/DataLoader для walk-forward training
  loss.py            — weighted pinball loss (та же что в backtest)
  head_trainer.py    — обучение только новой output head (linear probing)
  lora_trainer.py    — обучение LoRA-адаптеров через PEFT
  saver.py           — сохранение артефактов на диск + регистрация в БД
"""
from .artifact_loader import ArtifactLoader

__all__ = ["ArtifactLoader"]
