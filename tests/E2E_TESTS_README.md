# End-to-End Тесты для StorageManager

## Быстрый Старт

## Список E2E Тестов

| # | Тест | Что Проверяет |
|---|------|---------------|
| 1 | `test_e2e_upload_download_cycle` | Полный цикл: создание → upload → delete → download с checksums |
| 2 | `test_e2e_finalize_artifact` | Workflow finalize_artifact: temp → final + cloud |
| 3 | `test_e2e_write_mode_cloud_sync` | Автоматическая синхронизация при записи файла |
| 4 | `test_e2e_cache_behavior` | Поведение кэша (cache_files=True/False) |
| 5 | `test_e2e_large_file_upload` | Загрузка больших файлов (10MB) с checksums |
| 6 | `test_e2e_concurrent_uploads` | Параллельная загрузка 5 файлов |
| 7 | `test_e2e_manifest_push` | Загрузка манифестов в облако |
| 8 | `test_e2e_error_recovery` | Retry механизм и восстановление |
| 9 | `test_e2e_empty_file` | Обработка пустых файлов |
| 10 | `test_e2e_unicode_content` | Unicode контент (русский, китайский, арабский) |
| 11 | `test_e2e_nested_directories` | Файлы во вложенных директориях |

---

### Ошибка: "S3 is not enabled"

**Причина**: S3 не настроен в `.env`

**Решение**:
1. Проверьте что `.env` файл существует
2. Убедитесь что `CCLANG_S3_ENABLE=true`
3. Проверьте что переменные окружения загружаются:
   ```bash
   python -c "from cclang.config.s3 import load_s3_config; print(load_s3_config())"
   ```

### Ошибка: "Access Denied" или "Invalid credentials"

**Причина**: Неверные или истекшие credentials

**Решение**:
1. Проверьте `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`
2. Убедитесь что ключи имеют права на bucket
3. Проверьте что bucket существует

### Ошибка: "Bucket not found"

**Причина**: Bucket не существует или неверное имя

**Решение**:
1. Проверьте `CCLANG_S3_BUCKET` в `.env`
2. Создайте bucket если его нет
3. Убедитесь в правильности региона (`AWS_REGION`)

### Ошибка: "Connection timeout"

**Причина**: Проблемы с сетью или endpoint

**Решение**:
1. Проверьте интернет соединение
2. Для Yandex Cloud убедитесь что endpoint правильный (автоматически `storage.yandexcloud.net`)
3. Проверьте firewall настройки

---

## Проверка Конфигурации

Простой скрипт для проверки что S3 настроен корректно:

```python
from pathlib import Path
from cclang.config.s3 import load_s3_config
from cclang.core.storage import StorageManager, CloudConfig
from cclang.io.fs import LocalConfig

# Загрузить конфигурацию
s3_cfg = load_s3_config()

print("S3 Configuration:")
print(f"  Enabled: {s3_cfg.enable}")
print(f"  Bucket: {s3_cfg.bucket}")
print(f"  Prefix: {s3_cfg.root_prefix}")
print(f"  Region: {s3_cfg.region}")
print(f"  Has credentials: {bool(s3_cfg.access_key and s3_cfg.secret_key)}")

if s3_cfg.enable and s3_cfg.bucket:
    # Попробовать создать StorageManager
    local_cfg = LocalConfig(
        base_path=Path("data"),
        save_local=True,
        cache_files=True,
        temp_base=Path("data/temp")
    )

    cloud_cfg = CloudConfig(
        enable=True,
        s3_config=s3_cfg
    )

    try:
        sm = StorageManager(local_cfg, cloud_cfg)
        print("\n✅ StorageManager создан успешно!")

        # Проверить подключение к S3
        # (это не делает реальных запросов, только инициализацию)
        print("✅ S3Store инициализирован")

        sm.close()
        print("✅ Cleanup выполнен")

    except Exception as e:
        print(f"\n❌ Ошибка при создании StorageManager: {e}")
else:
    print("\n⚠️  S3 не настроен. E2E тесты будут пропущены.")
```

Сохраните как `check_s3_config.py` и запустите:
```bash
python check_s3_config.py
```

---

## Расширенное Использование

### Запустить тесты с подробным выводом

```bash
pytest tests/test_storage_e2e.py -v -s --log-cli-level=DEBUG
```

### Запустить только быстрые тесты

```bash
# Пропустить тест с большим файлом (10MB)
pytest tests/test_storage_e2e.py -v -k "not large_file"
```

### Запустить с coverage

```bash
pytest tests/test_storage_e2e.py -v --cov=src/cclang/core/storage --cov-report=html
```

### Параллельный запуск (с pytest-xdist)

```bash
pip install pytest-xdist
pytest tests/test_storage_e2e.py -v -n auto
```

---

## Структура E2E Теста

Каждый E2E тест следует этой структуре:

```python
def test_e2e_example(storage_manager, temp_base):
    """
    E2E Test: Краткое описание

    Workflow:
    1. Шаг 1
    2. Шаг 2
    3. Шаг 3
    """
    # 1. Создать тестовые данные
    test_file = temp_base / "test.txt"
    test_file.write_text("test data")

    # 2. Выполнить операцию
    storage_manager.push_data(Path("test.txt"), blocking=True)
    time.sleep(1)  # Подождать eventual consistency

    # 3. Проверить результат
    assert storage_manager._exists_cloud(Path("test.txt"))

    # 4. Cleanup (если нужно)
    test_file.unlink()

    print("✅ Test passed")
```

---


 Обычно 10-30 секунд для всех 11 тестов, в зависимости от скорости интернета.
