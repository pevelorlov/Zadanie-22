# MCP-серверы Дней 17–18

Каждый сервер — отдельный локальный процесс со своим набором инструментов:

```powershell
.\.venv\Scripts\python.exe -m mcp_servers.open_meteo.server
.\.venv\Scripts\python.exe -m mcp_servers.openalex.server
.\.venv\Scripts\python.exe -m mcp_servers.openaq.server
.\.venv\Scripts\python.exe -m mcp_servers.scheduler.server
```

Они используют транспорт `stdio`, поэтому обычно запускаются MCP-клиентом, а
не вручную в отдельном окне. Open‑Meteo и OpenAlex работают без регистрации.
Для OpenAQ v3 нужен бесплатный ключ в переменной `OPENAQ_API_KEY`. Не добавляйте
ключ в репозиторий — храните его в `.env`.

К приложению DeepSeek Agent подключены Open‑Meteo и Scheduler. Scheduler при
обычном запуске стартует автоматически и использует `data/scheduler.sqlite3`.
OpenAlex и OpenAQ пока оставлены независимыми для следующих дней.
