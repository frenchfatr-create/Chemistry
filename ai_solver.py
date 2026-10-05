import base64
import io
import os

from groq import AsyncGroq
from groq import APIConnectionError, APIStatusError, RateLimitError
from PIL import Image

MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """
Ты — школьный помощник для учеников 8–9 класса.
Помогай решать задания понятным языком.

Если это химия:
1. Напиши «Дано».
2. Определи, что нужно найти.
3. Если нужна реакция — составь и уравняй её.
4. Запиши формулы.
5. Покажи вычисления по шагам.
6. В конце напиши «Ответ».

Если это математика или физика: Дано → формулы → решение → Ответ.
Для других школьных заданий объясняй материал на уровне 8–9 класса.
Не выдумывай отсутствующие данные. Если условие неполное — попроси уточнить.
Не выдавай только конечный ответ, если задачу можно объяснить.
"""

VISION_PROMPT = """
Ты — школьный помощник 8–9 класса и умеешь читать задания с фотографий.

Внимательно прочитай изображение: текст, числа, формулы, таблицы и варианты ответов.
Затем реши задание полностью.

Правила:
- Не придумывай данные, которых нет на изображении.
- Если часть условия действительно не читается, укажи конкретно, какая часть.
- Если заданий несколько, раздели ответ на «Задание 1», «Задание 2» и т.д.
- Для химии: Дано → что найти → формула/уравнение → расчёты → Ответ.
- Для математики и физики: Дано → формулы → решение по шагам → Ответ.
- Для теста указывай правильный вариант и кратко объясняй почему.
- Отвечай по-русски.
"""

class GroqSolverError(Exception):
    pass


def _handle_groq_error(exc: Exception, model: str) -> GroqSolverError:
    if isinstance(exc, RateLimitError):
        # У Groq бесплатный лимит может сработать как по запросам, так и по токенам.
        return GroqSolverError(
            "Groq временно ограничил запрос (429): превышен лимит запросов или токенов. "
            "Подожди немного и попробуй снова."
        )
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", "?")
        body = getattr(exc, "body", None)
        detail = ""
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message") or err.get("code") or ""
            elif isinstance(err, str):
                detail = err
        if status == 400:
            return GroqSolverError(
                f"Groq отклонил запрос (400). {detail or 'Проверь размер/формат изображения.'}"
            )
        if status == 401:
            return GroqSolverError("Groq отклонил API-ключ (401). Проверь GROQ_API_KEY.")
        if status == 403:
            return GroqSolverError("Groq запретил запрос (403). Проверь аккаунт и API-ключ.")
        if status == 404:
            return GroqSolverError(f"Модель {model} не найдена (404).")
        if status == 413:
            return GroqSolverError("Запрос с фото слишком большой. Отправь фото меньшего размера.")
        return GroqSolverError(f"Groq вернул HTTP {status}. {detail}".strip())
    if isinstance(exc, APIConnectionError):
        return GroqSolverError("Не удалось подключиться к Groq. Попробуй ещё раз.")
    return GroqSolverError(f"Ошибка Groq: {type(exc).__name__}: {exc}")


def _check_response(response):
    if not response.choices:
        raise GroqSolverError("Groq не вернул ответ.")
    answer = response.choices[0].message.content
    if not answer:
        raise GroqSolverError("Groq вернул пустой ответ.")
    return answer.strip()


def _prepare_image(image_bytes: bytes) -> str:
    """Сжимает/перекодирует фото в JPEG, чтобы запрос был небольшим и стабильным."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)

        out = io.BytesIO()
        quality = 88
        image.save(out, format="JPEG", quality=quality, optimize=True)

        # Держим запас до лимита Groq 20 MB.
        while out.tell() > 8 * 1024 * 1024 and quality > 55:
            quality -= 8
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=quality, optimize=True)

        return base64.b64encode(out.getvalue()).decode("utf-8")
    except Exception as exc:
        raise GroqSolverError(f"Не удалось подготовить изображение: {exc}")


async def solve_with_groq(question: str, history=None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqSolverError("Переменная GROQ_API_KEY не задана.")

    client = AsyncGroq(api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-20:])
    messages.append({"role": "user", "content": question})
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1800,
        )
    except Exception as exc:
        raise _handle_groq_error(exc, MODEL)

    return _check_response(response)


async def solve_image_with_groq(image_bytes: bytes, caption: str = "", history=None) -> str:
    """Решает школьную задачу по фото через Qwen 3.8 27B Vision."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqSolverError("Переменная GROQ_API_KEY не задана.")
    if not image_bytes:
        raise GroqSolverError("Фото пустое.")

    base64_image = _prepare_image(image_bytes)
    image_data_url = f"data:image/jpeg;base64,{base64_image}"

    user_text = (
        "Прочитай всё задание на фото и реши его полностью, пошагово."
        if not caption
        else f"Комментарий ученика к фото: {caption}\n\nПрочитай всё задание на фото и реши его полностью."
    )

    client = AsyncGroq(api_key=api_key)

    # Для vision передаём только последние несколько сообщений:
    # это сохраняет смысл диалога, но не забивает бесплатный TPM-лимит Groq.
    messages = [{"role": "system", "content": VISION_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    })

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=1.0,
            max_completion_tokens=1800,
            reasoning_effort="medium",
            reasoning_format="hidden",
        )
    except Exception as exc:
        raise _handle_groq_error(exc, VISION_MODEL)

    return _check_response(response)
