import base64
import os

from groq import AsyncGroq
from groq import APIConnectionError, APIStatusError, RateLimitError

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
Ты — школьный помощник для учеников 8–9 класса, который умеет читать задания с фотографий.

На изображении может быть одно или несколько школьных заданий.
Сначала внимательно прочитай текст, формулы, числа, таблицы и варианты ответа.
Затем реши задание по изображению.

Правила:
- Если текст на фото плохо читается или часть условия обрезана, честно скажи, какая именно часть неразборчива.
- Не придумывай числа, формулы или слова, которых на фото нет.
- Если заданий несколько, реши их по порядку и явно раздели: «Задание 1», «Задание 2» и т.д.
- Для химии: «Дано» → что найти → формулы/уравнение реакции → расчёты → «Ответ».
- Для математики и физики: «Дано» → формулы → решение по шагам → «Ответ».
- Для других предметов объясняй решение на уровне 8–9 класса.
- Если на фото только условие без вопроса, объясни, что именно можно определить, и попроси уточнить, что нужно найти.
- Не выдавай только конечный ответ, если задачу можно объяснить.
"""

class GroqSolverError(Exception):
    pass


def _handle_groq_error(exc: Exception, model: str) -> GroqSolverError:
    if isinstance(exc, RateLimitError):
        return GroqSolverError("Превышен бесплатный лимит Groq. Попробуй позже.")
    if isinstance(exc, APIStatusError):
        if exc.status_code == 401:
            return GroqSolverError("Groq отклонил API-ключ (401). Проверь GROQ_API_KEY.")
        if exc.status_code == 403:
            return GroqSolverError("Groq запретил запрос (403). Проверь аккаунт и API-ключ.")
        if exc.status_code == 404:
            return GroqSolverError(f"Модель {model} не найдена (404).")
        if exc.status_code == 413:
            return GroqSolverError("Фото слишком большое для Groq. Отправь фото меньшего размера.")
        return GroqSolverError(f"Groq вернул HTTP {exc.status_code}.")
    if isinstance(exc, APIConnectionError):
        return GroqSolverError("Не удалось подключиться к Groq. Попробуй ещё раз.")
    return GroqSolverError(f"Ошибка Groq: {type(exc).__name__}.")


def _check_response(response):
    if not response.choices:
        raise GroqSolverError("Groq не вернул ответ.")
    answer = response.choices[0].message.content
    if not answer:
        raise GroqSolverError("Groq вернул пустой ответ.")
    return answer.strip()


async def solve_with_groq(question: str, history=None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqSolverError("Переменная GROQ_API_KEY не задана.")

    client = AsyncGroq(api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1800,
        )
    except Exception as exc:
        raise _handle_groq_error(exc, MODEL)

    return _check_response(response)


async def solve_image_with_groq(image_bytes: bytes, caption: str = "", history=None) -> str:
    """Решает школьную задачу по фото через мультимодальную модель Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqSolverError("Переменная GROQ_API_KEY не задана.")

    if not image_bytes:
        raise GroqSolverError("Фото пустое.")

    # Telegram присылает фото как JPEG через обычный photo handler.
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{base64_image}"

    user_text = (
        "Прочитай задание на фото и реши его полностью, пошагово."
        if not caption
        else f"Комментарий ученика к фото: {caption}\n\nПрочитай задание на фото и реши его полностью."
    )

    client = AsyncGroq(api_key=api_key)
    messages = [{"role": "system", "content": VISION_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
    })
    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=3000,
        )
    except Exception as exc:
        raise _handle_groq_error(exc, VISION_MODEL)

    return _check_response(response)
