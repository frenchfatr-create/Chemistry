import os
from groq import AsyncGroq
from groq import APIConnectionError, APIStatusError, RateLimitError

MODEL = "openai/gpt-oss-120b"

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

class GroqSolverError(Exception):
    pass

async def solve_with_groq(question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqSolverError("Переменная GROQ_API_KEY не задана.")

    client = AsyncGroq(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=1800,
        )
    except RateLimitError:
        raise GroqSolverError("Превышен бесплатный лимит Groq. Попробуй позже.")
    except APIStatusError as exc:
        if exc.status_code == 401:
            raise GroqSolverError("Groq отклонил API-ключ (401). Проверь GROQ_API_KEY.")
        if exc.status_code == 403:
            raise GroqSolverError("Groq запретил запрос (403). Проверь аккаунт и API-ключ.")
        if exc.status_code == 404:
            raise GroqSolverError(f"Модель {MODEL} не найдена (404).")
        raise GroqSolverError(f"Groq вернул HTTP {exc.status_code}.")
    except APIConnectionError:
        raise GroqSolverError("Не удалось подключиться к Groq. Попробуй ещё раз.")
    except Exception as exc:
        raise GroqSolverError(f"Ошибка Groq: {type(exc).__name__}.")

    if not response.choices:
        raise GroqSolverError("Groq не вернул ответ.")
    answer = response.choices[0].message.content
    if not answer:
        raise GroqSolverError("Groq вернул пустой ответ.")
    return answer.strip()
