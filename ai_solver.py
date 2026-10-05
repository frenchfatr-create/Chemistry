import os
from groq import AsyncGroq

SYSTEM_PROMPT = """
Ты — школьный помощник для учеников 8–9 класса.

Помогай решать школьные задания понятным языком.
Если это химия:
- запиши Дано;
- определи, что нужно найти;
- при необходимости составь и уравняй реакцию;
- используй формулы;
- показывай вычисления по шагам;
- в конце обязательно напиши Ответ.

Если это математика или физика:
- запиши Дано;
- нужные формулы;
- решение по шагам;
- Ответ.

Не выдумывай данные, которых нет в условии.
Если условие неполное или непонятное — попроси пользователя уточнить его.
Не выдавай только конечный ответ, если можно объяснить решение.
"""

async def solve_with_groq(question: str) -> str:
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

    completion = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=1800,
    )

    return completion.choices[0].message.content.strip()
