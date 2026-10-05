from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler
from .topics import TOPICS

PREFIX = "chem:"

def keyboard():
    rows = [
        [InlineKeyboardButton(v["title"], callback_data=f"{PREFIX}topic:{k}")]
        for k, v in TOPICS.items()
    ]
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="subject:chemistry")])
    return InlineKeyboardMarkup(rows)

async def show_menu(query):
    await query.edit_message_text(
        "🧪 <b>Химия 8–9 класса</b>\n\nВыбери тему:",
        reply_markup=keyboard(),
        parse_mode="HTML",
    )

def register(app):
    async def handler(update, context):
        query = update.callback_query
        await query.answer()

        if query.data == "chem:menu":
            await show_menu(query)
            return

        key = query.data.split(":")[-1]
        topic = TOPICS.get(key)

        if topic:
            await query.edit_message_text(
                f"<b>{topic['title']}</b>\n\n{topic['text']}",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton(
                        "⬅️ К химии",
                        callback_data="chem:menu"
                    )]
                ]),
            )

    app.add_handler(
        CallbackQueryHandler(handler, pattern=r"^chem:")
    )
