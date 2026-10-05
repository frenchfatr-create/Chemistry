import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from subjects.chemistry import router as chemistry_router
from ai_solver import solve_with_groq

logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🧪 Химия 8–9 класс", callback_data="subject:chemistry")],
    ]
    await update.message.reply_text(
        "📚 Школьный помощник 8–9 класса\n\n"
        "Выбери предмет или просто отправь мне задачу текстом.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def subject_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "subject:chemistry":
        await chemistry_router.show_menu(query)

async def text_solver(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    if not question:
        return

    await update.message.chat.send_action("typing")

    try:
        answer = await solve_with_groq(question)
        await update.message.reply_text(answer)
    except Exception as exc:
        logging.exception("Groq error: %s", exc)
        await update.message.reply_text(
            "⚠️ Не удалось получить решение. Проверь, что GROQ_API_KEY "
            "правильно добавлен в настройках Bothost."
        )

def main():
    if not TOKEN:
        raise RuntimeError("Не задан BOT_TOKEN.")

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Не задан GROQ_API_KEY.")

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        CallbackQueryHandler(subject_callback, pattern=r"^subject:")
    )
    chemistry_router.register(app)

    # Любой обычный текст отправляем ИИ.
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_solver)
    )

    app.run_polling()

if __name__ == "__main__":
    main()
