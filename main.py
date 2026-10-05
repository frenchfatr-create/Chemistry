import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from subjects.chemistry import router as chemistry_router
from ai_solver import solve_with_groq, GroqSolverError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("🧪 Химия 8–9 класс", callback_data="subject:chemistry")]]
    await update.message.reply_text(
        "📚 <b>Школьный помощник 8–9 класса</b>\n\n"
        "Выбери предмет или просто отправь мне задачу текстом.",
        reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML"
    )

async def subject_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "subject:chemistry":
        await chemistry_router.show_menu(query)

async def text_solver(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if not question:
        return
    await update.message.chat.send_action("typing")
    try:
        await update.message.reply_text(await solve_with_groq(question))
    except GroqSolverError as exc:
        logger.error("Groq solver error: %s", exc)
        await update.message.reply_text(
            "⚠️ Не удалось решить задачу через ИИ.\n\n"
            f"Причина: {exc}\n\n"
            "Если проблема повторяется, проверь GROQ_API_KEY в Bothost."
        )
    except Exception:
        logger.exception("Unexpected bot error")
        await update.message.reply_text(
            "⚠️ Произошла внутренняя ошибка бота.\n"
            "Подробности записаны в лог Bothost."
        )

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN не задан. Добавь его в Bothost.")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY не задан. Добавь его в Bothost.")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(subject_callback, pattern=r"^subject:"))
    chemistry_router.register(app)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_solver))
    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
