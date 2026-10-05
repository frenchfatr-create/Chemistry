import os
import json
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from subjects.chemistry import router as chemistry_router
from ai_solver import solve_with_groq, solve_image_with_groq, GroqSolverError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
TOKEN = os.getenv("BOT_TOKEN")

# Последние сообщения каждого пользователя. Храним 20 сообщений
# (включая ответы бота), чтобы ИИ видел контекст диалога.
MAX_HISTORY_MESSAGES = 20
MEMORY_FILE = os.getenv("CHAT_MEMORY_FILE", "chat_memory.json")

def load_histories():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}

chat_histories = load_histories()

def save_histories():
    temp_file = MEMORY_FILE + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(chat_histories, file, ensure_ascii=False, indent=2)
        os.replace(temp_file, MEMORY_FILE)
    except OSError:
        logger.exception("Не удалось сохранить память диалогов")

def get_history(user_id):
    key = str(user_id)
    return chat_histories.setdefault(key, [])

def add_history(user_id, role, content):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]
    save_histories()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /start начинает новый диалог и очищает старый контекст.
    if update.effective_user:
        chat_histories.pop(str(update.effective_user.id), None)
        save_histories()
    keyboard = [[InlineKeyboardButton("🧪 Химия 8–9 класс", callback_data="subject:chemistry")]]
    await update.message.reply_text(
        "📚 <b>Школьный помощник 8–9 класса</b>\n\n"
        "Выбери предмет, отправь задачу текстом или пришли фото задания — "
        "я попробую прочитать условие и решить его.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
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

    user_id = update.effective_user.id
    history = list(get_history(user_id))
    await update.message.chat.send_action("typing")
    try:
        answer = await solve_with_groq(question, history=history)
        add_history(user_id, "user", question)
        add_history(user_id, "assistant", answer)
        await update.message.reply_text(answer)
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


async def photo_solver(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Скачивает фото из Telegram и отправляет его в vision-модель Groq."""
    photo = update.message.photo[-1]  # самое большое доступное разрешение
    caption = (update.message.caption or "").strip()

    await update.message.chat.send_action("typing")
    try:
        telegram_file = await context.bot.get_file(photo.file_id)
        image_bytes = await telegram_file.download_as_bytearray()

        user_id = update.effective_user.id
        history = list(get_history(user_id))
        answer = await solve_image_with_groq(
            bytes(image_bytes), caption=caption, history=history
        )
        photo_context = (
            f"[Фото задания] Комментарий: {caption}"
            if caption
            else "[Фото задания] Пользователь отправил фото задания."
        )
        add_history(user_id, "user", photo_context)
        add_history(user_id, "assistant", answer)
        await update.message.reply_text(answer)
    except GroqSolverError as exc:
        logger.error("Groq vision solver error: %s", exc)
        await update.message.reply_text(
            "⚠️ Не удалось прочитать или решить задание с фото.\n\n"
            f"Причина: {exc}"
        )
    except Exception:
        logger.exception("Unexpected photo solver error")
        await update.message.reply_text(
            "⚠️ Не удалось обработать фото.\n"
            "Попробуй отправить более чёткое фото, где полностью видно условие."
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

    # Фото обрабатываем отдельно от обычного текста.
    app.add_handler(MessageHandler(filters.PHOTO, photo_solver))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_solver))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
