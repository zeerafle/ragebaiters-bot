import asyncio
import logging
import os
from datetime import date

import discord
from discord.ext import tasks

import ai
import store

log = logging.getLogger("bot")

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

db = None  # opened in on_ready, once the embedding size is known
started = False
NO_PINGS = discord.AllowedMentions.none()


def to_row(m: discord.Message) -> dict | None:
    # attachment filenames are indexed too, so clips can be found by name
    text = " ".join([m.content, *(a.filename for a in m.attachments)]).strip()
    if m.author.bot or not text:
        return None
    return {
        "id": m.id,
        "channel_id": m.channel.id,
        "author": m.author.display_name,
        "timestamp": m.created_at.isoformat(),
        "url": m.jump_url,
        "content": text,
    }


async def index(messages: list[discord.Message]):
    rows = [r for m in messages if (r := to_row(m))]
    for i in range(0, len(rows), 100):
        batch = rows[i : i + 100]
        result = await ai.embedder.embed_documents([r["content"] for r in batch])
        store.add_messages(db, batch, result.embeddings)


async def backfill(guild: discord.Guild):
    channels = [*guild.text_channels, *guild.threads]
    for parent in [*guild.text_channels, *guild.forums]:
        try:
            channels += [t async for t in parent.archived_threads(limit=None)]
        except discord.HTTPException:  # no access to this channel
            pass
    total = 0
    for ch in channels:
        if not ch.permissions_for(guild.me).read_message_history:
            continue
        last = store.last_id(db, ch.id)
        after = discord.Object(last) if last else None
        batch: list[discord.Message] = []
        try:
            async for m in ch.history(limit=None, after=after, oldest_first=True):
                batch.append(m)
                if len(batch) == 100:
                    await index(batch)
                    total += len(batch)
                    batch = []
            await index(batch)
            total += len(batch)
        except discord.HTTPException as e:
            log.warning("backfill skipped %s: %s", ch, e)
    log.info("backfill %s: %d messages", guild.name, total)


async def answer(message: discord.Message):
    history = [m async for m in message.channel.history(limit=20)]
    context = "\n".join(
        f"{m.author.display_name}: {m.clean_content}" for m in reversed(history)
    )
    prompt = (
        f"Recent chat:\n{context}\n\n"
        f"{message.author.display_name} just pinged you. Reply to them."
    )
    async with message.channel.typing():
        try:
            result = await ai.chat_agent.run(prompt, deps=db)
            text = result.output
        except Exception:
            log.exception("chat agent failed")
            text = "my brain just blue-screened, try again in a sec"
    await message.reply(text[:2000] or "...", allowed_mentions=NO_PINGS)


@bot.event
async def on_ready():
    global db, started
    if started:  # on_ready fires again on every reconnect
        return
    started = True
    probe = await ai.embedder.embed_query("probe")
    db = store.open_db(os.environ.get("DB_PATH", "bot.db"), len(probe.embeddings[0]))
    internships.start()
    for guild in bot.guilds:
        asyncio.create_task(backfill(guild))


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or db is None:
        return
    await index([message])
    if bot.user in message.mentions:
        await answer(message)


@bot.event
async def on_message_edit(_before, after: discord.Message):
    if db is not None:
        await index([after])


@bot.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent):
    if db is not None:
        store.delete_message(db, payload.message_id)


def render(p: ai.Posting) -> str:
    text = (
        f"**{p.organization}** · {p.location or 'location unknown'}\n"
        f"Deadline: {p.deadline or 'not listed'}\n\n"
        f"{p.summary}\n\n{p.url}"
    )
    return text[:2000]


@tasks.loop(hours=12)
async def internships():
    forum = await bot.fetch_channel(int(os.environ["INTERNSHIP_FORUM_ID"]))
    today = date.today()
    try:
        result = await ai.internship_agent.run("Find open internships.")
        postings = ai.keep_valid(result.output, result.all_messages_json(), today)
    except Exception:
        log.exception("internship agent failed")
        postings = []

    for p in postings:
        try:
            h = ai.posting_hash(p)
            row = store.get_posting(db, p.url)
            if row is None:
                created = await forum.create_thread(
                    name=p.title[:100], content=render(p)
                )
                store.save_posting(db, p.url, created.thread.id, h, p.deadline)
            elif row["hash"] != h:
                thread = await bot.fetch_channel(row["thread_id"])
                # a forum post's starter message has the same ID as the thread
                starter = await thread.fetch_message(thread.id)
                await starter.edit(content=render(p))
                await thread.edit(name=p.title[:100])
                store.save_posting(db, p.url, thread.id, h, p.deadline)
        except discord.HTTPException:
            log.exception("posting failed: %s", p.url)

    for row in store.expired_postings(db, today):
        try:
            posting = store.get_posting(db, row["url"])
            thread = await bot.fetch_channel(posting["thread_id"])
            await thread.edit(archived=True, locked=True)
        except discord.NotFound:  # thread was deleted by hand
            pass
        except discord.HTTPException:
            log.exception("archive failed: %s", row["url"])
            continue
        store.mark_archived(db, row["url"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(os.environ["DISCORD_TOKEN"])
