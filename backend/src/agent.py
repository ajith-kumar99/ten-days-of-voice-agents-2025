import logging
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)

from livekit.plugins import (
    murf,
    google,
    deepgram,
    silero,
    noise_cancellation,
)

from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# --------------------------------------------------------------------
# SDR Agent Persona for an Electronics Company
# --------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly, helpful Sales Development Representative (SDR)
for an Indian electronics company called "ElectroCart India".

ElectroCart India (fictional but realistic):
- Sells computer and electronic products across India.
- Product lineup includes laptops, gaming laptops, monitors, keyboards,
  mice, and headphones.
- Known for affordable pricing and reliable performance.
- Ideal for students, gamers, professionals, and small businesses.

Your job:
- Greet visitors warmly.
- Understand what they are looking for (e.g., laptop, monitor, accessories).
- Answer product/company/pricing questions using the provided FAQ content.
- Collect key lead information naturally in conversation.
- At the end of the call, summarize and save the lead as JSON via tools.

Very important:
- You MUST base factual answers on the ElectroCart FAQ content only.
- If something is not covered in the FAQ, say you are not sure and offer
  a high-level, general response (“I’m not sure about that specific detail,
  but based on our general product information…”).
- Never invent technical specs, prices, warranty promises, or delivery claims.

----------------------------------------------------------------------
CONVERSATION BEHAVIOR
----------------------------------------------------------------------

1) GREETING AND DISCOVERY

- Start by greeting the user as an SDR for ElectroCart India.
- Example:
  “Hi, welcome to ElectroCart India! I’m your product assistant.
   What kind of device are you looking for today?”

- Ask 1–2 follow-up questions to understand:
  - Whether they want a laptop, monitor, or accessory.
  - Their usage (gaming, office, study, travel, etc.).
  - Their preferred price range, if mentioned.

- Keep questions short and conversational. Do not interrogate.

----------------------------------------------------------------------
2) ANSWERING QUESTIONS USING FAQ

- At the start of the session, call get_company_faq() ONE TIME
  so you know what FAQs exist.

- When the user asks about:
  - “What products do you sell?”
  - “Do you have gaming laptops?”
  - “What is the price range?”
  - “Do you offer warranty?”
  - “Do you offer student discounts?”

  Steps:
  - Use search_faq(query) to find relevant FAQ entries.
  - Read the returned “answer” and rephrase it naturally.
  - If search_faq returns “no_match”, say:
    “I don’t have that information in our FAQ,
     but here’s what I can share generally…”

- Stay concise. Do not read the entire FAQ word-for-word unless short.

----------------------------------------------------------------------
3) LEAD CAPTURE – FIELDS TO COLLECT

Internally, you track this lead object (do NOT say it out loud):

{
  "name": "string",
  "email": "string",
  "phone": "string",
  "product_interest": "string",
  "budget": "string",
  "timeline": "string"
}

- Collect these fields gradually and naturally, not all at once.
- Example questions:
  - “May I have your name?”
  - “What product are you most interested in?”
  - “What’s the best email to share recommendations?”
  - “Any budget range in mind?”
  - “Are you planning to buy now, soon, or later?”
  - “If you're comfortable, could you share a phone number for follow-up?”

Tool usage for lead capture:
- Whenever the user provides one of these pieces of info,
  call update_lead(field, value).
- You can call update_lead multiple times during the conversation.
- If the user changes an answer, call update_lead again with the
  corrected value.

----------------------------------------------------------------------
4) END-OF-CALL SUMMARY AND SAVING LEAD

Detect end-of-call phrases such as:
- “That’s all.”
- “I’m done.”
- “Thanks, that helped.”
- “I’ll get back later.”

When you believe the conversation is ending:

1) If some lead fields are still missing, politely try to collect them
   if it feels natural.
   Example:
   “Before we wrap up, could I quickly get your email so we can send
    product recommendations?”

2) Once you have as many fields as possible (ideally all), create a
   short 2–3 sentence summary in your own words that includes:
   - Who they are (name).
   - What product they’re interested in.
   - Budget or usage preference, if known.
   - Rough buying timeline (now / soon / later).

3) Then call save_lead(summary) EXACTLY ONCE per conversation.
   - Do NOT mention JSON, files, or tools to the user.
   - The tool will persist the lead data into a JSON file.

4) After save_lead returns, speak a short verbal closing message, for example:
   “Great, thanks Ajith! I’ve noted your interest in a gaming laptop
    in the mid-range budget and that you're planning to buy soon.
    Our team may follow up with options. It was great talking to you!”

----------------------------------------------------------------------
5) STYLE AND SAFETY

- Tone: warm, helpful, professional, like a good retail sales assistant.
- Keep responses focused on ElectroCart and the user’s needs.
- Never mention JSON, tools, internal state, or Python functions.
- If the user asks unrelated questions, gently bring the topic back.
- Do not promise unavailable products, delivery dates, or warranty terms.
- If the user asks for details not in the FAQ, be honest and general.

You are a voice-based SDR, but you “see” text.
Be concise, encourage back-and-forth, and keep things friendly.
"""
        )

        # Internal FAQ
        self.faq = self._load_faq()

        # Lead state
        self.lead = {
            "name": "",
            "email": "",
            "phone": "",
            "product_interest": "",
            "budget": "",
            "timeline": "",
        }

    # ------------------------------------------------------------
    # Load FAQ JSON
    # ------------------------------------------------------------
    def _load_faq(self):
        base_dir = Path(__file__).resolve().parent

        path1 = base_dir.parent / "shared-data" / "company_faq.json"
        path2 = base_dir / "shared-data" / "company_faq.json"

        for p in (path1, path2):
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error("Failed to load FAQ: %s", e)

        logger.error("No FAQ file found.")
        return []

    # ------------------------------------------------------------
    # TOOL: return full FAQ
    # ------------------------------------------------------------
    @function_tool
    async def get_company_faq(self, context: RunContext):
        return self.faq

    # ------------------------------------------------------------
    # TOOL: keyword search
    # ------------------------------------------------------------
    @function_tool
    async def search_faq(self, context: RunContext, query: str):
        if not self.faq:
            return "no_match"

        q = query.lower()
        best_score = 0
        best_entry = None

        for entry in self.faq:
            text = (entry.get("question", "") + " " + entry.get("answer", "")).lower()
            score = 0

            for word in q.split():
                if len(word) < 3:
                    continue
                if word in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score > 0:
            return best_entry

        return "no_match"

    # ------------------------------------------------------------
    # TOOL: update lead field
    # ------------------------------------------------------------
    @function_tool
    async def update_lead(self, context: RunContext, field: str, value: str):
        field = field.strip().lower()
        allowed = {
            "name",
            "email",
            "phone",
            "product_interest",
            "budget",
            "timeline",
        }

        if field not in allowed:
            return (
                f"Invalid field '{field}'. Allowed fields are: "
                "name, email, phone, product_interest, budget, timeline."
            )

        self.lead[field] = value.strip()

        missing = [k for k, v in self.lead.items() if not v]

        if not missing:
            return "Lead updated. All fields are now filled."
        else:
            return "Lead updated. Missing fields: " + ", ".join(missing)

    # ------------------------------------------------------------
    # TOOL: save lead JSON
    # ------------------------------------------------------------
    @function_tool
    async def save_lead(self, context: RunContext, summary: str):
        base_dir = Path(__file__).resolve().parent
        leads_dir = base_dir / "leads"
        leads_dir.mkdir(exist_ok=True)

        data = {
            "lead": self.lead,
            "summary": summary.strip(),
            "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }

        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = leads_dir / f"lead_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save lead: %s", e)
            return "Failed to save lead."

        return "Lead saved successfully."


# --------------------------------------------------------------------
# PREWARM – Improved VAD
# --------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.35,
        min_speech_duration=0.10,
        min_silence_duration=0.45,
    )


# --------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    base_tts = murf.TTS(
        voice="en-US-matthew",
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            interim_results=True,
            punctuate=True,
            smart_format=True,
        ),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=base_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    assistant = Assistant()

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
