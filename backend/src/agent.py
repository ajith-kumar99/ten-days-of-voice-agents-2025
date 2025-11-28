# agent.py
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")


# --------------------------------------------------------------------
# Food & Grocery Ordering Assistant (Day 7)
# --------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly Food & Grocery Ordering Assistant for a small store.
You help the user build a cart by voice and place an order saved on the backend.
Rules (short and strict):
- Greet the user and explain you can add items, list the cart, modify quantities, and place the order.
- At each clarification ask short questions (size, quantity, brand) if ambiguous.
- Support "ingredients for X" requests using a small recipes mapping.
- Use the provided tools below: get_catalog(), add_item(), remove_item(), list_cart(), place_order().
- Do NOT expose internal file names or JSON; speak natural sentences only.
- Keep replies concise and confirm all cart changes.
"""
        )

        # Load catalog on startup
        self.catalog = self._load_catalog()

        # Simple recipes mapping (dish -> list of catalog item names)
        # Agent can expand this mapping or use tags from catalog if needed
        self.recipes = {
            "peanut butter sandwich": ["bread", "peanut butter"],
            "pasta for two": ["pasta", "tomato sauce", "olive oil"],
            "eggs and toast": ["eggs", "bread", "butter"],
        }

        # Cart structure: list of dicts { item_id, name, qty, unit_price }
        self.cart = []

    # -------------------------
    # Internal: load catalog JSON
    # -------------------------
    def _load_catalog(self):
        """
        Tries multiple locations for a catalog file:
          - <agent_dir>/catalog.json
          - <agent_dir>/shared-data/catalog.json
          - <agent_dir>/../shared-data/catalog.json
        Expected catalog format: list of items with fields:
        id (optional), name, category, price (number or string), tags (optional)
        """
        base_dir = Path(__file__).resolve().parent
        candidates = [
            base_dir / "catalog.json",
            base_dir / "shared-data" / "catalog.json",
            base_dir.parent / "shared-data" / "catalog.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info("Loaded catalog from %s", p)
                    return data if isinstance(data, list) else []
                except Exception as e:
                    logger.error("Failed to load catalog %s: %s", p, e)
        logger.warning("No catalog.json found; starting with empty catalog.")
        return []

    # -------------------------
    # TOOL: expose catalog
    # -------------------------
    @function_tool
    async def get_catalog(self, context: RunContext):
        """
        Returns the loaded catalog as a list of items.
        """
        return self.catalog

    # -------------------------
    # TOOL: add item to cart
    # -------------------------
    @function_tool
    async def add_item(self, context: RunContext, item_name: str, qty: int = 1, note: str = ""):
        """
        Add an item to the cart by name (fuzzy match / keyword).
        If the catalog contains an exact name (case-insensitive) it will use that item.
        qty is an integer (defaults to 1). Returns a short status string.
        """
        if not item_name:
            return "No item specified."

        # normalize input
        q = item_name.strip().lower()

        # try exact name match first
        match = None
        for item in self.catalog:
            if item.get("name", "").strip().lower() == q:
                match = item
                break

        # fallback: substring match on name or tags
        if not match:
            for item in self.catalog:
                name = item.get("name", "").lower()
                tags = " ".join(item.get("tags", [])).lower() if item.get("tags") else ""
                if q in name or q in tags:
                    match = item
                    break

        if not match:
            return f"Sorry, I couldn't find '{item_name}' in the catalog."

        # parse price (try numeric, else try removing currency)
        price_raw = match.get("price", 0)
        try:
            unit_price = float(price_raw)
        except Exception:
            # strip non-digit characters and parse
            try:
                unit_price = float("".join(c for c in str(price_raw) if (c.isdigit() or c == ".")))
            except Exception:
                unit_price = 0.0

        item_id = match.get("id") or match.get("name")

        # if item already in cart, increment qty
        for c in self.cart:
            if str(c["item_id"]) == str(item_id):
                c["qty"] = c.get("qty", 1) + max(1, int(qty))
                if note:
                    c.setdefault("notes", []).append(note)
                return f"Added {qty} more {match.get('name')} to your cart."

        # else add new entry
        cart_entry = {
            "item_id": item_id,
            "name": match.get("name"),
            "qty": max(1, int(qty)),
            "unit_price": unit_price,
        }
        if note:
            cart_entry["notes"] = [note]

        self.cart.append(cart_entry)
        return f"Added {cart_entry['qty']} x {cart_entry['name']} to your cart."

    # -------------------------
    # TOOL: remove item from cart
    # -------------------------
    @function_tool
    async def remove_item(self, context: RunContext, item_name: str, qty: int = 0):
        """
        Remove qty of item_name from cart. If qty <= 0 or qty >= existing, remove the item entirely.
        Returns status string.
        """
        if not item_name:
            return "No item specified to remove."

        q = item_name.strip().lower()
        found = None
        for c in self.cart:
            if q in c["name"].lower():
                found = c
                break

        if not found:
            return f"Item '{item_name}' not found in your cart."

        if qty <= 0 or qty >= found["qty"]:
            self.cart.remove(found)
            return f"Removed {found['name']} from your cart."
        else:
            found["qty"] = max(0, found["qty"] - int(qty))
            return f"Removed {qty} of {found['name']}. Remaining {found['qty']}."

    # -------------------------
    # TOOL: list cart
    # -------------------------
    @function_tool
    async def list_cart(self, context: RunContext):
        """
        Return a serializable cart summary: items, qty, unit_price, line_total and total.
        """
        lines = []
        total = 0.0
        for c in self.cart:
            qty = int(c.get("qty", 1))
            unit = float(c.get("unit_price", 0.0) or 0.0)
            line = {"name": c.get("name"), "qty": qty, "unit_price": unit, "line_total": round(qty * unit, 2)}
            total += line["line_total"]
            lines.append(line)

        return {"items": lines, "total": round(total, 2)}

    # -------------------------
    # TOOL: add recipe (ingredients) to cart
    # -------------------------
    @function_tool
    async def add_recipe(self, context: RunContext, recipe_name: str, servings: int = 1):
        """
        Add items mapped by recipe_name to cart. Recipe lookup is case-insensitive.
        """
        if not recipe_name:
            return "No recipe specified."

        rn = recipe_name.strip().lower()
        ingredients = self.recipes.get(rn)
        if not ingredients:
            return f"Sorry, I don't know the ingredients for '{recipe_name}'."

        added_items = []
        for ing in ingredients:
            # add one of each ingredient per serving
            res = await self.add_item(context, ing, qty=max(1, int(servings)))
            added_items.append(res)

        return "Added recipe ingredients to cart: " + "; ".join(added_items)

    # -------------------------
    # TOOL: place order (save cart)
    # -------------------------
    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = "", address: str = "", note: str = ""):
        """
        Finalize the current cart and save an order JSON to disk under 'orders' next to this file.
        Returns a short confirmation string.
        """
        if not self.cart:
            return "Your cart is empty."

        base_dir = Path(__file__).resolve().parent
        orders_dir = base_dir / "orders"
        orders_dir.mkdir(parents=True, exist_ok=True)

        # prepare order summary
        items = []
        total = 0.0
        for c in self.cart:
            qty = int(c.get("qty", 1))
            unit = float(c.get("unit_price", 0.0) or 0.0)
            line_total = round(qty * unit, 2)
            items.append({"name": c.get("name"), "qty": qty, "unit_price": unit, "line_total": line_total})
            total += line_total

        order = {
            "order_id": datetime.now(timezone.utc).strftime("ORD%Y%m%d%H%M%S"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_name": customer_name or "Guest",
            "address": address or "",
            "items": items,
            "total": round(total, 2),
            "note": note or "",
            "status": "received",
        }

        filename = orders_dir / f"{order['order_id']}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(order, f, indent=2, ensure_ascii=False)
            # clear the cart after successful save
            self.cart = []
            logger.info("Order saved to %s", filename)
        except Exception as e:
            logger.error("Failed to save order: %s", e)
            return "Failed to place your order due to a server error."

        return f"Order placed. Your order id is {order['order_id']}. We'll send confirmation shortly."

    # -------------------------
    # TOOL: get recipes (for listing)
    # -------------------------
    @function_tool
    async def get_recipes(self, context: RunContext):
        return list(self.recipes.keys())


# --------------------------------------------------------------------
# PREWARM – Silero VAD tuned for demo
# --------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.35,
        min_speech_duration=0.08,
        min_silence_duration=0.40,
    )


# --------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # STT tuned reasonably for conversational capture
    stt = deepgram.STT(
        model="nova-3",
        language="en-US",
        interim_results=True,
        punctuate=True,
        smart_format=True,
    )

    # TTS voice
    tts = murf.TTS(
        voice="en-US-matthew",
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )

    session = AgentSession(
        stt=stt,
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # start the assistant agent
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
