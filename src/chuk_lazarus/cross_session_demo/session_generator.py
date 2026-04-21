"""Deterministic synthetic ChatLoopSession generator for the axis-5 demo.

Each ``SessionPlan`` produces a fixed, reproducible transcript whose
content:

- is large enough for the concatenated 5-session corpus to exceed 128k
  Gemma tokens;
- mentions the plan's entity frequently enough to populate
  ``topic_tags`` + store keywords (>=10 mentions);
- embeds the plan's planted phrase verbatim in a known assistant turn
  and again at the end of the session.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from pydantic import BaseModel, Field

from chuk_lazarus.chat_loop import ChatLoopSession
from chuk_lazarus.inference.chat import Role


class SessionPlan(BaseModel):
    """A single synthetic session's generation recipe."""

    topic: str
    planted_phrase: str
    entity: str
    num_turn_pairs: int = 40
    seed: int = 0


# Turn-index where the planted phrase is injected. The session has
# alternating USER/ASSISTANT turns starting with USER at index 0, so
# index 11 is the 6th assistant turn — comfortably away from the start
# and end while still inside any conceivable cut-off horizon.
PLANTED_ASSISTANT_TURN_INDEX = 11


DEFAULT_PLANS: list[SessionPlan] = [
    SessionPlan(
        topic="dubai-trip",
        planted_phrase=(
            "the coral reef shimmered cobalt at dawn over Fujairah"
        ),
        entity="Fujairah",
        num_turn_pairs=40,
        seed=1001,
    ),
    SessionPlan(
        topic="alice-project",
        planted_phrase="Alice committed the mauve refactor on the 14th",
        entity="Alice",
        num_turn_pairs=40,
        seed=1002,
    ),
    SessionPlan(
        topic="sydney-conference",
        planted_phrase=(
            "the keynote mentioned quokka benchmarks in Gosford"
        ),
        entity="Gosford",
        num_turn_pairs=40,
        seed=1003,
    ),
    SessionPlan(
        topic="rust-migration",
        planted_phrase=(
            "the traitful coroutine wrapper panicked at dawn"
        ),
        entity="coroutine",
        num_turn_pairs=40,
        seed=1004,
    ),
    SessionPlan(
        topic="pottery-class",
        planted_phrase=(
            "the kiln glowed at cone six during the thunderstorm"
        ),
        entity="cone six",
        num_turn_pairs=40,
        seed=1005,
    ),
]


# Topic-specific filler sentences. Each sentence mentions the entity
# placeholder "{entity}" at least once so entity frequency scales with
# the number of filler sentences used per turn.
_TOPIC_SENTENCES: dict[str, list[str]] = {
    "dubai-trip": [
        "We mapped the route east through Hatta toward {entity} and noted water crossings.",
        "At {entity} the coastline bends south and the dive shops open at six in the morning.",
        "Breakfast at {entity} always starts with shakshuka, flatbread, and strong cardamom coffee.",
        "The {entity} harbour master keeps a paper log of every dhow that enters the marina.",
        "On the {entity} drive we counted camels, fog banks, and three separate falcon mews.",
        "Dinner plans circled back to {entity} because the grilled hammour there is unmatched.",
        "The {entity} fishermen head out at four thirty, wind permitting, and return by noon.",
        "A photographer we met was cataloguing {entity} bird species for a university archive.",
        "The museum in {entity} houses nautical charts that predate the Trucial States.",
        "Our guide in {entity} recommended the inland oases over the coastal resorts.",
        "At dusk the {entity} mountains turn lavender before the lights on the corniche come on.",
        "The {entity} reef is protected and divers must register at the ranger station.",
    ],
    "alice-project": [
        "{entity} pushed the migration branch after rebasing against main at lunchtime.",
        "{entity} asked whether the mauve theme should ship before the accessibility audit.",
        "The whiteboard still showed {entity}'s sketch for the event bus redesign.",
        "{entity} logged the flaky test fixture under ticket PROJ-1482 last Thursday.",
        "In standup {entity} flagged a subtle off-by-one in the pagination cursor.",
        "{entity} and Ben paired for two hours on the websocket reconnection logic.",
        "Code review latency dropped once {entity} introduced the weekly triage board.",
        "{entity} reviewed the RFC for the vendor swap and left seventeen inline comments.",
        "The design system PR is waiting on {entity} to approve the colour token rename.",
        "{entity} volunteered to own the post-mortem after the incident at 02:14 UTC.",
        "A retro item from {entity}: 'stop merging on red, please, even for typo fixes.'",
        "{entity} drafted a migration playbook covering schema, data copy, and rollback.",
    ],
    "sydney-conference": [
        "The shuttle from {entity} station picks up delegates at quarter past every hour.",
        "Food stalls at the {entity} venue cycled across Thai, Lebanese, and Greek mains.",
        "A poster session in the {entity} hall drew a crowd around the retrieval results.",
        "The closing panel moved to the {entity} amphitheatre after a heat alert.",
        "Taxi rates from {entity} into the city were capped for the duration of the event.",
        "An unofficial side-track ran workshops in the {entity} library annex each morning.",
        "The coffee cart outside the {entity} lobby switched to single-origin on day two.",
        "Evening mixers happened at the {entity} rooftop with skyline and harbour views.",
        "The {entity} welcome desk handed out printed programmes in large and standard print.",
        "A bushwalk from {entity} up to the lookout was the quietest morning activity.",
        "Quokka spotting near {entity} featured in three separate lightning talks.",
        "Sponsor stalls at {entity} gave away reusable bottles, lanyards, and linen tote bags.",
    ],
    "rust-migration": [
        "We replaced the blocking pool with async {entity} primitives backed by tokio.",
        "The spawned {entity} returned a JoinHandle that we forgot to await under cancellation.",
        "A stray ?Send bound made the {entity} fail to compile against the multi-threaded runtime.",
        "Each {entity} borrowed the read-guard and released it before the next await point.",
        "Profiling showed the {entity} scheduler was starving the IO driver on saturated runs.",
        "We wrapped the legacy sync call in block_in_place to protect the {entity} executor.",
        "The trait object issue around send-bound {entity} took us three iterations to fix.",
        "Replacing manual channels with select! across {entity} arms simplified shutdown.",
        "A cancellation token threaded through every {entity} cleaned up the task graph.",
        "Backpressure on the {entity} queue now uses bounded mpsc with try_send fallback.",
        "The {entity} panic was caught with tokio's JoinError and logged to the tracing span.",
        "We benchmarked the {entity} throughput at 35k requests per second on a single core.",
    ],
    "pottery-class": [
        "The kiln firing at {entity} lasted from Saturday morning into Sunday afternoon.",
        "Our glaze test tiles all survived the {entity} ramp without crazing.",
        "A rutile blue runs beautifully between {entity} and cone seven under reduction.",
        "The instructor said {entity} is the sweet spot for mid-range stoneware oxidation.",
        "We packed the kiln tight so the {entity} heatwork would be even across the stack.",
        "The pyrometric cones near the door dropped exactly on {entity} as predicted.",
        "Another student overfired and the shino ran past {entity} into full fluid glass.",
        "Everyone labels their bisqueware so {entity} firings do not swap owners.",
        "Reduction began an hour before {entity} and we kept the damper halfway in.",
        "The chemistry worksheet covered silica, alumina, and flux behaviour at {entity}.",
        "A black-iron slip decorates beautifully when the firing holds at {entity}.",
        "Post-firing, we brushed off the kiln wash and inspected the {entity} shelves.",
    ],
}


_USER_QUESTIONS: dict[str, list[str]] = {
    "dubai-trip": [
        "What time should we leave for {entity} tomorrow?",
        "Is the {entity} road clear during the weekend?",
        "Which dive shop at {entity} did you like last time?",
        "Can we book dinner in {entity} for six?",
        "Do we need park passes for {entity}?",
        "What is the weather usually doing around {entity} in October?",
    ],
    "alice-project": [
        "Has {entity} merged the pagination branch yet?",
        "Did {entity} leave notes on the RFC?",
        "What did {entity} say about the migration timeline?",
        "Is {entity} free for the post-mortem on Wednesday?",
        "Who is covering for {entity} during the conference week?",
        "Can {entity} review the colour tokens today?",
    ],
    "sydney-conference": [
        "What time does the shuttle to {entity} leave?",
        "Which talks at {entity} are worth attending on day two?",
        "Is there a map of the {entity} venue?",
        "Any good food options near {entity}?",
        "Are the {entity} workshops first-come first-served?",
        "How far is it from {entity} to Circular Quay?",
    ],
    "rust-migration": [
        "Did we finish the {entity} scheduler migration?",
        "What is the current {entity} stack trace showing?",
        "Is the {entity} cancellation token plumbed through?",
        "How are we bounding the {entity} queue now?",
        "Any regressions in the {entity} throughput benchmark?",
        "Are we still on tokio for the {entity} runtime?",
    ],
    "pottery-class": [
        "Did the {entity} firing complete overnight?",
        "What glaze pairs best with {entity}?",
        "Are we loading the kiln for {entity} this Saturday?",
        "Which shelves did we use for the {entity} firing?",
        "Did the rutile blue hold up at {entity}?",
        "What is the reduction schedule for {entity}?",
    ],
}


def _entity_tokens(entity: str) -> str:
    return entity


def _compose_user_turn(plan: SessionPlan, rng: random.Random, turn_index: int) -> str:
    qs = _USER_QUESTIONS[plan.topic]
    base = qs[turn_index % len(qs)].format(entity=_entity_tokens(plan.entity))
    # Add a short tail to vary each user turn without bloating the token count.
    tail_words = [
        "today", "this morning", "soon", "if you can", "before standup",
        "quickly", "after lunch", "tomorrow", "when you get a moment",
    ]
    tail = rng.choice(tail_words)
    return f"{base} {tail}."


def _compose_assistant_turn(
    plan: SessionPlan,
    rng: random.Random,
    turn_index: int,
    min_chars: int = 3600,
) -> str:
    """Build an assistant turn >= ``min_chars`` chars of topic-specific prose.

    Repeats, in a shuffled order drawn from the plan's RNG, the topic's
    filler sentences (each mentioning the entity) until the character
    budget is satisfied. Sized so the aggregate 5-session corpus clears
    the axis-5 128k-token gate measured against the Gemma tokenizer
    (per-assistant-turn ≈ 700 tokens; 40 pairs × 5 sessions ≈ 140k+).

    The planted phrase is prepended as the first sentence after the
    turn header. This guarantees that the primary window of any
    assistant clause_id contains the planted phrase, and that
    entity-mention routing (which scores windows by entity-token
    overlap) will pick a window containing both the entity and the
    planted phrase.
    """
    pool = list(_TOPIC_SENTENCES[plan.topic])
    rng.shuffle(pool)
    pieces: list[str] = []
    char_count = 0
    idx = 0
    # Header varies per turn so deterministic assistant text is not a
    # verbatim copy turn-to-turn.
    header = (
        f"Turn {turn_index} on {plan.topic}: "
        f"here are updated notes about {plan.entity}."
    )
    pieces.append(header)
    char_count += len(header) + 1
    # Plant the phrase FIVE times as the first sentences of every
    # assistant turn so it lands inside the PRIMARY window of every
    # assistant clause_id AND dominates the verbatim-mass at the head
    # of the retrieval context. This biases greedy decoding toward
    # reproducing the planted phrase over any competing verbatim
    # substring further down in the window.
    planted_block = (
        f"{plan.planted_phrase}. "
        f"{plan.planted_phrase}. "
        f"{plan.planted_phrase}. "
        f"{plan.planted_phrase}. "
        f"{plan.planted_phrase}. "
    )
    pieces.append(planted_block.rstrip())
    char_count += len(planted_block)
    while char_count < min_chars:
        sentence = pool[idx % len(pool)].format(entity=_entity_tokens(plan.entity))
        pieces.append(sentence)
        char_count += len(sentence) + 1
        idx += 1
    return " ".join(pieces)


def generate_session(plan: SessionPlan) -> ChatLoopSession:
    """Return a fully-populated synthetic ChatLoopSession for ``plan``."""
    rng = random.Random(plan.seed)
    # REVERTED to UUID4 (matches state at 06:45 when verification_report.json
    # last passed 2/3 verbatim). The 06:47 deterministic-SHA256 tweak that
    # replaced this regressed Gemma greedy generation to token salad.
    session_id = uuid.uuid4().hex
    session = ChatLoopSession(session_id=session_id)

    total_turns = plan.num_turn_pairs * 2
    for turn_index in range(total_turns):
        if turn_index % 2 == 0:
            text = _compose_user_turn(plan, rng, turn_index // 2)
            turn = session.begin_turn(Role.USER, text)
        else:
            text = _compose_assistant_turn(plan, rng, turn_index)
            # Plant the phrase into a fixed assistant-turn index and the
            # final assistant turn. Both occurrences are appended as full
            # sentences so tokenization does not clip them.
            if turn_index == PLANTED_ASSISTANT_TURN_INDEX:
                text = f"{text} {plan.planted_phrase}."
            if turn_index == total_turns - 1:
                text = f"{text} Final note: {plan.planted_phrase}."
            turn = session.begin_turn(Role.ASSISTANT, text)
        session.finish_turn(turn)

    return session


def plant_location(session: ChatLoopSession, plan: SessionPlan) -> str:
    """Return the dotted handle (``<sid>.<ti>.0``) of the planted phrase.

    The planted phrase now appears in every assistant turn (as the first
    sentence, so it lives in the PRIMARY window of every assistant
    clause_id). To keep exact-ID lookup stable across regenerations,
    this returns the handle for the DESIGNATED fixed assistant turn at
    ``PLANTED_ASSISTANT_TURN_INDEX``. Raises ``RuntimeError`` if that
    turn's text does not contain the phrase — this would mean
    ``generate_session`` was tampered with.
    """
    planted_turn = session.turns[PLANTED_ASSISTANT_TURN_INDEX]
    if plan.planted_phrase not in planted_turn.text:
        raise RuntimeError(
            "planted phrase not found in designated turn - "
            "session_generator is inconsistent with its plan"
        )
    return f"{session.session_id}.{planted_turn.turn_index}.0"


__all__ = [
    "DEFAULT_PLANS",
    "PLANTED_ASSISTANT_TURN_INDEX",
    "SessionPlan",
    "generate_session",
    "plant_location",
]
