"""Unit tests for the deterministic synthetic session generator."""

from __future__ import annotations

from chuk_lazarus.cross_session_demo.session_generator import (
    DEFAULT_PLANS,
    PLANTED_ASSISTANT_TURN_INDEX,
    SessionPlan,
    generate_session,
    plant_location,
)
from chuk_lazarus.inference.chat import Role


def test_default_plans_has_five_distinct_sessions() -> None:
    assert len(DEFAULT_PLANS) == 5
    topics = [p.topic for p in DEFAULT_PLANS]
    phrases = [p.planted_phrase for p in DEFAULT_PLANS]
    assert len(set(topics)) == 5
    assert len(set(phrases)) == 5


def test_generate_session_is_deterministic_for_same_seed() -> None:
    plan = DEFAULT_PLANS[0]
    s_a = generate_session(plan)
    s_b = generate_session(plan)
    # Session IDs are UUID4 so they will differ, but the text stream
    # must be identical when seeded with the same plan.
    texts_a = [t.text for t in s_a.turns]
    texts_b = [t.text for t in s_b.turns]
    assert texts_a == texts_b


def test_generate_session_different_plans_differ() -> None:
    a = generate_session(DEFAULT_PLANS[0])
    b = generate_session(DEFAULT_PLANS[1])
    texts_a = " ".join(t.text for t in a.turns)
    texts_b = " ".join(t.text for t in b.turns)
    assert texts_a != texts_b


def test_planted_phrase_appears_in_planted_turn() -> None:
    plan = DEFAULT_PLANS[0]
    session = generate_session(plan)
    planted_turn = session.turns[PLANTED_ASSISTANT_TURN_INDEX]
    assert planted_turn.role == Role.ASSISTANT
    assert plan.planted_phrase in planted_turn.text


def test_planted_phrase_also_in_final_turn() -> None:
    plan = DEFAULT_PLANS[2]
    session = generate_session(plan)
    assert plan.planted_phrase in session.turns[-1].text


def test_plant_location_returns_valid_dotted_handle() -> None:
    plan = DEFAULT_PLANS[3]
    session = generate_session(plan)
    handle = plant_location(session, plan)
    sid, ti, ci = handle.split(".")
    assert sid == session.session_id
    assert int(ti) >= 0
    assert int(ci) == 0
    # The handle must actually point at a turn whose text contains the phrase.
    turn = session.turns[int(ti)]
    assert plan.planted_phrase in turn.text


def test_assistant_turns_are_substantial() -> None:
    plan = DEFAULT_PLANS[1]
    session = generate_session(plan)
    assistant_turns = [t for t in session.turns if t.role == Role.ASSISTANT]
    # All assistant turns should be prose blocks (>= 1500 chars).
    assert all(len(t.text) >= 1500 for t in assistant_turns)


def test_entity_appears_many_times_per_session() -> None:
    """Every plan's entity must appear >=10 times across its session."""
    for plan in DEFAULT_PLANS:
        session = generate_session(plan)
        full = " ".join(t.text for t in session.turns)
        # Case-insensitive: some templates put the entity at sentence head.
        count = full.lower().count(plan.entity.lower())
        assert count >= 10, f"{plan.topic}: entity {plan.entity!r} only {count} mentions"


def test_turn_count_matches_plan() -> None:
    plan = SessionPlan(
        topic="dubai-trip",
        planted_phrase="the coral reef shimmered cobalt at dawn over Fujairah",
        entity="Fujairah",
        num_turn_pairs=5,
        seed=42,
    )
    session = generate_session(plan)
    assert len(session.turns) == 10
    assert session.turns[0].role == Role.USER
    assert session.turns[1].role == Role.ASSISTANT


def test_planted_phrase_in_every_assistant_turn() -> None:
    """The planted phrase must appear >=5 times in EVERY assistant turn.

    Five repetitions at the HEAD of every assistant turn ensure the
    planted phrase is both (a) inside the PRIMARY window of every
    assistant clause_id and (b) the dominant verbatim mass at the top
    of the retrieval context. This biases greedy decoding toward
    reproducing the planted phrase rather than any competing verbatim
    substring further down the window.
    """
    for plan in DEFAULT_PLANS:
        session = generate_session(plan)
        assistant_turns = [t for t in session.turns if t.role == Role.ASSISTANT]
        for t in assistant_turns:
            # At least 5 occurrences of the planted phrase per turn.
            occurrences = t.text.count(plan.planted_phrase)
            assert occurrences >= 5, (
                f"{plan.topic}: planted phrase only appears "
                f"{occurrences} times in assistant turn "
                f"{t.turn_index} (need >= 5)"
            )
            # The phrase must appear at the very HEAD of the turn,
            # immediately after the "Turn N on <topic>: here are
            # updated notes about <entity>." preamble (<=70 chars).
            # A 150-char slice comfortably covers preamble + one
            # full phrase occurrence for the longest plan.
            head = t.text[:150]
            assert plan.planted_phrase in head, (
                f"{plan.topic}: planted phrase not within first 150 "
                f"chars of assistant turn {t.turn_index} "
                f"(head repr: {head!r})"
            )
