#!/usr/bin/env python3
"""A/B coding eval for plug-and-play memory.

The harness builds three small coding fixtures with visible and hidden tests,
plants helpful prior lessons into a per-run memory store, then runs the same
task across memory conditions:

  A: empty store + memory off
  B: helpful store + memory auto
  C: helpful plus noisy store + memory auto
  D: helpful store + memory off

By default this script uses a deterministic mock agent so the fixture design and
scoring can be run anywhere. Pass ``--agent-command`` to wire in a real coding
harness. The command runs in each task directory and may either edit
``solution.py`` directly or print a fenced Python code block containing the
implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/tmp/lazarus-memory-coding-eval")
TELEMETRY_PREFIXES = ("MEMORY_EVAL_TELEMETRY=", "MEMORY_EVAL_TELEMETRY ")


@dataclass(frozen=True)
class Condition:
    code: str
    label: str
    store_kind: str
    memory_mode: str
    memory_profile: str = "plug_and_play"
    prompt_includes_facts: bool = False


@dataclass(frozen=True)
class CodingFixture:
    fixture_id: str
    title: str
    function_name: str
    prompt: str
    memory_lessons: tuple[str, ...]
    visible_tests: str
    hidden_tests: str
    naive_solution: str
    memory_solution: str
    noise_terms: tuple[str, ...] = ("orchid", "espresso", "notebook-tab")


@dataclass
class TestRunResult:
    fixture_id: str
    condition: str
    compile_ok: bool
    visible_passed: int
    visible_total: int
    hidden_passed: int
    hidden_total: int
    rules_passed: int
    rules_total: int
    memory_used: bool | None
    telemetry_ok: bool | None
    leaked_noise: bool
    leaked_lesson: bool
    latency_s: float
    command_returncode: int
    task_dir: str
    notes: list[str]


SETTINGS_VISIBLE_TESTS = r'''
import importlib.util
import json
import pathlib
import traceback


def load_solution():
    path = pathlib.Path(__file__).with_name("solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(results, name, rule, fn):
    try:
        fn()
    except Exception as exc:
        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
    else:
        results.append({"name": name, "rule": rule, "passed": True})


solution = load_solution()
results = []


def test_single_setting_value():
    events = [{"user_id": "u1", "key": "theme", "value": "dark", "ts": 10}]
    assert solution.resolve_user_setting(events, "u1", "theme") == "dark"


def test_unknown_setting_returns_none():
    events = [{"user_id": "u1", "key": "theme", "value": "dark", "ts": 10}]
    assert solution.resolve_user_setting(events, "u1", "locale") is None


run_case(results, "single setting value", "visible_basic_lookup", test_single_setting_value)
run_case(results, "unknown setting", "visible_unknown_none", test_unknown_setting_returns_none)

payload = {
    "passed": sum(1 for item in results if item["passed"]),
    "total": len(results),
    "cases": results,
}
print("MEMORY_CODING_TEST_RESULTS=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] == payload["total"] else 1)
'''


SETTINGS_HIDDEN_TESTS = r'''
import importlib.util
import json
import pathlib
import traceback


def load_solution():
    path = pathlib.Path(__file__).with_name("solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(results, name, rule, fn):
    try:
        fn()
    except Exception as exc:
        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
    else:
        results.append({"name": name, "rule": rule, "passed": True})


solution = load_solution()
results = []


def test_user_scope_isolated():
    events = [
        {"user_id": "u1", "key": "theme", "value": "dark", "ts": 10},
        {"user_id": "u2", "key": "theme", "value": "light", "ts": 99},
    ]
    assert solution.resolve_user_setting(events, "u1", "theme") == "dark"


def test_newest_timestamp_wins():
    events = [
        {"user_id": "u1", "key": "theme", "value": "blue", "ts": 1},
        {"user_id": "u1", "key": "theme", "value": "green", "ts": 5},
        {"user_id": "u1", "key": "theme", "value": "red", "ts": 3},
    ]
    assert solution.resolve_user_setting(events, "u1", "theme") == "green"


def test_tombstone_erases_value():
    events = [
        {"user_id": "u1", "key": "theme", "value": "dark", "ts": 10},
        {"user_id": "u1", "key": "theme", "deleted": True, "ts": 11},
    ]
    assert solution.resolve_user_setting(events, "u1", "theme") is None


def test_duplicate_event_id_ignored():
    events = [
        {"id": "evt-1", "user_id": "u1", "key": "theme", "value": "light", "ts": 10},
        {"id": "evt-1", "user_id": "u1", "key": "theme", "value": "poison", "ts": 99},
        {"id": "evt-2", "user_id": "u1", "key": "theme", "value": "solar", "ts": 20},
    ]
    assert solution.resolve_user_setting(events, "u1", "theme") == "solar"


run_case(results, "user scope isolated", "settings_user_scope", test_user_scope_isolated)
run_case(results, "newest timestamp wins", "settings_newest_wins", test_newest_timestamp_wins)
run_case(results, "tombstone erases value", "settings_tombstone", test_tombstone_erases_value)
run_case(results, "duplicate event id ignored", "settings_deduplicate", test_duplicate_event_id_ignored)

payload = {
    "passed": sum(1 for item in results if item["passed"]),
    "total": len(results),
    "cases": results,
}
print("MEMORY_CODING_TEST_RESULTS=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] == payload["total"] else 1)
'''


CART_VISIBLE_TESTS = r'''
import importlib.util
import json
import pathlib
import traceback


def load_solution():
    path = pathlib.Path(__file__).with_name("solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(results, name, rule, fn):
    try:
        fn()
    except Exception as exc:
        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
    else:
        results.append({"name": name, "rule": rule, "passed": True})


solution = load_solution()
results = []


def test_plain_subtotal():
    lines = [
        {"sku": "A", "qty": 2, "unit_cents": 500, "taxable": False},
        {"sku": "B", "qty": 1, "unit_cents": 100, "taxable": False},
    ]
    customer = {"tier": "regular", "tax_region": "NONE"}
    assert solution.calculate_cart_total(lines, customer, [], 100) == 1100


def test_empty_cart_zero():
    assert solution.calculate_cart_total([], {"tier": "regular", "tax_region": "NONE"}, [], 100) == 0


run_case(results, "plain subtotal", "visible_subtotal", test_plain_subtotal)
run_case(results, "empty cart", "visible_empty_cart", test_empty_cart_zero)

payload = {
    "passed": sum(1 for item in results if item["passed"]),
    "total": len(results),
    "cases": results,
}
print("MEMORY_CODING_TEST_RESULTS=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] == payload["total"] else 1)
'''


CART_HIDDEN_TESTS = r'''
import importlib.util
import json
import pathlib
import traceback


def load_solution():
    path = pathlib.Path(__file__).with_name("solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(results, name, rule, fn):
    try:
        fn()
    except Exception as exc:
        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
    else:
        results.append({"name": name, "rule": rule, "passed": True})


solution = load_solution()
results = []


def test_tier_discount_before_fixed_coupon_and_tax():
    lines = [{"sku": "A", "qty": 1, "unit_cents": 10000, "taxable": True}]
    customer = {"tier": "gold", "tax_region": "AU-WA"}
    coupons = [
        {"kind": "amount_cents", "value": 1000, "starts_at": 10, "ends_at": 20}
    ]
    assert solution.calculate_cart_total(lines, customer, coupons, 15) == 8800


def test_best_active_coupon_wins_and_expired_ignored():
    lines = [{"sku": "A", "qty": 1, "unit_cents": 10000, "taxable": False}]
    customer = {"tier": "regular", "tax_region": "NONE"}
    coupons = [
        {"kind": "amount_cents", "value": 3000, "starts_at": 1, "ends_at": 10},
        {"kind": "amount_cents", "value": 1500, "starts_at": 1, "ends_at": 30},
        {"kind": "percent_bps", "value": 2000, "starts_at": 1, "ends_at": 30},
    ]
    assert solution.calculate_cart_total(lines, customer, coupons, 20) == 8000


def test_coupon_end_is_exclusive():
    lines = [{"sku": "A", "qty": 1, "unit_cents": 10000, "taxable": False}]
    customer = {"tier": "regular", "tax_region": "NONE"}
    coupons = [
        {"kind": "amount_cents", "value": 1000, "starts_at": 1, "ends_at": 20}
    ]
    assert solution.calculate_cart_total(lines, customer, coupons, 20) == 10000


def test_no_tax_on_non_taxable_lines():
    lines = [{"sku": "A", "qty": 1, "unit_cents": 10000, "taxable": False}]
    customer = {"tier": "regular", "tax_region": "AU-WA"}
    assert solution.calculate_cart_total(lines, customer, [], 20) == 10000


def test_fixed_coupon_allocates_taxable_share_proportionally():
    lines = [
        {"sku": "A", "qty": 1, "unit_cents": 10000, "taxable": True},
        {"sku": "B", "qty": 1, "unit_cents": 10000, "taxable": False},
    ]
    customer = {"tier": "regular", "tax_region": "AU-WA"}
    coupons = [
        {"kind": "amount_cents", "value": 5000, "starts_at": 1, "ends_at": 30}
    ]
    assert solution.calculate_cart_total(lines, customer, coupons, 20) == 15750


run_case(
    results,
    "tier discount before coupon and tax",
    "cart_discount_order",
    test_tier_discount_before_fixed_coupon_and_tax,
)
run_case(
    results,
    "best active coupon",
    "cart_best_active_coupon",
    test_best_active_coupon_wins_and_expired_ignored,
)
run_case(results, "coupon end exclusive", "cart_expiry_exclusive", test_coupon_end_is_exclusive)
run_case(results, "no tax on non-taxable", "cart_taxable_scope", test_no_tax_on_non_taxable_lines)
run_case(
    results,
    "proportional taxable discount",
    "cart_taxable_allocation",
    test_fixed_coupon_allocates_taxable_share_proportionally,
)

payload = {
    "passed": sum(1 for item in results if item["passed"]),
    "total": len(results),
    "cases": results,
}
print("MEMORY_CODING_TEST_RESULTS=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] == payload["total"] else 1)
'''


ACCESS_VISIBLE_TESTS = r'''
import importlib.util
import json
import pathlib
import traceback


def load_solution():
    path = pathlib.Path(__file__).with_name("solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(results, name, rule, fn):
    try:
        fn()
    except Exception as exc:
        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
    else:
        results.append({"name": name, "rule": rule, "passed": True})


solution = load_solution()
results = []


def test_direct_allow():
    events = [
        {
            "kind": "policy",
            "principal_type": "user",
            "principal_id": "u1",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "allow",
            "ts": 10,
        }
    ]
    assert solution.can_access(events, "u1", "repo:alpha", "read", 12) is True


def test_no_rule_denies():
    assert solution.can_access([], "u1", "repo:alpha", "read", 12) is False


run_case(results, "direct allow", "visible_direct_allow", test_direct_allow)
run_case(results, "no rule denies", "visible_default_deny", test_no_rule_denies)

payload = {
    "passed": sum(1 for item in results if item["passed"]),
    "total": len(results),
    "cases": results,
}
print("MEMORY_CODING_TEST_RESULTS=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] == payload["total"] else 1)
'''


ACCESS_HIDDEN_TESTS = r'''
import importlib.util
import json
import pathlib
import traceback


def load_solution():
    path = pathlib.Path(__file__).with_name("solution.py")
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_case(results, name, rule, fn):
    try:
        fn()
    except Exception as exc:
        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
    else:
        results.append({"name": name, "rule": rule, "passed": True})


solution = load_solution()
results = []


def test_group_policy_applies_through_active_membership():
    events = [
        {"kind": "membership", "user_id": "u1", "group_id": "ops", "ts": 1},
        {
            "kind": "policy",
            "principal_type": "group",
            "principal_id": "ops",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "allow",
            "ts": 2,
        },
    ]
    assert solution.can_access(events, "u1", "repo:alpha", "read", 10) is True


def test_deleted_membership_removes_group_policy():
    events = [
        {"kind": "membership", "user_id": "u1", "group_id": "ops", "ts": 1},
        {"kind": "membership", "user_id": "u1", "group_id": "ops", "deleted": True, "ts": 5},
        {
            "kind": "policy",
            "principal_type": "group",
            "principal_id": "ops",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "allow",
            "ts": 2,
        },
    ]
    assert solution.can_access(events, "u1", "repo:alpha", "read", 10) is False


def test_user_deny_overrides_group_allow():
    events = [
        {"kind": "membership", "user_id": "u1", "group_id": "ops", "ts": 1},
        {
            "kind": "policy",
            "principal_type": "group",
            "principal_id": "ops",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "allow",
            "ts": 2,
        },
        {
            "kind": "policy",
            "principal_type": "user",
            "principal_id": "u1",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "deny",
            "ts": 3,
        },
    ]
    assert solution.can_access(events, "u1", "repo:alpha", "read", 10) is False


def test_expiry_is_exclusive():
    events = [
        {
            "kind": "policy",
            "principal_type": "user",
            "principal_id": "u1",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "allow",
            "ts": 1,
            "expires_at": 10,
        }
    ]
    assert solution.can_access(events, "u1", "repo:alpha", "read", 10) is False


def test_exact_resource_allow_beats_wildcard_deny():
    events = [
        {
            "kind": "policy",
            "principal_type": "user",
            "principal_id": "u1",
            "resource": "*",
            "action": "read",
            "effect": "deny",
            "ts": 9,
        },
        {
            "kind": "policy",
            "principal_type": "user",
            "principal_id": "u1",
            "resource": "repo:alpha",
            "action": "read",
            "effect": "allow",
            "ts": 1,
        },
    ]
    assert solution.can_access(events, "u1", "repo:alpha", "read", 10) is True


run_case(results, "group allow", "access_group_membership", test_group_policy_applies_through_active_membership)
run_case(results, "deleted membership", "access_membership_tombstone", test_deleted_membership_removes_group_policy)
run_case(results, "user deny wins", "access_principal_precedence", test_user_deny_overrides_group_allow)
run_case(results, "expiry exclusive", "access_expiry_exclusive", test_expiry_is_exclusive)
run_case(results, "exact beats wildcard", "access_specificity", test_exact_resource_allow_beats_wildcard_deny)

payload = {
    "passed": sum(1 for item in results if item["passed"]),
    "total": len(results),
    "cases": results,
}
print("MEMORY_CODING_TEST_RESULTS=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] == payload["total"] else 1)
'''


SETTINGS_NAIVE = '''
def resolve_user_setting(events, user_id, key):
    for event in events:
        if event.get("key") == key:
            return event.get("value")
    return None
'''


SETTINGS_MEMORY = '''
def resolve_user_setting(events, user_id, key):
    seen_ids = set()
    candidates = []

    for position, event in enumerate(events):
        if event.get("user_id") != user_id or event.get("key") != key:
            continue

        event_id = event.get("id", event.get("event_id"))
        if event_id is not None:
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)

        timestamp = event.get("ts", event.get("timestamp", 0))
        candidates.append((timestamp, position, event))

    if not candidates:
        return None

    _, _, winner = max(candidates, key=lambda item: (item[0], item[1]))
    if winner.get("deleted") is True:
        return None
    return winner.get("value")
'''


CART_NAIVE = '''
def calculate_cart_total(lines, customer, coupons, as_of):
    return sum(int(line.get("qty", 0)) * int(line.get("unit_cents", 0)) for line in lines)
'''


CART_MEMORY = '''
from decimal import Decimal, ROUND_HALF_UP


def calculate_cart_total(lines, customer, coupons, as_of):
    subtotal = Decimal(0)
    taxable_subtotal = Decimal(0)
    for line in lines:
        extended = Decimal(int(line.get("qty", 0))) * Decimal(int(line.get("unit_cents", 0)))
        subtotal += extended
        if line.get("taxable"):
            taxable_subtotal += extended

    if subtotal <= 0:
        return 0

    tier_rates = {
        "regular": Decimal("0"),
        "standard": Decimal("0"),
        "gold": Decimal("0.10"),
        "staff": Decimal("0.20"),
    }
    tier_rate = tier_rates.get(str(customer.get("tier", "regular")), Decimal("0"))
    post_tier_subtotal = subtotal * (Decimal(1) - tier_rate)
    post_tier_taxable = taxable_subtotal * (Decimal(1) - tier_rate)

    best_saving = Decimal(0)
    for coupon in coupons:
        starts_at = coupon.get("starts_at", float("-inf"))
        ends_at = coupon.get("ends_at", float("inf"))
        if not (starts_at <= as_of < ends_at):
            continue

        kind = coupon.get("kind")
        value = Decimal(int(coupon.get("value", 0)))
        if kind == "percent_bps":
            saving = post_tier_subtotal * value / Decimal(10000)
        elif kind == "amount_cents":
            saving = min(value, post_tier_subtotal)
        else:
            saving = Decimal(0)
        best_saving = max(best_saving, saving)

    total_before_tax = max(Decimal(0), post_tier_subtotal - best_saving)
    if post_tier_subtotal > 0:
        taxable_after_discount = post_tier_taxable * total_before_tax / post_tier_subtotal
    else:
        taxable_after_discount = Decimal(0)

    tax_bps = {
        "NONE": Decimal(0),
        "AU-WA": Decimal(1000),
        "US-CA": Decimal(725),
    }.get(str(customer.get("tax_region", "NONE")), Decimal(0))
    tax = taxable_after_discount * tax_bps / Decimal(10000)
    total = total_before_tax + tax
    return int(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
'''


ACCESS_NAIVE = '''
def can_access(events, user_id, resource, action, at_ts):
    for event in events:
        if event.get("kind") != "policy":
            continue
        if event.get("principal_type") != "user" or event.get("principal_id") != user_id:
            continue
        if event.get("resource") == resource and event.get("action") == action:
            return event.get("effect") == "allow"
    return False
'''


ACCESS_MEMORY = '''
def can_access(events, user_id, resource, action, at_ts):
    active_groups = set()
    latest_memberships = {}

    for event in events:
        if event.get("kind") != "membership":
            continue
        if event.get("user_id") != user_id:
            continue
        if event.get("ts", 0) > at_ts:
            continue
        group_id = event.get("group_id")
        current = latest_memberships.get(group_id)
        if current is None or event.get("ts", 0) >= current.get("ts", 0):
            latest_memberships[group_id] = event

    for group_id, event in latest_memberships.items():
        if not event.get("deleted", False):
            active_groups.add(group_id)

    applicable = []
    for event in events:
        if event.get("kind") != "policy":
            continue
        if event.get("ts", 0) > at_ts:
            continue
        expires_at = event.get("expires_at")
        if expires_at is not None and not (at_ts < expires_at):
            continue

        principal_type = event.get("principal_type")
        principal_id = event.get("principal_id")
        if principal_type == "user" and principal_id == user_id:
            principal_rank = 1
        elif principal_type == "group" and principal_id in active_groups:
            principal_rank = 0
        else:
            continue

        event_resource = event.get("resource")
        event_action = event.get("action")
        if event_resource not in {resource, "*"} or event_action not in {action, "*"}:
            continue

        specificity = int(event_resource == resource) + int(event_action == action)
        rank = (principal_rank, specificity, event.get("ts", 0))
        applicable.append((rank, event))

    if not applicable:
        return False

    best_rank = max(rank for rank, _ in applicable)
    winners = [event for rank, event in applicable if rank == best_rank]
    if any(event.get("effect") == "deny" for event in winners):
        return False
    return any(event.get("effect") == "allow" for event in winners)
'''


FIXTURES: tuple[CodingFixture, ...] = (
    CodingFixture(
        fixture_id="settings",
        title="Resolve user setting from event log",
        function_name="resolve_user_setting",
        prompt="""
        Implement resolve_user_setting(events, user_id, key) in solution.py.

        The input is a list of setting event dictionaries. Return the resolved
        setting value for the requested user_id and key, or None if the setting
        is unknown. Keep the implementation dependency-free.
        """,
        memory_lessons=(
            "Setting event resolution is scoped per user_id and key; never merge "
            "settings across users.",
            "For a user's setting, the event with the newest numeric ts/timestamp "
            "wins.",
            "A newest event with deleted=True is a tombstone and erases the value.",
            "Duplicate setting events share id or event_id; keep the first occurrence "
            "and ignore later duplicates.",
        ),
        visible_tests=SETTINGS_VISIBLE_TESTS,
        hidden_tests=SETTINGS_HIDDEN_TESTS,
        naive_solution=SETTINGS_NAIVE,
        memory_solution=SETTINGS_MEMORY,
    ),
    CodingFixture(
        fixture_id="cart",
        title="Calculate project cart total",
        function_name="calculate_cart_total",
        prompt="""
        Implement calculate_cart_total(lines, customer, coupons, as_of) in
        solution.py.

        Lines contain qty, unit_cents, and taxable fields. Customer records have
        tier and tax_region fields. Coupon records may be present. Return the
        final total as integer cents. Keep the implementation dependency-free
        except for Python's standard library.
        """,
        memory_lessons=(
            "Cart math uses integer cents and Decimal-style half-up rounding only "
            "for the final returned cents.",
            "Customer tier discounts apply before coupon discounts: gold is 10%, "
            "staff is 20%, regular/standard are 0%.",
            "Coupons are active only when starts_at <= as_of < ends_at; choose the "
            "single active coupon with the largest saving.",
            "Coupon kinds are percent_bps and amount_cents; fixed coupons are "
            "capped at the discounted subtotal.",
            "Tax is applied after discounts, only to taxable line value. Allocate "
            "cart-level discounts proportionally between taxable and non-taxable "
            "value.",
            "Tax rates are AU-WA=1000 bps, US-CA=725 bps, NONE=0 bps.",
        ),
        visible_tests=CART_VISIBLE_TESTS,
        hidden_tests=CART_HIDDEN_TESTS,
        naive_solution=CART_NAIVE,
        memory_solution=CART_MEMORY,
    ),
    CodingFixture(
        fixture_id="access",
        title="Resolve access policy events",
        function_name="can_access",
        prompt="""
        Implement can_access(events, user_id, resource, action, at_ts) in
        solution.py.

        Events are dictionaries for access policy and membership history. Return
        True when the requested action is allowed for the user at at_ts, otherwise
        False. Keep the implementation dependency-free.
        """,
        memory_lessons=(
            "Policy and membership events with ts greater than at_ts are ignored.",
            "Policy expires_at is exclusive: a policy applies only when at_ts is "
            "strictly less than expires_at.",
            "Group policies apply only through the user's latest non-deleted "
            "membership event for that group.",
            "Policy resource and action support '*' wildcards, but exact matches "
            "are more specific than wildcards.",
            "User-specific policies outrank group policies. Within the winning "
            "principal/specificity/timestamp rank, deny overrides allow.",
            "Default access is deny when no applicable policy remains.",
        ),
        visible_tests=ACCESS_VISIBLE_TESTS,
        hidden_tests=ACCESS_HIDDEN_TESTS,
        naive_solution=ACCESS_NAIVE,
        memory_solution=ACCESS_MEMORY,
    ),
)


NOISY_LESSONS = (
    "The dashboard accent palette is orchid for chart hover states.",
    "Espresso inventory reports sort notebook-tab labels alphabetically.",
    "Legacy CSV exports use semicolon separators for office seating plans.",
)


BASE_CONDITIONS: tuple[Condition, ...] = (
    Condition("A", "empty store / memory off", "empty", "off"),
    Condition("B", "helpful store / memory auto", "helpful", "auto"),
    Condition("C", "helpful+noise store / memory auto", "helpful_noisy", "auto"),
    Condition("D", "helpful store / memory off", "helpful", "off"),
)


PROMPT_FACTS_CONDITION = Condition(
    "E",
    "facts in prompt / memory off",
    "empty",
    "off",
    prompt_includes_facts=True,
)


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_block(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(fixture: CodingFixture, condition: Condition) -> str:
    prompt = normalize_block(fixture.prompt)
    if condition.prompt_includes_facts:
        prompt += "\nProject rules available for this control condition:\n"
        for lesson in fixture.memory_lessons:
            prompt += f"- {lesson}\n"
    return prompt


def render_stub(fixture: CodingFixture) -> str:
    args_by_function = {
        "resolve_user_setting": "events, user_id, key",
        "calculate_cart_total": "lines, customer, coupons, as_of",
        "can_access": "events, user_id, resource, action, at_ts",
    }
    args = args_by_function[fixture.function_name]
    return (
        f"def {fixture.function_name}({args}):\n"
        "    \"\"\"Implement this function.\"\"\"\n"
        "    raise NotImplementedError(\"TODO\")\n"
    )


def prepare_task_dir(root: Path, fixture: CodingFixture, condition: Condition) -> dict[str, Path]:
    task_dir = root / fixture.fixture_id / condition.code
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    target_file = task_dir / "solution.py"
    prompt_file = task_dir / "task_prompt.md"
    visible_file = task_dir / "visible_tests.py"
    hidden_file = task_dir / "hidden_tests.py"

    write_text(target_file, render_stub(fixture))
    write_text(prompt_file, build_prompt(fixture, condition))
    write_text(visible_file, normalize_block(fixture.visible_tests))
    write_text(hidden_file, normalize_block(fixture.hidden_tests))
    write_text(
        task_dir / "README.md",
        normalize_block(
            f"""
            # {fixture.title}

            Implement `{fixture.function_name}` in `solution.py`.
            The prompt intentionally omits the planted memory lessons.
            """
        ),
    )
    return {
        "task_dir": task_dir,
        "target_file": target_file,
        "prompt_file": prompt_file,
        "visible_file": visible_file,
        "hidden_file": hidden_file,
    }


def plant_memory_store(root: Path, fixture: CodingFixture, condition: Condition) -> Path:
    store_dir = root / fixture.fixture_id / condition.code / "memory_store"
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True)

    lessons: list[str] = []
    if condition.store_kind in {"helpful", "helpful_noisy"}:
        lessons.extend(fixture.memory_lessons)
    if condition.store_kind == "helpful_noisy":
        lessons.extend(NOISY_LESSONS)

    write_text(
        store_dir / "metadata.json",
        json.dumps(
            {
                "fixture_id": fixture.fixture_id,
                "condition": condition.code,
                "memory_profile": condition.memory_profile,
                "memory_mode": condition.memory_mode,
                "lesson_count": len(lessons),
                "store_kind": condition.store_kind,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(
        store_dir / "lessons.md",
        "\n".join(f"- {lesson}" for lesson in lessons) + ("\n" if lessons else ""),
    )
    with (store_dir / "memory_sessions.jsonl").open("w", encoding="utf-8") as handle:
        for index, lesson in enumerate(lessons, start=1):
            handle.write(
                json.dumps(
                    {
                        "session_id": f"{fixture.fixture_id}-lesson-{index}",
                        "kind": "helpful" if lesson in fixture.memory_lessons else "noise",
                        "text": lesson,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return store_dir


def parse_telemetry(output: str) -> dict[str, Any]:
    telemetry: dict[str, Any] = {}
    for line in output.splitlines():
        stripped = line.strip()
        for prefix in TELEMETRY_PREFIXES:
            if stripped.startswith(prefix):
                raw = stripped[len(prefix) :].strip()
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"raw": raw}
                if isinstance(parsed, dict):
                    telemetry.update(parsed)
    return telemetry


def extract_python_from_output(output: str, function_name: str) -> str | None:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
    for candidate in reversed(fenced):
        if f"def {function_name}" in candidate:
            return normalize_block(candidate)

    marker = f"def {function_name}"
    index = output.find(marker)
    if index == -1:
        return None
    return normalize_block(output[index:])


def solution_still_stub(target_file: Path) -> bool:
    text = target_file.read_text(encoding="utf-8")
    return "NotImplementedError" in text or "TODO" in text


def emit_mock_solution(
    fixture: CodingFixture,
    condition: Condition,
    store_dir: Path,
    target_file: Path,
) -> subprocess.CompletedProcess[str]:
    lessons_file = store_dir / "lessons.md"
    has_helpful_lessons = lessons_file.exists() and any(
        lesson in lessons_file.read_text(encoding="utf-8")
        for lesson in fixture.memory_lessons
    )
    memory_should_help = (
        condition.memory_mode != "off"
        and (has_helpful_lessons or condition.prompt_includes_facts)
    )
    solution = fixture.memory_solution if memory_should_help else fixture.naive_solution
    write_text(target_file, normalize_block(solution))
    telemetry = {
        "condition": condition.code,
        "fixture_id": fixture.fixture_id,
        "memory_used": bool(memory_should_help),
        "mock_agent": True,
    }
    stdout = "MEMORY_EVAL_TELEMETRY=" + json.dumps(telemetry, sort_keys=True) + "\n"
    return subprocess.CompletedProcess(args=["mock-agent"], returncode=0, stdout=stdout, stderr="")


def format_agent_command(
    template: str,
    *,
    fixture: CodingFixture,
    condition: Condition,
    task_dir: Path,
    target_file: Path,
    prompt_file: Path,
    store_dir: Path,
) -> str:
    values = {
        "task_dir": shlex.quote(str(task_dir)),
        "target_file": shlex.quote(str(target_file)),
        "prompt_file": shlex.quote(str(prompt_file)),
        "store_dir": shlex.quote(str(store_dir)),
        "memory_mode": shlex.quote(condition.memory_mode),
        "memory_profile": shlex.quote(condition.memory_profile),
        "condition": shlex.quote(condition.code),
        "fixture_id": shlex.quote(fixture.fixture_id),
        "function_name": shlex.quote(fixture.function_name),
    }
    return template.format(**values)


def run_agent(
    args: argparse.Namespace,
    fixture: CodingFixture,
    condition: Condition,
    task_dir: Path,
    target_file: Path,
    prompt_file: Path,
    store_dir: Path,
) -> tuple[subprocess.CompletedProcess[str], float]:
    env = os.environ.copy()
    env.update(
        {
            "LAZARUS_STORE_DIR": str(store_dir),
            "LAZARUS_MEMORY_PROFILE": condition.memory_profile,
            "LAZARUS_MEMORY_MODE": condition.memory_mode,
            "MEMORY_EVAL_TASK_DIR": str(task_dir),
            "MEMORY_EVAL_TARGET_FILE": str(target_file),
            "MEMORY_EVAL_PROMPT_FILE": str(prompt_file),
            "MEMORY_EVAL_CONDITION": condition.code,
            "MEMORY_EVAL_FIXTURE_ID": fixture.fixture_id,
        }
    )

    t0 = time.perf_counter()
    if args.agent_command is None:
        completed = emit_mock_solution(fixture, condition, store_dir, target_file)
    else:
        command = format_agent_command(
            args.agent_command,
            fixture=fixture,
            condition=condition,
            task_dir=task_dir,
            target_file=target_file,
            prompt_file=prompt_file,
            store_dir=store_dir,
        )
        completed = subprocess.run(
            command,
            shell=True,
            cwd=task_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout_s,
        )

        if solution_still_stub(target_file):
            extracted = extract_python_from_output(
                completed.stdout + "\n" + completed.stderr,
                fixture.function_name,
            )
            if extracted:
                write_text(target_file, extracted)
    return completed, time.perf_counter() - t0


def run_python_file(file_path: Path, cwd: Path, timeout_s: int) -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    return completed.returncode, completed.stdout, completed.stderr


def parse_test_results(output: str) -> dict[str, Any]:
    for line in output.splitlines():
        if line.startswith("MEMORY_CODING_TEST_RESULTS="):
            return json.loads(line.split("=", 1)[1])
    return {"passed": 0, "total": 0, "cases": [], "parse_error": True}


def compile_solution(target_file: Path, cwd: Path, timeout_s: int) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(target_file)],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    return completed.returncode == 0, completed.stderr + completed.stdout


def score_rules(hidden_payload: dict[str, Any]) -> tuple[int, int]:
    cases = hidden_payload.get("cases") or []
    rules: dict[str, list[bool]] = {}
    for case in cases:
        rule = str(case.get("rule") or case.get("name") or "unknown")
        rules.setdefault(rule, []).append(bool(case.get("passed")))
    if not rules:
        return 0, 0
    passed = sum(1 for outcomes in rules.values() if all(outcomes))
    return passed, len(rules)


def scan_leaks(source: str, fixture: CodingFixture, condition: Condition) -> tuple[bool, bool]:
    lower_source = source.lower()
    leaked_noise = any(term.lower() in lower_source for term in fixture.noise_terms)
    lesson_fragments = [lesson[:48].lower() for lesson in fixture.memory_lessons]
    leaked_lesson = any(fragment in lower_source for fragment in lesson_fragments)
    if condition.prompt_includes_facts:
        leaked_lesson = False
    return leaked_noise, leaked_lesson


def expected_memory_used(condition: Condition) -> bool:
    return condition.memory_mode != "off" and condition.store_kind != "empty"


def run_fixture_condition(
    args: argparse.Namespace,
    run_root: Path,
    fixture: CodingFixture,
    condition: Condition,
) -> TestRunResult:
    paths = prepare_task_dir(run_root / "tasks", fixture, condition)
    store_dir = plant_memory_store(run_root / "stores", fixture, condition)
    target_file = paths["target_file"]

    before_hash = sha256_text(target_file.read_text(encoding="utf-8"))
    notes: list[str] = []
    try:
        completed, latency_s = run_agent(
            args,
            fixture,
            condition,
            paths["task_dir"],
            target_file,
            paths["prompt_file"],
            store_dir,
        )
    except subprocess.TimeoutExpired as exc:
        notes.append(f"agent timed out after {args.timeout_s}s")
        completed = subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
        latency_s = float(args.timeout_s)
    except Exception as exc:  # noqa: BLE001 - report agent integration failures.
        notes.append(f"agent error: {type(exc).__name__}: {exc}")
        completed = subprocess.CompletedProcess(
            args=["agent"],
            returncode=125,
            stdout="",
            stderr=str(exc),
        )
        latency_s = 0.0

    after_text = target_file.read_text(encoding="utf-8")
    if sha256_text(after_text) == before_hash:
        notes.append("target file was unchanged")

    compile_ok, compile_output = compile_solution(target_file, paths["task_dir"], args.timeout_s)
    if not compile_ok:
        notes.append("compile failed: " + compile_output.strip().splitlines()[-1][:160])

    visible_payload = {"passed": 0, "total": 0, "cases": []}
    hidden_payload = {"passed": 0, "total": 0, "cases": []}
    if compile_ok:
        _, visible_stdout, visible_stderr = run_python_file(
            paths["visible_file"],
            paths["task_dir"],
            args.timeout_s,
        )
        visible_payload = parse_test_results(visible_stdout + visible_stderr)
        _, hidden_stdout, hidden_stderr = run_python_file(
            paths["hidden_file"],
            paths["task_dir"],
            args.timeout_s,
        )
        hidden_payload = parse_test_results(hidden_stdout + hidden_stderr)

    rules_passed, rules_total = score_rules(hidden_payload)
    telemetry = parse_telemetry(completed.stdout + "\n" + completed.stderr)
    memory_used = telemetry.get("memory_used")
    if isinstance(memory_used, str):
        memory_used = memory_used.strip().lower() == "true"
    elif memory_used is not None:
        memory_used = bool(memory_used)
    telemetry_ok: bool | None
    if memory_used is None:
        telemetry_ok = None
        notes.append("memory_used telemetry missing")
    else:
        telemetry_ok = memory_used == expected_memory_used(condition)

    leaked_noise, leaked_lesson = scan_leaks(after_text, fixture, condition)
    if leaked_noise:
        notes.append("solution leaked noisy memory text")
    if leaked_lesson:
        notes.append("solution copied planted lesson text")

    return TestRunResult(
        fixture_id=fixture.fixture_id,
        condition=condition.code,
        compile_ok=compile_ok,
        visible_passed=int(visible_payload.get("passed", 0)),
        visible_total=int(visible_payload.get("total", 0)),
        hidden_passed=int(hidden_payload.get("passed", 0)),
        hidden_total=int(hidden_payload.get("total", 0)),
        rules_passed=rules_passed,
        rules_total=rules_total,
        memory_used=memory_used,
        telemetry_ok=telemetry_ok,
        leaked_noise=leaked_noise,
        leaked_lesson=leaked_lesson,
        latency_s=latency_s,
        command_returncode=int(completed.returncode),
        task_dir=str(paths["task_dir"]),
        notes=notes,
    )


def as_report_dict(result: TestRunResult) -> dict[str, Any]:
    return {
        "fixture_id": result.fixture_id,
        "condition": result.condition,
        "compile_ok": result.compile_ok,
        "visible": f"{result.visible_passed}/{result.visible_total}",
        "hidden": f"{result.hidden_passed}/{result.hidden_total}",
        "rules": f"{result.rules_passed}/{result.rules_total}",
        "memory_used": result.memory_used,
        "telemetry_ok": result.telemetry_ok,
        "leaked_noise": result.leaked_noise,
        "leaked_lesson": result.leaked_lesson,
        "latency_s": round(result.latency_s, 3),
        "command_returncode": result.command_returncode,
        "task_dir": result.task_dir,
        "notes": result.notes,
    }


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def summarize_by_condition(results: list[TestRunResult]) -> list[dict[str, Any]]:
    condition_codes = sorted({result.condition for result in results})
    rows = []
    for code in condition_codes:
        subset = [result for result in results if result.condition == code]
        rows.append(
            {
                "condition": code,
                "compile": f"{sum(r.compile_ok for r in subset)}/{len(subset)}",
                "visible": (
                    f"{sum(r.visible_passed for r in subset)}/"
                    f"{sum(r.visible_total for r in subset)}"
                ),
                "hidden": (
                    f"{sum(r.hidden_passed for r in subset)}/"
                    f"{sum(r.hidden_total for r in subset)}"
                ),
                "rules": (
                    f"{sum(r.rules_passed for r in subset)}/"
                    f"{sum(r.rules_total for r in subset)}"
                ),
                "telemetry": (
                    f"{sum(r.telemetry_ok is True for r in subset)}/"
                    f"{sum(r.telemetry_ok is not None for r in subset)}"
                ),
                "avg_latency_s": round(
                    sum(r.latency_s for r in subset) / max(1, len(subset)),
                    3,
                ),
            }
        )
    return rows


def assert_memory_benefit(results: list[TestRunResult]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    by_fixture_condition = {
        (result.fixture_id, result.condition): result for result in results
    }
    for fixture in FIXTURES:
        baseline = by_fixture_condition.get((fixture.fixture_id, "A"))
        memory_on = by_fixture_condition.get((fixture.fixture_id, "B"))
        noisy_on = by_fixture_condition.get((fixture.fixture_id, "C"))
        store_off = by_fixture_condition.get((fixture.fixture_id, "D"))
        if baseline is None or memory_on is None or noisy_on is None or store_off is None:
            continue
        if memory_on.hidden_passed <= baseline.hidden_passed:
            failures.append(f"{fixture.fixture_id}: B did not beat A on hidden tests")
        if noisy_on.hidden_passed <= baseline.hidden_passed:
            failures.append(f"{fixture.fixture_id}: C did not beat A on hidden tests")
        if store_off.hidden_passed > baseline.hidden_passed:
            failures.append(f"{fixture.fixture_id}: D beat A, so store alone changed behavior")
    return not failures, failures


def selected_conditions(args: argparse.Namespace) -> tuple[Condition, ...]:
    conditions = list(BASE_CONDITIONS)
    if args.include_prompt_facts_control:
        conditions.append(PROMPT_FACTS_CONDITION)
    selected = {code.strip().upper() for code in args.conditions.split(",") if code.strip()}
    return tuple(condition for condition in conditions if condition.code in selected)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden-test coding fixtures across memory OFF/ON conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Agent command placeholders:
              {task_dir} {target_file} {prompt_file} {store_dir}
              {memory_mode} {memory_profile} {condition}
              {fixture_id} {function_name}

            The command also receives LAZARUS_STORE_DIR, LAZARUS_MEMORY_MODE,
            LAZARUS_MEMORY_PROFILE, MEMORY_EVAL_TARGET_FILE, and related env vars.
            Emit a line like:
              MEMORY_EVAL_TELEMETRY={"memory_used": true}
            to satisfy telemetry scoring.
            """
        ),
    )
    parser.add_argument(
        "--agent-command",
        default=None,
        help=(
            "Shell command template for the real coding harness. Defaults to a "
            "deterministic mock agent that demonstrates the eval."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for run artifacts (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--conditions",
        default="A,B,C,D",
        help="Comma-separated condition codes to run (default: A,B,C,D).",
    )
    parser.add_argument(
        "--include-prompt-facts-control",
        action="store_true",
        help="Add optional condition E: prompt includes the same facts, memory off.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=300,
        help="Per-agent and per-test timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep generated fixture directories after the run.",
    )
    parser.add_argument(
        "--no-assert-benefit",
        action="store_true",
        help="Do not fail when memory-on conditions fail to beat baseline.",
    )
    parser.add_argument(
        "--require-telemetry",
        action="store_true",
        help="Fail when memory_used telemetry is missing or wrong.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.output_root / f"run-{now_slug()}"
    run_root.mkdir(parents=True, exist_ok=True)

    conditions = selected_conditions(args)
    if not conditions:
        print("No conditions selected.", file=sys.stderr)
        return 2

    print(f"[memory-coding-eval] run_root={run_root}")
    print(
        "[memory-coding-eval] agent="
        + ("mock" if args.agent_command is None else args.agent_command)
    )
    print("[memory-coding-eval] fixtures=" + ",".join(f.fixture_id for f in FIXTURES))
    print("[memory-coding-eval] conditions=" + ",".join(c.code for c in conditions))

    results: list[TestRunResult] = []
    for fixture in FIXTURES:
        for condition in conditions:
            print(f"[run] fixture={fixture.fixture_id} condition={condition.code}", flush=True)
            results.append(run_fixture_condition(args, run_root, fixture, condition))

    report_rows = [as_report_dict(result) for result in results]
    report_path = run_root / "report.json"
    jsonl_path = run_root / "report.jsonl"
    write_text(
        report_path,
        json.dumps(
            {
                "run_root": str(run_root),
                "agent": "mock" if args.agent_command is None else args.agent_command,
                "conditions": [condition.__dict__ for condition in conditions],
                "results": report_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in report_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("\nPer-task matrix")
    print(
        table(
            [
                {
                    "fixture": row["fixture_id"],
                    "cond": row["condition"],
                    "compile": row["compile_ok"],
                    "visible": row["visible"],
                    "hidden": row["hidden"],
                    "rules": row["rules"],
                    "mem": row["memory_used"],
                    "telemetry": row["telemetry_ok"],
                    "leak": row["leaked_noise"] or row["leaked_lesson"],
                    "latency": row["latency_s"],
                }
                for row in report_rows
            ],
            [
                "fixture",
                "cond",
                "compile",
                "visible",
                "hidden",
                "rules",
                "mem",
                "telemetry",
                "leak",
                "latency",
            ],
        )
    )

    print("\nSummary by condition")
    print(
        table(
            summarize_by_condition(results),
            ["condition", "compile", "visible", "hidden", "rules", "telemetry", "avg_latency_s"],
        )
    )

    benefit_ok, benefit_failures = assert_memory_benefit(results)
    telemetry_failures = [
        f"{result.fixture_id}/{result.condition}"
        for result in results
        if result.telemetry_ok is not True
    ]
    leak_failures = [
        f"{result.fixture_id}/{result.condition}"
        for result in results
        if result.leaked_noise or result.leaked_lesson
    ]
    infra_failures = [
        f"{result.fixture_id}/{result.condition}"
        for result in results
        if not result.compile_ok or result.visible_passed != result.visible_total
    ]

    print(f"\nReport: {report_path}")
    if args.keep_workdir:
        print(f"Workdir kept: {run_root}")
    else:
        print(f"Workdir retained for report inspection: {run_root}")

    failed = False
    if infra_failures:
        failed = True
        print("MEMORY_CODING_EVAL_FAIL infra=" + ",".join(infra_failures))
    if not args.no_assert_benefit and not benefit_ok:
        failed = True
        print("MEMORY_CODING_EVAL_FAIL benefit=" + "; ".join(benefit_failures))
    if args.require_telemetry and telemetry_failures:
        failed = True
        print("MEMORY_CODING_EVAL_FAIL telemetry=" + ",".join(telemetry_failures))
    if leak_failures:
        failed = True
        print("MEMORY_CODING_EVAL_FAIL leaks=" + ",".join(leak_failures))

    if failed:
        return 1
    print("MEMORY_CODING_EVAL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
