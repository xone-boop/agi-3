"""Deterministic execution and world-model ledger for ARC game play.

This module deliberately separates engine facts from learned semantics:

* action identifiers and parameter arity are interface contracts;
* documented labels are priors, never facts;
* action meaning is learned at game scope by default;
* level availability, state preconditions, and target affordances are overlays;
* an unexecuted request is never represented as an executed no-op.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Iterable

from inference.agent.action_names import (
    CANONICAL_ACTIONS,
    INTERFACE_PRIORS,
    to_engine_action,
)


LEDGER_SCHEMA_VERSION = 1
OUTCOME_TYPES = {
    "not_advertised",
    "adapter_unmapped",
    "parameter_invalid",
    "engine_rejected",
    "govern_suppressed",
    "executed_no_observable_change",
    "executed_observable_change",
    "executed_task_progress",
    "executed_terminal_failure",
}
VERIFICATION_STATUSES = {
    "not_tested",
    "inconclusive",
    "supported",
    "partially_supported",
    "refuted",
}


def _parameter_schema(action_id: str) -> dict[str, Any]:
    if action_id == "ACTION6":
        return {
            "type": "object",
            "required": ["row", "col"],
            "properties": {
                "row": {"type": "integer", "minimum": 0, "maximum": 63},
                "col": {"type": "integer", "minimum": 0, "maximum": 63},
            },
            "coordinate_transform": {"engine_x": "col", "engine_y": "row"},
        }
    return {"type": "object", "required": [], "properties": {}}


def _action_contract(action_id: str) -> dict[str, Any]:
    return {
        "engine_action_id": action_id,
        "parameter_schema": _parameter_schema(action_id),
        "interface_prior": copy.deepcopy(INTERFACE_PRIORS[action_id]),
        "semantic_scope": "game",
        "semantic_status": "undefined" if action_id != "RESET" else "interface_contract",
        "semantic_candidates": [],
        "availability_by_level": {},
        "outcome_counts": {},
    }


def new_ledger(*, game_id: str = "") -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "objective": {
            "type": "complete_game",
            "success_signal": {"field": "engine_state", "equals": "WIN"},
            "local_completion_condition": {
                "status": "undefined",
                "candidates": [],
            },
        },
        "scope": {
            "game_id": str(game_id or ""),
            "learned_action_semantics_default": "game",
            "level_semantics_policy": "inherit_game_unless_contradicted",
            "cross_game_transfer": "prior_only",
        },
        "interface": {
            "actions": {
                action_id: _action_contract(action_id)
                for action_id in CANONICAL_ACTIONS
            }
        },
        "execution": {"attempts": [], "transitions": [], "sequences": []},
        "world_model": {
            "hypotheses": [],
            "object_roles": [],
            "action_functions": [],
            "affordances": [],
            "preconditions": [],
            "completion_conditions": [],
        },
        "verification": {"records": []},
        "coverage": {
            "unknowns": [
                {
                    "key": "local_completion_condition",
                    "scope": "game",
                    "status": "unmeasured",
                },
                *[
                    {
                        "key": f"action_function:{action_id}",
                        "scope": "game",
                        "status": "unmeasured",
                    }
                    for action_id in CANONICAL_ACTIONS
                    if action_id != "RESET"
                ],
            ]
        },
        "governance": {"events": []},
    }


def ensure_ledger(value: Any, *, game_id: str = "") -> dict[str, Any]:
    """Upgrade a partial ledger without discarding already recorded evidence."""

    if not isinstance(value, dict):
        return new_ledger(game_id=game_id)
    # Runtime sessions hold this mapping by reference.  Upgrade it in place so
    # observations recorded by helpers remain visible to the solver owner.
    ledger = value
    template = new_ledger(game_id=game_id)
    for key, default in template.items():
        if key not in ledger or not isinstance(ledger[key], type(default)):
            ledger[key] = copy.deepcopy(default)
    if game_id and not str(ledger["scope"].get("game_id") or "").strip():
        ledger["scope"]["game_id"] = str(game_id)
    actions = ledger["interface"].setdefault("actions", {})
    for action_id in CANONICAL_ACTIONS:
        existing = actions.setdefault(action_id, _action_contract(action_id))
        contract = _action_contract(action_id)
        for key, default in contract.items():
            existing.setdefault(key, copy.deepcopy(default))
    for section, fields in {
        "execution": ("attempts", "transitions", "sequences"),
        "world_model": (
            "hypotheses",
            "object_roles",
            "action_functions",
            "affordances",
            "preconditions",
            "completion_conditions",
        ),
        "verification": ("records",),
        "coverage": ("unknowns",),
        "governance": ("events",),
    }.items():
        target = ledger.setdefault(section, {})
        for field in fields:
            if not isinstance(target.get(field), list):
                target[field] = []
    ledger["schema_version"] = LEDGER_SCHEMA_VERSION
    return ledger


def _bounded_append(items: list[Any], item: Any, *, limit: int) -> None:
    items.append(item)
    if len(items) > limit:
        del items[:-limit]


def _next_id(items: Iterable[dict[str, Any]], prefix: str) -> str:
    maximum = 0
    for item in items:
        raw = str(item.get("id", ""))
        if raw.startswith(prefix):
            try:
                maximum = max(maximum, int(raw[len(prefix) :]))
            except ValueError:
                pass
    return f"{prefix}{maximum + 1}"


def observe_action_availability(
    ledger: dict[str, Any],
    *,
    level: int,
    advertised_actions: Iterable[str],
    step: int,
) -> dict[str, Any]:
    ledger = ensure_ledger(ledger)
    advertised = {
        action_id
        for value in advertised_actions
        for action_id in [to_engine_action(value)]
        if action_id is not None
    }
    actions = ledger["interface"]["actions"]
    level_key = str(int(level))
    for action_id, contract in actions.items():
        availability = contract["availability_by_level"].setdefault(
            level_key,
            {
                "status": "unknown",
                "advertised_count": 0,
                "not_advertised_count": 0,
                "first_observed_step": int(step),
                "last_observed_step": int(step),
            },
        )
        is_advertised = action_id in advertised
        availability["status"] = "advertised" if is_advertised else "not_advertised"
        availability["currently_advertised"] = is_advertised
        availability["last_observed_step"] = int(step)
        counter = "advertised_count" if is_advertised else "not_advertised_count"
        availability[counter] = int(availability.get(counter, 0) or 0) + 1
    return ledger


def classify_executed_outcome(observation: dict[str, Any]) -> str:
    if bool(observation.get("game_over")):
        return "executed_terminal_failure"
    if bool(
        observation.get("progress_changed")
        or observation.get("level_completed")
        or observation.get("run_complete")
        or float(observation.get("reward_delta") or 0.0) != 0.0
    ):
        return "executed_task_progress"
    if bool(observation.get("state_changed") or observation.get("visual_changed")):
        return "executed_observable_change"
    return "executed_no_observable_change"


def _information_gain(
    attempts: list[dict[str, Any]],
    *,
    action_id: str | None,
    action_data: dict[str, Any],
    level: int,
    observation_hash_before: str | None,
    outcome: str,
) -> dict[str, Any]:
    if outcome == "executed_task_progress":
        return {"score": 1.0, "class": "task_progress", "novel": True}
    context = (
        action_id,
        json.dumps(action_data, sort_keys=True, separators=(",", ":")),
        int(level),
        observation_hash_before,
    )
    matches = []
    for attempt in attempts:
        previous_context = (
            attempt.get("action_id"),
            json.dumps(attempt.get("action_data") or {}, sort_keys=True, separators=(",", ":")),
            int(attempt.get("level") or 0),
            attempt.get("observation_hash_before"),
        )
        if previous_context == context:
            matches.append(attempt)
    if not matches:
        return {"score": 0.7, "class": "new_context_probe", "novel": True}
    if all(item.get("outcome") != outcome for item in matches):
        return {"score": 0.6, "class": "new_outcome_for_context", "novel": True}
    if outcome == "executed_no_observable_change":
        return {"score": 0.05, "class": "repeated_uninformative_outcome", "novel": False}
    return {"score": 0.2, "class": "repeated_context_outcome", "novel": False}


def _attempt_observation_summary(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None
    return {
        key: copy.deepcopy(observation.get(key))
        for key in (
            "observation_hash_before",
            "observation_hash_after",
            "state_changed",
            "visual_changed",
            "gameplay_changed",
            "hud_changed",
            "progress_changed",
            "reward_delta",
            "level_completed",
            "game_over",
            "run_complete",
            "changed_cell_count",
            "outcome",
        )
        if key in observation
    }


def record_action_attempt(
    ledger: dict[str, Any],
    *,
    requested_action: str | None,
    action_id: str | None,
    action_data: dict[str, Any] | None,
    level: int,
    step: int,
    observation_hash_before: str | None,
    advertised_actions: Iterable[str],
    outcome: str,
    executed: bool,
    error: str | None = None,
    observation: dict[str, Any] | None = None,
    sequence_id: str | None = None,
    batch_index: int | None = None,
) -> dict[str, Any]:
    ledger = observe_action_availability(
        ensure_ledger(ledger),
        level=level,
        advertised_actions=advertised_actions,
        step=step,
    )
    if outcome not in OUTCOME_TYPES:
        raise ValueError(f"Unknown ledger outcome: {outcome}")
    canonical_id = to_engine_action(action_id or requested_action)
    data = dict(action_data or {})
    target_binding = None
    if canonical_id == "ACTION6":
        target_binding = {
            "type": "coordinate",
            "row": data.get("row"),
            "col": data.get("col"),
            "status": "coordinate_bound_object_identity_undefined",
        }
    attempts = ledger["execution"]["attempts"]
    attempt = {
        "id": _next_id(attempts, "A"),
        "requested_action": str(requested_action or ""),
        "action_id": canonical_id,
        "action_data": data,
        "target_binding": target_binding,
        "level": int(level),
        "step": int(step),
        "observation_hash_before": observation_hash_before,
        "advertised_before": bool(canonical_id and canonical_id in {
            to_engine_action(item) for item in advertised_actions
        }),
        "executed": bool(executed),
        "outcome": outcome,
        "error": str(error) if error else None,
        "sequence_id": sequence_id,
        "batch_index": batch_index,
        "observation": _attempt_observation_summary(observation) if executed else None,
    }
    attempt["information_gain"] = _information_gain(
        attempts,
        action_id=canonical_id,
        action_data=data,
        level=level,
        observation_hash_before=observation_hash_before,
        outcome=outcome,
    )
    _bounded_append(attempts, attempt, limit=256)
    retained_attempt_ids = {str(item.get("id") or "") for item in attempts}
    ledger["execution"]["transitions"] = [
        item
        for item in ledger["execution"]["transitions"]
        if str(item.get("attempt_id") or "") in retained_attempt_ids
    ]

    if canonical_id in ledger["interface"]["actions"]:
        contract = ledger["interface"]["actions"][canonical_id]
        counts = contract["outcome_counts"]
        counts[outcome] = int(counts.get(outcome, 0) or 0) + 1

    if executed and isinstance(observation, dict):
        transition = {
            "attempt_id": attempt["id"],
            "sequence_id": sequence_id,
            "batch_index": batch_index,
            "action_id": canonical_id,
            **copy.deepcopy(observation),
            "outcome": outcome,
            "information_gain": copy.deepcopy(attempt["information_gain"]),
        }
        _bounded_append(ledger["execution"]["transitions"], transition, limit=128)

    if (
        outcome == "executed_no_observable_change"
        and attempt["information_gain"]["novel"] is False
    ):
        _bounded_append(
            ledger["governance"]["events"],
            {
                "id": _next_id(ledger["governance"]["events"], "G"),
                "decision": "allow_with_structured_risk",
                "risk": "same_action_same_observation_repeated_no_change",
                "attempt_id": attempt["id"],
                "reason": (
                    "A repeat can be necessary under hidden state or sequence dependence; "
                    "the controller records risk but does not silently block it."
                ),
            },
            limit=64,
        )
    return attempt


def record_sequence(
    ledger: dict[str, Any],
    *,
    sequence_id: str,
    requested_actions: list[str],
    attempt_ids: list[str],
    stopped_early: bool,
    stop_reason: str | None,
) -> dict[str, Any]:
    ledger = ensure_ledger(ledger)
    item = {
        "id": sequence_id,
        "requested_actions": list(requested_actions),
        "attempt_ids": list(attempt_ids),
        "requested_count": len(requested_actions),
        "attempted_count": len(attempt_ids),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
    }
    _bounded_append(ledger["execution"]["sequences"], item, limit=64)
    return item


def next_sequence_id(ledger: dict[str, Any]) -> str:
    return _next_id(ensure_ledger(ledger)["execution"]["sequences"], "S")


def _normalized_action_sequence(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        raw = item.get("action") if isinstance(item, dict) else item
        action_id = to_engine_action(raw)
        if action_id:
            result.append(action_id)
    return result


def _normalize_predictions(expectation: dict[str, Any]) -> list[dict[str, Any]]:
    raw = expectation.get("predictions", expectation.get("predicted_observations", []))
    predictions: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and str(item.get("field") or "").strip():
                predictions.append(copy.deepcopy(item))
            elif isinstance(item, str) and item.strip():
                predictions.append({"description": item.strip(), "evaluable": False})
    gameplay_change = expectation.get("gameplay_change")
    if isinstance(gameplay_change, bool):
        predictions.append(
            {"field": "gameplay_changed", "op": "eq", "value": gameplay_change}
        )
    return predictions


def _world_model_bucket(kind: str) -> str | None:
    return {
        "object_role": "object_roles",
        "action_function": "action_functions",
        "affordance": "affordances",
        "object_action_affordance": "affordances",
        "precondition": "preconditions",
        "completion_condition": "completion_conditions",
    }.get(kind)


def _upsert_world_model_candidate(
    ledger: dict[str, Any], hypothesis: dict[str, Any]
) -> None:
    bucket_name = _world_model_bucket(str(hypothesis.get("kind") or ""))
    if bucket_name is None:
        return
    bucket = ledger["world_model"][bucket_name]
    candidate = {
        key: copy.deepcopy(hypothesis.get(key))
        for key in (
            "id",
            "kind",
            "claim",
            "action_id",
            "scope",
            "origin",
            "target_binding",
            "preconditions",
            "confidence_before",
            "confidence_after",
            "status",
            "verification_status",
            "verification_reason",
        )
    }
    bucket[:] = [item for item in bucket if item.get("id") != hypothesis.get("id")]
    _bounded_append(bucket, candidate, limit=64)
    if bucket_name == "completion_conditions":
        local_condition = ledger["objective"]["local_completion_condition"]
        candidates = local_condition.setdefault("candidates", [])
        candidates[:] = [
            item for item in candidates if item.get("id") != hypothesis.get("id")
        ]
        _bounded_append(candidates, copy.deepcopy(candidate), limit=16)
        if hypothesis.get("verification_status") in {"supported", "partially_supported"}:
            local_condition["status"] = "candidate_supported"
        elif candidates:
            local_condition["status"] = "candidate_only"


def _set_coverage_status(
    ledger: dict[str, Any], *, key: str, scope: str, status: str, evidence_id: str
) -> None:
    unknowns = ledger["coverage"]["unknowns"]
    item = next((entry for entry in unknowns if entry.get("key") == key), None)
    if item is None:
        item = {"key": key, "scope": scope}
        unknowns.append(item)
    item["status"] = status
    item["evidence_id"] = evidence_id


def record_hypothesis(
    ledger: dict[str, Any],
    expectation: dict[str, Any],
    *,
    current_step: int | None = None,
    current_level: int | None = None,
) -> dict[str, Any]:
    ledger = ensure_ledger(ledger)
    hypotheses = ledger["world_model"]["hypotheses"]
    action_id = to_engine_action(expectation.get("action"))
    sequence = _normalized_action_sequence(
        expectation.get("action_sequence", expectation.get("actions"))
    )
    if action_id is None and len(sequence) == 1:
        action_id = sequence[0]
    raw_scope = expectation.get("scope")
    scope = copy.deepcopy(raw_scope) if isinstance(raw_scope, dict) else {"type": str(raw_scope or "game")}
    scope.setdefault("type", "game")
    if scope.get("type") == "level" and scope.get("level") is None:
        scope["level"] = expectation.get("level", current_level)
    target = expectation.get("target_binding", expectation.get("target"))
    if isinstance(target, dict):
        target_binding = copy.deepcopy(target)
    elif target not in (None, ""):
        target_binding = {"description": str(target), "status": "unbound"}
    else:
        target_binding = {"status": "undefined"}
    target_binding.setdefault("status", "candidate")
    claim = expectation.get("claim", expectation.get("effect", ""))
    item = {
        "id": _next_id(hypotheses, "H"),
        "kind": str(expectation.get("kind") or "action_effect_prediction"),
        "claim": copy.deepcopy(claim),
        "action_id": action_id,
        "action_sequence": sequence,
        "scope": scope,
        "origin": {
            "step": expectation.get("step", current_step),
            "level": expectation.get("level", current_level),
        },
        "target_binding": target_binding,
        "preconditions": copy.deepcopy(expectation.get("preconditions") or []),
        "predictions": _normalize_predictions(expectation),
        "confidence_before": expectation.get("confidence"),
        "confidence_after": None,
        "status": "open",
        "verification_status": "not_tested",
        "verification_reason": "No matching executed transition has been observed.",
        "test_progress": [],
        "test_observations": [],
        "test_sequence_id": None,
    }
    _bounded_append(hypotheses, item, limit=64)
    _upsert_world_model_candidate(ledger, item)
    if action_id:
        _set_coverage_status(
            ledger,
            key=f"action_function:{action_id}",
            scope="game",
            status="hypothesis_recorded",
            evidence_id=item["id"],
        )
    if target_binding.get("status") in {"undefined", "unbound", "unknown"}:
        _set_coverage_status(
            ledger,
            key=f"target_binding:{item['id']}",
            scope=str(scope.get("type") or "game"),
            status="unmeasured",
            evidence_id=item["id"],
        )
    return item


def _resolve_field(value: Any, dotted_field: str) -> tuple[bool, Any]:
    current = value
    for part in dotted_field.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _evaluate_predicate(
    predicate: dict[str, Any], observation: dict[str, Any]
) -> tuple[str, str]:
    field = str(predicate.get("field") or "").strip()
    if not field or predicate.get("evaluable") is False:
        return "unknown", "Prediction is descriptive but has no observable field predicate."
    found, actual = _resolve_field(observation, field)
    if not found:
        return "unknown", f"Observation does not measure {field}."
    op = str(predicate.get("op") or "eq").lower()
    expected = predicate.get("value")
    try:
        if op == "eq":
            matched = actual == expected
        elif op == "ne":
            matched = actual != expected
        elif op == "truthy":
            matched = bool(actual)
        elif op == "falsy":
            matched = not bool(actual)
        elif op == "nonempty":
            matched = bool(actual)
        elif op == "empty":
            matched = not bool(actual)
        elif op == "gt":
            matched = actual > expected
        elif op == "gte":
            matched = actual >= expected
        elif op == "lt":
            matched = actual < expected
        elif op == "lte":
            matched = actual <= expected
        elif op == "contains":
            matched = expected in actual
        else:
            return "unknown", f"Unsupported predicate operator {op}."
    except (TypeError, ValueError):
        return "unknown", f"Predicate {field} {op} could not be evaluated."
    return ("supported" if matched else "refuted"), f"{field} {op} {expected!r}; observed {actual!r}."


def _verification_observation(
    hypothesis: dict[str, Any], observation: dict[str, Any]
) -> dict[str, Any]:
    result = observation
    observations = hypothesis.get("test_observations")
    if isinstance(observations, list) and observations:
        result = dict(observations[-1])
        result["sequence_transitions"] = copy.deepcopy(observations)
        result["sequence_any_state_changed"] = any(
            bool(item.get("state_changed")) for item in observations if isinstance(item, dict)
        )
        result["sequence_any_progress_changed"] = any(
            bool(item.get("progress_changed")) for item in observations if isinstance(item, dict)
        )
    return result


def _store_semantic_evidence(ledger: dict[str, Any], hypothesis: dict[str, Any]) -> None:
    if hypothesis.get("kind") != "action_function":
        return
    action_id = hypothesis.get("action_id")
    if action_id not in ledger["interface"]["actions"]:
        return
    candidate = {
        "hypothesis_id": hypothesis["id"],
        "claim": copy.deepcopy(hypothesis.get("claim")),
        "scope": copy.deepcopy(hypothesis.get("scope")),
        "status": hypothesis.get("verification_status"),
        "confidence": hypothesis.get("confidence_after"),
    }
    contract = ledger["interface"]["actions"][action_id]
    existing = [
        item for item in contract["semantic_candidates"]
        if item.get("hypothesis_id") != hypothesis["id"]
    ]
    existing.append(candidate)
    contract["semantic_candidates"] = existing[-16:]
    if hypothesis.get("verification_status") in {"supported", "partially_supported"}:
        contract["semantic_status"] = "candidate_supported"


def _close_inconclusive_hypothesis(
    ledger: dict[str, Any], hypothesis: dict[str, Any], *, reason: str
) -> None:
    hypothesis["status"] = "closed"
    hypothesis["verification_status"] = "inconclusive"
    hypothesis["verification_reason"] = reason
    hypothesis["confidence_after"] = hypothesis.get("confidence_before")
    _bounded_append(
        ledger["verification"]["records"],
        {
            "id": _next_id(ledger["verification"]["records"], "V"),
            "hypothesis_id": hypothesis["id"],
            "status": "inconclusive",
            "reason": reason,
            "evaluations": [],
            "actual": {
                "observed_action_sequence": list(hypothesis.get("test_progress") or []),
                "sequence_id": hypothesis.get("test_sequence_id"),
            },
        },
        limit=128,
    )
    _upsert_world_model_candidate(ledger, hypothesis)


def _finalize_hypothesis_verification(
    ledger: dict[str, Any], hypothesis: dict[str, Any], observation: dict[str, Any]
) -> None:
    evaluations: list[dict[str, Any]] = []
    verification_observation = _verification_observation(hypothesis, observation)
    for predicate in hypothesis.get("predictions") or []:
        if not isinstance(predicate, dict):
            continue
        at_index = predicate.get("at_index")
        predicate_observation = verification_observation
        if at_index is not None and isinstance(hypothesis.get("test_observations"), list):
            try:
                predicate_observation = hypothesis["test_observations"][int(at_index)]
            except (IndexError, TypeError, ValueError):
                evaluations.append(
                    {"predicate": copy.deepcopy(predicate), "result": "unknown", "detail": "Requested sequence index was not observed."}
                )
                continue
        target_status = str((hypothesis.get("target_binding") or {}).get("status") or "undefined")
        preconditions = hypothesis.get("preconditions") or []
        unverified_preconditions = any(
            not isinstance(item, dict)
            or str(item.get("status") or "unknown") not in {"verified", "satisfied"}
            for item in preconditions
        )
        if predicate.get("target_required") and target_status not in {"bound", "verified"}:
            result, detail = "unknown", "The prediction requires a target, but target binding is not verified."
        elif predicate.get("preconditions_required") and unverified_preconditions:
            result, detail = "unknown", "The prediction requires preconditions that are not verified."
        else:
            result, detail = _evaluate_predicate(predicate, predicate_observation)
        evaluations.append(
            {"predicate": copy.deepcopy(predicate), "result": result, "detail": detail}
        )
    known = [item["result"] for item in evaluations if item["result"] != "unknown"]
    target_status = str((hypothesis.get("target_binding") or {}).get("status") or "undefined")
    preconditions = hypothesis.get("preconditions") or []
    unverified_preconditions = any(
        not isinstance(item, dict)
        or str(item.get("status") or "unknown") not in {"verified", "satisfied"}
        for item in preconditions
    )
    no_change_is_ambiguous = bool(
        verification_observation.get("outcome") == "executed_no_observable_change"
        and (
            target_status in {"unbound", "unknown"}
            or hypothesis.get("action_id") == "ACTION6" and target_status == "undefined"
            or unverified_preconditions
        )
    )
    if not evaluations or not known:
        status = "inconclusive"
        reason = "The claim has no fully evaluable observable prediction."
    elif no_change_is_ambiguous and not any(result == "supported" for result in known):
        status = "inconclusive"
        reason = "No observable change cannot distinguish a false claim from an unbound target or unmet precondition."
    elif all(result == "supported" for result in known) and len(known) == len(evaluations):
        status = "supported"
        reason = "All observable predictions matched."
    elif any(result == "supported" for result in known):
        status = "partially_supported"
        reason = "Only part of the observable prediction matched."
    else:
        status = "refuted"
        reason = "The matching executed transition contradicted the observable prediction."
    hypothesis["status"] = "closed"
    hypothesis["verification_status"] = status
    hypothesis["verification_reason"] = reason
    before = hypothesis.get("confidence_before")
    try:
        base = float(before) if before is not None else 0.5
    except (TypeError, ValueError):
        base = 0.5
    if status == "supported":
        hypothesis["confidence_after"] = max(base, 0.75)
    elif status == "partially_supported":
        hypothesis["confidence_after"] = min(max(base, 0.45), 0.7)
    elif status == "refuted":
        hypothesis["confidence_after"] = min(base, 0.2)
    else:
        hypothesis["confidence_after"] = base
    record = {
        "id": _next_id(ledger["verification"]["records"], "V"),
        "hypothesis_id": hypothesis["id"],
        "status": status,
        "reason": reason,
        "evaluations": evaluations,
        "actual": {
            key: copy.deepcopy(verification_observation.get(key))
            for key in (
                "action_id",
                "outcome",
                "state_changed",
                "gameplay_changed",
                "hud_changed",
                "progress_changed",
                "reward_delta",
                "object_changes",
                "observation_hash_before",
                "observation_hash_after",
            )
            if key in verification_observation
        },
    }
    _bounded_append(ledger["verification"]["records"], record, limit=128)
    _store_semantic_evidence(ledger, hypothesis)
    _upsert_world_model_candidate(ledger, hypothesis)
    action_id = hypothesis.get("action_id")
    if action_id:
        coverage_status = (
            "candidate_supported"
            if status in {"supported", "partially_supported"}
            else "still_unknown"
        )
        _set_coverage_status(
            ledger,
            key=f"action_function:{action_id}",
            scope="game",
            status=coverage_status,
            evidence_id=hypothesis["id"],
        )


def verify_open_hypotheses(
    ledger: dict[str, Any],
    observation: dict[str, Any],
    *,
    sequence_id: str | None = None,
) -> dict[str, Any]:
    ledger = ensure_ledger(ledger)
    observed_action = to_engine_action(
        observation.get("action_id", observation.get("action_type", observation.get("action")))
    )
    observed_level = observation.get("level_before", observation.get("level_after"))
    for hypothesis in ledger["world_model"]["hypotheses"]:
        if hypothesis.get("status") != "open":
            continue
        scope = hypothesis.get("scope") or {}
        if scope.get("type") == "level" and scope.get("level") not in (None, observed_level):
            continue
        expected_sequence = list(hypothesis.get("action_sequence") or [])
        expected_action = hypothesis.get("action_id")
        if expected_sequence:
            progress = list(hypothesis.get("test_progress") or [])
            active_sequence_id = hypothesis.get("test_sequence_id")
            if active_sequence_id not in (None, sequence_id):
                progress = []
                hypothesis["test_observations"] = []
            if not progress and observed_action != expected_sequence[0]:
                continue
            next_index = len(progress)
            if next_index >= len(expected_sequence) or observed_action != expected_sequence[next_index]:
                _close_inconclusive_hypothesis(
                    ledger,
                    hypothesis,
                    reason="The tested action sequence diverged before completion.",
                )
                continue
            hypothesis["test_sequence_id"] = sequence_id
            progress.append(observed_action)
            hypothesis["test_progress"] = progress
            hypothesis.setdefault("test_observations", []).append(copy.deepcopy(observation))
            if len(progress) < len(expected_sequence):
                continue
        elif expected_action and observed_action != expected_action:
            continue
        _finalize_hypothesis_verification(ledger, hypothesis, observation)
    return ledger


def finalize_incomplete_sequence_hypotheses(
    ledger: dict[str, Any], *, sequence_id: str, stopped_early: bool
) -> dict[str, Any]:
    ledger = ensure_ledger(ledger)
    if not stopped_early:
        return ledger
    for hypothesis in ledger["world_model"]["hypotheses"]:
        if (
            hypothesis.get("status") == "open"
            and hypothesis.get("test_sequence_id") == sequence_id
            and hypothesis.get("test_progress")
        ):
            _close_inconclusive_hypothesis(
                ledger,
                hypothesis,
                reason="The environment stopped the tested sequence before completion.",
            )
    return ledger


def sync_legacy_hypotheses(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(ensure_ledger(ledger)["world_model"]["hypotheses"])


def compact_inform_view(
    ledger: dict[str, Any], *, current_level: int, valid_actions: Iterable[str]
) -> dict[str, Any]:
    ledger = ensure_ledger(ledger)
    advertised = {
        action_id
        for value in valid_actions
        for action_id in [to_engine_action(value)]
        if action_id is not None
    }
    interface_view = []
    for action_id, contract in ledger["interface"]["actions"].items():
        if action_id == "RESET" or action_id in advertised or contract["semantic_candidates"]:
            interface_view.append(
                {
                    "action_id": action_id,
                    "advertised_now": action_id in advertised,
                    "parameter_schema": copy.deepcopy(contract["parameter_schema"]),
                    "interface_prior": copy.deepcopy(contract["interface_prior"]),
                    "semantic_status": contract["semantic_status"],
                    "semantic_candidates": copy.deepcopy(contract["semantic_candidates"][-4:]),
                }
            )
    hypotheses = ledger["world_model"]["hypotheses"]
    return {
        "schema_version": ledger["schema_version"],
        "objective": copy.deepcopy(ledger["objective"]),
        "scope": copy.deepcopy(ledger["scope"]),
        "current_level": int(current_level),
        "interface": interface_view,
        "recent_attempts": copy.deepcopy(ledger["execution"]["attempts"][-8:]),
        "open_hypotheses": copy.deepcopy(
            [item for item in hypotheses if item.get("status") == "open"][-8:]
        ),
        "recent_verifications": copy.deepcopy(ledger["verification"]["records"][-8:]),
        "unknowns": copy.deepcopy(ledger["coverage"]["unknowns"][-12:]),
        "governance": copy.deepcopy(ledger["governance"]["events"][-8:]),
        "validation": ledger_validation_report(ledger),
    }


def validate_ledger(value: Any) -> list[str]:
    """Validate semantic invariants without deleting or coercing evidence."""

    if not isinstance(value, dict):
        return ["ledger must be an object"]
    errors: list[str] = []
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {LEDGER_SCHEMA_VERSION}")
    if (value.get("objective") or {}).get("type") != "complete_game":
        errors.append("objective.type must equal complete_game")
    actions = ((value.get("interface") or {}).get("actions") or {})
    for action_id in CANONICAL_ACTIONS:
        if action_id not in actions:
            errors.append(f"interface action {action_id} is missing")

    execution = value.get("execution") or {}
    attempts = execution.get("attempts") or []
    attempts_by_id: dict[str, dict[str, Any]] = {}
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            errors.append(f"execution.attempts[{index}] must be an object")
            continue
        attempt_id = str(attempt.get("id") or "")
        if not attempt_id:
            errors.append(f"execution.attempts[{index}].id is missing")
        else:
            attempts_by_id[attempt_id] = attempt
        outcome = attempt.get("outcome")
        if outcome not in OUTCOME_TYPES:
            errors.append(f"attempt {attempt_id or index} has invalid outcome {outcome!r}")
        action_id = attempt.get("action_id")
        if action_id is not None and action_id not in CANONICAL_ACTIONS:
            errors.append(f"attempt {attempt_id or index} has non-canonical action_id {action_id!r}")
        executed = bool(attempt.get("executed"))
        if executed != str(outcome or "").startswith("executed_"):
            errors.append(f"attempt {attempt_id or index} execution flag conflicts with outcome")
        if not executed and attempt.get("observation") is not None:
            errors.append(f"unexecuted attempt {attempt_id or index} must not contain observation evidence")

    for index, transition in enumerate(execution.get("transitions") or []):
        if not isinstance(transition, dict):
            errors.append(f"execution.transitions[{index}] must be an object")
            continue
        attempt_id = str(transition.get("attempt_id") or "")
        attempt = attempts_by_id.get(attempt_id)
        if attempt is None:
            errors.append(f"transition {index} references unknown attempt {attempt_id!r}")
        elif not attempt.get("executed"):
            errors.append(f"transition {index} references unexecuted attempt {attempt_id}")

    hypotheses = ((value.get("world_model") or {}).get("hypotheses") or [])
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            errors.append(f"world_model.hypotheses[{index}] must be an object")
            continue
        verification_status = hypothesis.get("verification_status")
        if verification_status not in VERIFICATION_STATUSES:
            errors.append(
                f"hypothesis {hypothesis.get('id', index)} has invalid verification_status {verification_status!r}"
            )
    return errors


def ledger_validation_report(value: Any) -> dict[str, Any]:
    errors = validate_ledger(value)
    return {"valid": not errors, "errors": errors[:32], "error_count": len(errors)}
