from __future__ import annotations

import unittest

from inference.agent.lifecycle_events import LifecycleEvent, LifecycleEventBus, LifecycleEventType, LifecycleOutcome


class LifecycleEventBusTests(unittest.TestCase):
    def test_failure_is_reported_but_never_blocks_following_handler(self) -> None:
        bus = LifecycleEventBus()
        bus.register(lambda _event: (_ for _ in ()).throw(RuntimeError("memory unavailable")))
        bus.register(lambda _event: LifecycleOutcome(data={"ok": True}))
        outcome = bus.emit(LifecycleEvent(LifecycleEventType.GAME_START, "g", 1, 0))
        self.assertTrue(outcome.allow)
        self.assertTrue(outcome.data["ok"])
        self.assertIn("memory unavailable", outcome.errors[0])

    def test_unregister_is_idempotent(self) -> None:
        bus = LifecycleEventBus()
        calls: list[int] = []
        unregister = bus.register(lambda _event: calls.append(1))
        unregister(); unregister()
        bus.emit(LifecycleEvent(LifecycleEventType.GAME_END, "g", 1, 1))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
