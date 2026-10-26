"""Per-user monthly spending limit for model calls (AI20K D12 lab, ``cost_guard.py``).

Rate limits cap the number of requests; this caps money: ten requests a minute
with 50k tokens each still empties a budget. Spend is computed from the token
usage the provider reports on every model reply in a turn, priced with your
own table, and kept per user per calendar month (UTC) in the cache service
(Valkey when configured, otherwise process memory).
"""

import json
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Optional,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from app.core.cache import cache_service
from app.core.config import settings
from app.core.logging import logger

KEY_TTL_SECONDS = 40 * 24 * 3600  # keep last month's total around for reconciliation


class BudgetExceededError(Exception):
    """The user's monthly budget is used up."""


class CostGuard:
    """Checks and records spend against a monthly per-user budget."""

    def __init__(
        self,
        cache: Any,
        monthly_budget_usd: float,
        prices_per_million: dict[str, tuple[float, float]],
        default_price: tuple[float, float] = (0.0, 0.0),
    ):
        """Configure the guard.

        Args:
            cache: Object with async ``get(key)`` and ``set(key, value, ttl)``.
            monthly_budget_usd: Limit per user per month; 0 disables the guard.
            prices_per_million: ``{model_name: (input_usd, output_usd)}`` per million tokens.
            default_price: Used for models missing from the table.
        """
        self._cache = cache
        self.budget = monthly_budget_usd
        self._prices = prices_per_million
        self._default = default_price
        self._warned: set[str] = set()

    @property
    def enabled(self) -> bool:
        """Whether a budget is configured."""
        return self.budget > 0

    @staticmethod
    def _key(user_id: str, month: Optional[str] = None) -> str:
        return f"cost:{user_id}:{month or datetime.now(timezone.utc).strftime('%Y-%m')}"

    async def spent(self, user_id: str) -> float:
        """USD spent by ``user_id`` this month."""
        value = await self._cache.get(self._key(user_id))
        return float(value) if value is not None else 0.0

    async def check(self, user_id: str) -> None:
        """Raise ``BudgetExceededError`` when the user has nothing left this month."""
        if self.enabled and await self.spent(user_id) >= self.budget:
            logger.warning("cost_budget_exceeded", user_id=user_id, budget_usd=self.budget)
            raise BudgetExceededError(f"monthly budget of ${self.budget:.2f} reached")

    def _price(self, model: str) -> tuple[float, float]:
        if model in self._prices:
            return self._prices[model]
        if model not in self._warned:
            self._warned.add(model)
            logger.warning("cost_price_missing", model=model, using=self._default)
        return self._default

    def cost_of_turn(self, messages: list[BaseMessage]) -> float:
        """USD cost of the model replies after the latest user message."""
        total = 0.0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            usage = getattr(message, "usage_metadata", None) if isinstance(message, AIMessage) else None
            if not usage:
                continue
            model = str((message.response_metadata or {}).get("model_name", "unknown"))
            price_in, price_out = self._price(model)
            total += usage.get("input_tokens", 0) * price_in / 1e6 + usage.get("output_tokens", 0) * price_out / 1e6
        return total

    async def record(self, user_id: str, messages: list[BaseMessage]) -> float:
        """Add this turn's cost to the user's monthly total; returns the cost.

        Read-then-write, so concurrent turns of one user can undercount slightly;
        the guard is a budget brake, not an invoice.
        """
        if not self.enabled:
            return 0.0
        cost = self.cost_of_turn(messages)
        if cost > 0:
            key = self._key(user_id)
            current = float(await self._cache.get(key) or 0.0)
            await self._cache.set(key, str(current + cost), ttl=KEY_TTL_SECONDS)
        return cost


def parse_prices(raw: str) -> dict[str, tuple[float, float]]:
    """``'{"gpt-5.4-mini": [0.25, 2.0]}'`` -> ``{"gpt-5.4-mini": (0.25, 2.0)}`` (USD per million tokens)."""
    if not raw.strip():
        return {}
    data = json.loads(raw)
    return {str(model): (float(p[0]), float(p[1])) for model, p in data.items()}


_guard: Optional[CostGuard] = None


def get_cost_guard() -> CostGuard:
    """Process-wide guard built from the COST_* settings."""
    global _guard
    if _guard is None:
        _guard = CostGuard(
            cache_service,
            settings.COST_MONTHLY_BUDGET_USD,
            parse_prices(settings.COST_PRICES_JSON),
            (settings.COST_DEFAULT_INPUT_PER_1M, settings.COST_DEFAULT_OUTPUT_PER_1M),
        )
    return _guard


def set_cost_guard(guard: Optional[CostGuard]) -> None:
    """Replace the process-wide guard (used by tests)."""
    global _guard
    _guard = guard
