"""
Ранжирование sweep-конфигов по выбранной метрике.

Идея LCB-ранжирования:
  Для каждого конфига имеем mean метрики и её std (через bootstrap).
  LCB = mean - z * std — пессимистическая оценка («насколько уверены что хотя бы такой»).
  Сортируем по LCB убывающе → топ это «самый надёжный лучший».

Для метрик «больше = лучше» (skill_mae, directional_acc) — прямое использование.
Для метрик «меньше = лучше» (pinball_mean) — инвертируем знак:
  ranking_value = -metric
  ranking_lcb   = -UCB(metric) = -(mean + z*std) = -mean - z*std

Это даёт согласованную семантику: «больший ranking_lcb = лучше» для всех метрик.
"""
from __future__ import annotations


# Метрики где меньше = лучше (нужна инверсия для ранжирования)
_LOWER_IS_BETTER = {"pinball_mean", "pinball_q50", "winkler_q25_q75", "winkler_q10_q90"}


def ranking_value(metric_name: str, mean: float) -> float:
    """Значение метрики в шкале «больше = лучше»."""
    return -mean if metric_name in _LOWER_IS_BETTER else mean


def ranking_lcb(metric_name: str, mean: float, ci_low: float, ci_high: float) -> float:
    """
    LCB в шкале «больше = лучше».

    Для «больше=лучше»: LCB = ci_low (нижняя граница CI).
    Для «меньше=лучше»: инвертируем; ранжирующий LCB = -ci_high (нижняя граница
    инвертированной величины = верхняя граница оригинала, со знаком минус).
    """
    if metric_name in _LOWER_IS_BETTER:
        return -ci_high
    return ci_low


def overlapping_with_top(
        configs: list[dict],
        ranking_metric: str,
        lcb_tie_tolerance: float,
        max_candidates: int,
) -> list[dict]:
    """
    Возвращает список конфигов, чьи LCB перекрываются с top-1 в пределах tolerance.

    Логика: top-1 имеет ranking_lcb_max. Включаем все configs где
        ranking_lcb >= ranking_lcb_max - lcb_tie_tolerance
    То есть «не намного хуже лидера». Эти конфиги — кандидаты на CV.

    configs должны содержать ключи 'ranking_metric_lcb' и 'train_window_size'.
    """
    if not configs:
        return []
    sorted_configs = sorted(configs, key=lambda c: c["ranking_metric_lcb"], reverse=True)
    top_lcb = sorted_configs[0]["ranking_metric_lcb"]
    candidates = [
        c for c in sorted_configs
        if (top_lcb - c["ranking_metric_lcb"]) <= lcb_tie_tolerance
    ]
    return candidates[:max_candidates]


def select_recommended(
        configs: list[dict],
        ranking_metric: str,
        lcb_tie_tolerance: float,
) -> tuple[dict, str, float]:
    """
    Финальный выбор лучшего конфига.

    1. Если у конфигов есть CV-LCB — берём cv_ranking_metric_lcb, иначе primary ranking_metric_lcb.
    2. Сортируем по выбранному LCB убывающе.
    3. Среди тех, что в пределах tolerance от top — выбираем минимальный train_window_size.

    Возвращает (recommended_config, reason, lcb_margin_over_runner_up).
    """
    if not configs:
        raise ValueError("Нет конфигов для выбора recommended.")

    def effective_lcb(c: dict) -> float:
        cv = c.get("cv_ranking_metric_lcb")
        return cv if cv is not None else c["ranking_metric_lcb"]

    sorted_configs = sorted(configs, key=effective_lcb, reverse=True)
    top = sorted_configs[0]
    top_lcb = effective_lcb(top)

    # Tied: все в пределах tolerance
    tied = [c for c in sorted_configs if (top_lcb - effective_lcb(c)) <= lcb_tie_tolerance]
    if len(tied) == 1:
        # Чёткий лидер
        margin = (top_lcb - effective_lcb(sorted_configs[1])) if len(sorted_configs) > 1 else 0.0
        has_cv = any(c.get("cv_ranking_metric_lcb") is not None for c in configs)
        reason = "highest_cv_lcb" if has_cv else "no_cv_fallback_to_primary"
        return top, reason, margin

    # Есть ничья — берём минимальный контекст
    chosen = min(tied, key=lambda c: c["train_window_size"])
    # margin относительно следующего за tied
    rest = [c for c in sorted_configs if c not in tied]
    margin = (effective_lcb(chosen) - effective_lcb(rest[0])) if rest else 0.0
    return chosen, "tied_lcb_smaller_context", margin
