"""Custom MCP tools for querying the Prediction database."""

from claude_agent_sdk import tool
from typing import Any


@tool("get_prediction_stats", "Get aggregate statistics for predictions", {
    "league": str,
    "resolved_only": bool,
})
async def get_prediction_stats(args: dict[str, Any]) -> dict[str, Any]:
    """Get aggregate statistics: total, correct, accuracy, by-league breakdown."""
    from apps.home.models import Prediction
    from django.db.models import Avg, Count, Q, F

    league = args.get('league')
    resolved_only = args.get('resolved_only', True)

    qs = Prediction.objects.all()
    if resolved_only:
        qs = qs.filter(resolved=True)
    if league:
        qs = qs.filter(div=league)

    total = qs.count()
    correct = qs.filter(pred_ftr=F('actual_ftr')).count()
    accuracy = round(correct / total * 100, 1) if total > 0 else 0
    avg_value = qs.aggregate(avg=Avg('max_value'))['avg'] or 0

    by_league = list(qs.values('div').annotate(
        total=Count('id'),
        correct=Count('id', filter=Q(pred_ftr=F('actual_ftr'))),
    ).order_by('-total'))

    result = (
        f"Prediction Stats:\n"
        f"  Total: {total}\n"
        f"  Correct: {correct}\n"
        f"  Accuracy: {accuracy}%\n"
        f"  Avg Max Value: {round(avg_value, 4)}\n"
        f"  By League:\n"
    )
    for lg in by_league:
        lg_acc = round(lg['correct'] / lg['total'] * 100, 1) if lg['total'] else 0
        result += f"    {lg['div']}: {lg['correct']}/{lg['total']} ({lg_acc}%)\n"

    return {"content": [{"type": "text", "text": result}]}


@tool("get_recent_predictions", "Get the N most recent predictions from the database", {
    "n": int,
})
async def get_recent_predictions(args: dict[str, Any]) -> dict[str, Any]:
    """Get recent predictions with key fields."""
    from apps.home.models import Prediction

    n = min(args.get('n', 20), 100)

    predictions = list(
        Prediction.objects.all()
        .order_by('-date', '-created_at')[:n]
        .values(
            'div', 'date', 'home_team', 'away_team',
            'h_win', 'draw', 'a_win', 'pred_ftr',
            'max_value', 'max_value_result',
            'actual_ftr', 'resolved',
        )
    )

    lines = ["Recent Predictions:"]
    for p in predictions:
        status = "RESOLVED" if p['resolved'] else "PENDING"
        correct = ""
        if p['resolved']:
            correct = " CORRECT" if p['pred_ftr'] == p['actual_ftr'] else " WRONG"
        lines.append(
            f"  {p['date']} {p['div']} {p['home_team']} vs {p['away_team']} "
            f"| Pred: {p['pred_ftr']} Value: {p['max_value']:.4f} "
            f"| {status}{correct}"
        )

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("get_team_performance", "Get prediction performance for a specific team", {
    "team": str,
})
async def get_team_performance(args: dict[str, Any]) -> dict[str, Any]:
    """Get home/away accuracy, value bet record, recent results for a team."""
    from apps.home.models import Prediction
    from django.db.models import Q, F

    team = args['team']

    home_qs = Prediction.objects.filter(resolved=True, home_team=team)
    away_qs = Prediction.objects.filter(resolved=True, away_team=team)

    home_total = home_qs.count()
    home_correct = home_qs.filter(pred_ftr=F('actual_ftr')).count()
    away_total = away_qs.count()
    away_correct = away_qs.filter(pred_ftr=F('actual_ftr')).count()

    recent = list(
        Prediction.objects.filter(
            Q(home_team=team) | Q(away_team=team), resolved=True
        ).order_by('-date')[:10].values(
            'date', 'home_team', 'away_team', 'pred_ftr', 'actual_ftr', 'max_value'
        )
    )

    home_acc = round(home_correct / home_total * 100, 1) if home_total else 0
    away_acc = round(away_correct / away_total * 100, 1) if away_total else 0
    total = home_total + away_total
    total_correct = home_correct + away_correct
    overall_acc = round(total_correct / total * 100, 1) if total else 0

    lines = [
        f"Team Performance: {team}",
        f"  Overall: {total_correct}/{total} ({overall_acc}%)",
        f"  Home: {home_correct}/{home_total} ({home_acc}%)",
        f"  Away: {away_correct}/{away_total} ({away_acc}%)",
        f"  Recent matches:",
    ]
    for r in recent:
        correct = "CORRECT" if r['pred_ftr'] == r['actual_ftr'] else "WRONG"
        lines.append(
            f"    {r['date']} {r['home_team']} vs {r['away_team']} "
            f"Pred:{r['pred_ftr']} Actual:{r['actual_ftr']} {correct}"
        )

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
