"""Shared comparison baselines for Lazarus benchmark reports."""

COMPARISON_BASELINES = [
    {
        "system": "GPT-5.5 (API)",
        "benchmark": "nvidia/RULER (Variable Tracking)",
        "token_bin": "128k",
        "metric_name": "accuracy",
        "score": 0.865,
        "source": "NVIDIA RULER Leaderboard",
    },
    {
        "system": "Claude 4.7 Opus",
        "benchmark": "nvidia/RULER (Variable Tracking)",
        "token_bin": "128k",
        "metric_name": "accuracy",
        "score": 0.892,
        "source": "Anthropic Tech Report",
    },
    {
        "system": "Claude 4.7 Opus",
        "benchmark": "Salesforce/LoCoBench",
        "token_bin": "1M",
        "metric_name": "pass_at_1",
        "score": 0.624,
        "source": "Salesforce AI Research Leaderboard",
    },
]
