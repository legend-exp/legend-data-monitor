"""Compatibility shim (Phase 4): moved to :mod:`legend_data_monitor.contract.issues`."""

from .contract.issues import (  # noqa: F401
    Excursion,
    Issue,
    classify_severity,
    clear_details,
    collapse_correlated,
    evaluate_excursion,
    format_issue_block,
    issues_file_path,
    pop_detail,
    record_detail,
    write_issues,
)
