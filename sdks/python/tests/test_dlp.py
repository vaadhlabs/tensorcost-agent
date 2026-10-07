from tensorcost._dlp import scan_text_for_compliance

HIPAA_SNAP = {
    "frameworks": ["hipaa"],
    "mode": "enforce",
    "dlp": {"enabled": True, "detectors": ["phi_ssn"], "action": "refuse"},
}


def test_refuses_ssn_in_enforce_mode():
    result = scan_text_for_compliance("patient ssn 123-45-6789", HIPAA_SNAP)
    assert result.outcome == "refused"
    assert "phi_ssn" in result.matched_detectors


def test_none_when_no_match():
    result = scan_text_for_compliance("hello", HIPAA_SNAP)
    assert result.outcome == "none"
