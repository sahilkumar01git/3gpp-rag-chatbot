from app.verification.claims import clean_claim, split_into_claims
from app.verification.nli_verifier import _premises, _table_rows


def test_clean_claim_strips_markdown_and_citation_markers():
    assert clean_claim("The default value of timer **T300** is **1000 ms**【1】【2】") == (
        "The default value of timer T300 is 1000 ms"
    )
    assert clean_claim("The states are **RM‑DEREGISTERED** and `RM‑REGISTERED` [1].") == (
        "The states are RM-DEREGISTERED and RM-REGISTERED."
    )


def test_split_into_claims_returns_cleaned_sentences():
    claims = split_into_claims("K_SEAF is derived from **K_AUSF**. [1] It is an anchor key. [2]")
    assert claims == ["K_SEAF is derived from K_AUSF.", "It is an anchor key."]


def test_table_rows_render_one_statement_per_row():
    table = "| Timer | Default value |\n| --- | --- |\n| T300 | 1000 ms |\n| T310 | 2000 ms |"
    assert _table_rows(table) == ["Timer: T300; Default value: 1000 ms.", "Timer: T310; Default value: 2000 ms."]
    assert _premises(table) == _table_rows(table)


def test_premises_for_prose_include_whole_passage_and_each_sentence():
    text = "The MAC entity supports HARQ. Each process has a buffer."
    assert _premises(text) == [text, "The MAC entity supports HARQ.", "Each process has a buffer."]
    assert _premises("Single sentence only.") == ["Single sentence only."]
