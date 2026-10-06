from recovery_router.retrieval import TFIDFRetriever, attach_candidates, make_intervention


def example_index():
    rows = [{"id": "card1", "dataset": "clinc150", "text": "my bank card is lost", "label": "lost_card"},
            {"id": "card2", "dataset": "clinc150", "text": "I lost my debit card", "label": "lost_card"}]
    for i in range(24):
        rows.append({"id": f"other{i}", "dataset": "clinc150", "text": f"other service number {i}", "label": f"other_{i}"})
    return rows


def test_retriever_collapses_duplicates_and_nested_sets_are_explicit():
    retriever = TFIDFRetriever(example_index())
    ranked = retriever.retrieve("lost my card", k=20)
    assert ranked[0]["label"] == "lost_card"
    assert len(ranked) == len({c["label"] for c in ranked}) == 20
    query = {"id": "q1", "parent_id": "q1", "text": "lost my card", "label": "lost_card", "is_oos": False}
    row = attach_candidates(query, ranked, seed=42, first_k=5, expanded_k=20)
    assert {c["id"] for c in row["candidates"]} <= {c["id"] for c in row["expanded_candidates"]}
    assert row["retrieval_situation"] == "present_initially"
    assert row == attach_candidates(query, ranked, seed=42, first_k=5, expanded_k=20)
    assert {c["id"] for c in row["candidates"]} == {c["id"] for c in ranked[:5]}


def test_controlled_omission_does_not_inject_expanded_gold():
    ranked = TFIDFRetriever(example_index()).retrieve("lost my card", k=20)
    query = {"id": "q1", "parent_id": "q1", "text": "lost my card", "label": "lost_card", "is_oos": False}
    row = attach_candidates(query, ranked, first_k=5, expanded_k=20)
    changed = make_intervention(row, "drop_gold")
    assert changed["intervention"] == "drop_gold"
    assert changed["parent_id"] == row["id"]
    assert "lost_card" not in {c["label"] for c in changed["candidates"]}
    assert {c["id"] for c in changed["expanded_candidates"]} == {c["id"] for c in row["expanded_candidates"]}
    assert changed["retrieval_situation"] == "recoverable_after_expansion"
    assert len(changed["candidates"]) == 5
    absent = dict(row, label="unretrieved_label")
    assert make_intervention(absent, "drop_gold") is None
    assert make_intervention(dict(row, label="oos", is_oos=True), "drop_gold") is None


def test_shuffle_changes_order_only():
    ranked = TFIDFRetriever(example_index()).retrieve("lost my card", k=20)
    row = attach_candidates({"id": "q1", "parent_id": "q1", "label": "lost_card", "is_oos": False}, ranked)
    changed = make_intervention(row, "shuffle")
    assert changed["intervention"] == "shuffle"
    for key in ("candidates", "expanded_candidates"):
        assert {c["id"]: c for c in changed[key]} == {c["id"]: c for c in row[key]}


def test_primary_top_one_profile_and_omission_keep_nonempty_nested_candidates():
    ranked = TFIDFRetriever(example_index()).retrieve("lost my card", k=20)
    row = attach_candidates({"id": "q1", "parent_id": "q1", "label": "lost_card", "is_oos": False}, ranked)
    assert len(row["candidates"]) == 1
    assert len(row["expanded_candidates"]) == 5
    assert row["retrieval_situation"] == "present_initially"
    changed = make_intervention(row, "drop_gold")
    assert len(changed["candidates"]) == 1
    assert changed["candidates"][0]["label"] != "lost_card"
    assert {c["id"] for c in changed["candidates"]} <= {c["id"] for c in changed["expanded_candidates"]}
    assert changed["retrieval_situation"] == "recoverable_after_expansion"


def test_generic_recovery_cases_respect_configured_cutoffs():
    ranked = TFIDFRetriever(example_index()).retrieve("lost my card", k=20)
    base = {"id": "q1", "parent_id": "q1", "is_oos": False}
    assert attach_candidates(dict(base, label=ranked[3]["label"]), ranked)["retrieval_situation"] == "recoverable_after_expansion"
    assert attach_candidates(dict(base, label=ranked[6]["label"]), ranked)["retrieval_situation"] == "missing_after_expansion"
    assert attach_candidates(dict(base, label="oos", is_oos=True), ranked)["retrieval_situation"] == "true_oos"
